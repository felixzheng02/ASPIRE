"""Compact artifact recorder for one Code-as-Policy episode."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import contextlib
import contextvars
import dataclasses
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import time
from typing import Any

import imageio.v2 as imageio
import numpy as np

from cap_harness.contracts import (
    CameraObservation,
    GraspSet,
    Observation,
    PointCloud,
    RobotAction,
    Segmentation,
    SegmentationSet,
    StepResult,
)
from cap_harness.geometry import invert_transform, mask_to_point_cloud

SCHEMA_VERSION = "1.0"
TERMINATION_REASONS = {
    "program_completed": "The generated Python program returned normally.",
    "task_succeeded": "The environment ended after benchmark success.",
    "environment_terminated": "The environment terminated without benchmark success.",
    "environment_truncated": "The environment reached its horizon.",
    "step_limit": "The configured simulator-step limit was reached.",
    "program_error": "The generated program was rejected or raised an exception.",
    "provider_error": "An unrecoverable provider failure stopped execution.",
    "harness_error": "The adapter or artifact recorder failed.",
    "user_interrupt": "Execution received a user interrupt.",
}
#: zlib effort for recorded PNGs. Every level is lossless -- the level buys
#: file size, not fidelity -- and the default (6) is the wrong trade here: on a
#: measured 800x512 keyframe it costs 136 ms against 33 ms at level 1, to save
#: 57 kB of a 492 kB file. Keyframes are written twice per motion call, so that
#: default was ~16% of a recorded episode's wall clock.
PNG_COMPRESS_LEVEL = 1
_SCENE3D_SIZE = 480
_SCENE3D_MARGIN = 28
_MAX_SCENE_POINTS = 5000
_MAX_OBJECT_POINTS = 2500
_OBJECT_COLORS = (
    (255, 64, 64),
    (64, 255, 64),
    (64, 128, 255),
    (255, 192, 64),
    (224, 64, 255),
)
_PARENT: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_parent", default=None)
_SECRET_KEY = re.compile(r"authorization|cookie|credential|password|secret|token", re.IGNORECASE)
_SECRET_TEXT = re.compile(r"(?i)\b(?:bearer\s+\S+|hf_[A-Za-z0-9]{12,})")


class StepLimitReached(RuntimeError):
    """A run tried to exceed its global simulator-step limit."""


class RunTiming:
    """Disjoint wall-clock attribution for one run, reported in ``outcome.json``.

    Every bucket is a leaf: :meth:`measure` charges a region only the time not
    already charged to something nested inside it, so ``reset`` excludes the
    keyframe it writes and ``record_step`` excludes the video frame it encodes.
    What no bucket claims is the generated program's own Python plus harness
    glue, reported as ``unattributed_s`` rather than silently folded into a
    neighbour.
    """

    def __init__(self) -> None:
        self.buckets: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self._attributed = 0.0

    def add(self, bucket: str, seconds: float) -> None:
        self.buckets[bucket] = self.buckets.get(bucket, 0.0) + seconds
        self.counts[bucket] = self.counts.get(bucket, 0) + 1
        self._attributed += seconds

    @contextlib.contextmanager
    def measure(self, bucket: str) -> Iterator[None]:
        started, attributed_at_start = time.monotonic(), self._attributed
        try:
            yield
        finally:
            elapsed = time.monotonic() - started
            nested = self._attributed - attributed_at_start
            self.add(bucket, max(elapsed - nested, 0.0))

    def snapshot(self, *, wall_s: float) -> Mapping[str, object]:
        return {
            "wall_s": round(wall_s, 6),
            "attributed_s": round(self._attributed, 6),
            # Program Python and harness glue: no bucket claims it.
            "unattributed_s": round(max(wall_s - self._attributed, 0.0), 6),
            "buckets": {
                name: {"seconds": round(seconds, 6), "calls": self.counts[name]}
                for name, seconds in sorted(self.buckets.items(), key=lambda item: -item[1])
            },
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def slug(value: object) -> str:
    return (re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-") or "unnamed")[:96]


def _component(value: object) -> str:
    return (re.sub(r"[^a-z0-9_.-]+", "-", str(value).lower()).strip("-.") or "unnamed")[:96]


def _downsample_points(
    points: np.ndarray, colors: np.ndarray | None, limit: int
) -> tuple[np.ndarray, np.ndarray | None]:
    count = int(points.shape[0])
    if count <= limit:
        return points, colors
    stride = max(1, count // limit)
    sampled = points[::stride]
    sampled_colors = None if colors is None else colors[::stride]
    return sampled, sampled_colors


def _view_uv(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Isometric-style orthographic view: u right, v up, depth toward the camera."""
    azim = np.deg2rad(-60.0)
    elev = np.deg2rad(28.0)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    u = np.cos(azim) * x + np.sin(azim) * y
    v = (
        -np.sin(azim) * np.sin(elev) * x
        + np.cos(azim) * np.sin(elev) * y
        + np.cos(elev) * z
    )
    depth = (
        np.sin(azim) * np.cos(elev) * x
        - np.cos(azim) * np.cos(elev) * y
        + np.sin(elev) * z
    )
    return u, v, depth


def _rgb8(colors: np.ndarray) -> np.ndarray:
    values = np.asarray(colors)
    if np.issubdtype(values.dtype, np.floating):
        values = np.rint(values * 255.0)
    return np.clip(values, 0, 255).astype(np.uint8)


def redact(value: object) -> object:
    if isinstance(value, str):
        return _SECRET_TEXT.sub("[REDACTED]", value)
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if _SECRET_KEY.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [redact(item) for item in value]
    return value


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(redact(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Serializer:
    """Readable typed JSON with exact NumPy arrays stored beside it."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def dump(self, path: Path, value: object) -> None:
        atomic_json(path, self.encode(value, path.parent / "arrays"))

    def encode(self, value: object, arrays: Path) -> object:
        if value is None or isinstance(value, bool | int | str):
            return redact(value)
        if isinstance(value, float | np.floating):
            number = float(value)
            return number if np.isfinite(number) else {"$float": str(number)}
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, PointCloud):
            return {
                "$type": "PointCloud",
                "frame": value.frame,
                "point_count": len(value.points),
                "has_colors": value.colors is not None,
            }
        if isinstance(value, CameraObservation):
            return {
                "$type": "CameraObservation",
                "frame": value.frame,
                "rgb_shape": list(value.rgb.shape),
                "depth_shape": list(value.depth_m.shape),
                "intrinsics": value.intrinsics.tolist(),
                "camera_pose": self.encode(value.camera_pose, arrays),
                "timestamp_s": value.timestamp_s,
            }
        if isinstance(value, Observation):
            return {
                "$type": "Observation",
                "cameras": {
                    name: self.encode(camera, arrays) for name, camera in value.cameras.items()
                },
                "robot_state": self.encode(value.robot_state, arrays),
                "task_context": self.encode(value.task_context, arrays),
                "timestamp_s": value.timestamp_s,
            }
        if isinstance(value, Segmentation):
            return {
                "$type": "Segmentation",
                "label": value.label,
                "score": value.score,
                "camera_name": value.camera_name,
                "frame": value.frame,
                "mask_shape": list(value.mask.shape),
                "box_xyxy": value.box_xyxy.tolist() if value.box_xyxy is not None else None,
            }
        if isinstance(value, SegmentationSet):
            return {
                "$type": "SegmentationSet",
                "ok": value.ok,
                "segmentations": [self.encode(item, arrays) for item in value.segmentations[:5]],
                "error": self.encode(value.error, arrays),
                "diagnostics": self.encode(value.diagnostics, arrays),
            }
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            if array.dtype == np.bool_ and array.ndim == 2:
                return {
                    "$type": "MaskSummary",
                    "shape": list(array.shape),
                    "true_pixels": int(np.count_nonzero(array)),
                }
            if array.size <= 64:
                return {
                    "$array_inline": array.tolist(),
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                }
            identity = hashlib.sha256()
            identity.update(array.dtype.str.encode())
            identity.update(str(array.shape).encode())
            identity.update(array.tobytes())
            path = self.root / "payloads" / f"{identity.hexdigest()}.npy"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, array, allow_pickle=False)
            return {
                "$array": str(path.relative_to(self.root)),
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": _sha256(path),
            }
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                "$type": type(value).__name__,
                **{
                    field.name: self.encode(getattr(value, field.name), arrays)
                    for field in dataclasses.fields(value)
                },
            }
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {
                str(key): (
                    "[REDACTED]" if _SECRET_KEY.search(str(key)) else self.encode(item, arrays)
                )
                for key, item in value.items()
            }
        if isinstance(value, list | tuple | set | frozenset):
            return [self.encode(item, arrays) for item in value]
        return {"$type": type(value).__name__, "repr": redact(repr(value)[:512])}


class _Span:
    def __init__(
        self,
        path: Path,
        serializer: Serializer,
        state: dict[str, bool],
        recorder: RunRecorder,
        *,
        capture_segmentation: bool,
        capture_grasps: bool,
    ) -> None:
        self.path, self.serializer, self.state = path, serializer, state
        self.recorder = recorder
        self.capture_segmentation = capture_segmentation
        self.capture_grasps = capture_grasps

    def output(self, value: object, *, ok: bool = True) -> object:
        with self.recorder.timing.measure("trace_write"):
            self.serializer.dump(self.path / "output.json", value)
        if self.capture_segmentation and isinstance(value, SegmentationSet):
            self.recorder.segmentation_overlay(value)
        if self.capture_grasps and isinstance(value, GraspSet):
            self.recorder.grasp_overlay(value)
        self.state["ok"] = ok
        return value


class RunRecorder:
    """Write one immutable, self-contained run directory."""

    def __init__(
        self,
        *,
        output_root: Path,
        benchmark: str,
        suite: str,
        task_id: int,
        task_name: str,
        seed: int,
        max_steps: int,
        camera_width: int = 128,
        camera_height: int = 128,
        control_frequency_hz: float = 20.0,
        providers: Mapping[str, object] | None = None,
        init_mode: str = "saved",
        flat_layout: bool = False,
        capture_videos: bool = True,
        step_video: bool = True,
        video_frame_stride: int = 1,
    ) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.run_id = f"run-{stamp}-{secrets.token_hex(4)}"
        if flat_layout:
            # Task identity comes from the caller-chosen output_root (e.g. a
            # task-scoped directory); only seed and run id are nested.
            self.root = output_root / f"{seed:04d}" / self.run_id
        else:
            self.root = (
                output_root
                / _component(benchmark)
                / _component(suite)
                / f"{task_id:02d}-{slug(task_name)}"
                / f"{seed:04d}"
                / self.run_id
            )
        self.capture_videos = capture_videos
        #: Whether a recorded step contributes a VIDEO FRAME.
        #:
        #: True for embodiments that step at the control rate -- a simulator's
        #: per-step frame is a real video. False where one recorded step is a
        #: whole commanded motion: on the real YAM a 2.8 s descent is a single
        #: ``execute_trajectory``, so per-step frames produced a 10-frame,
        #: half-second "video" of a run lasting minutes. Those embodiments write
        #: media/videos from a sampler running on its own clock, and each step
        #: is captured here as a KEYFRAME instead -- same images, filed as the
        #: stills they always were rather than assembled into a misleading movie.
        self.step_video = bool(step_video)
        # Record every Nth step as a video frame; slow simulators at 30 Hz use 3 (10 fps).
        self.video_frame_stride = max(1, int(video_frame_stride))
        self.root.mkdir(parents=True)
        for name in (
            "source",
            "episode",
            "evaluation",
            "trace/calls",
            "trace/model",
            "media/videos",
            "media/keyframes",
            "logs",
        ):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        (self.root / "trace/model/turns.jsonl").touch()
        self.serializer = Serializer(self.root)
        self.max_steps, self.step_count = max_steps, 0
        self.frequency = float(control_frequency_hz)
        self.cumulative_reward = 0.0
        self.terminated = self.truncated = False
        self.terminal_reason: str | None = None
        self.started_at, self.started = _now(), time.monotonic()
        self.timing = RunTiming()
        #: Set when the adapter is asked to step, cleared when it answers. The
        #: gap is the simulator step plus its observation normalization -- the
        #: one cost the recorder can attribute without reaching into an adapter.
        self._step_entered: float | None = None
        self.events = (self.root / "trace/events.jsonl").open("a", encoding="utf-8")
        self.steps = (self.root / "episode/steps.jsonl").open("a", encoding="utf-8")
        self.writers: dict[str, Any] = {}
        self.video_frames: dict[str, int] = {}
        self.last_observation: Observation | None = None
        self.last_video_step: int | None = None
        self.depth_saved = False
        self.overlay_index = 0
        self.grasp_overlay_index = 0
        self.scene3d_overlay_index = 0
        self.last_segmentations: tuple[Segmentation, ...] = ()
        self.event_index = self.call_index = self.keyframe_index = 0
        self.finalized = False
        self.identity = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": "in_progress",
            "benchmark": benchmark,
            "suite": suite,
            "task_id": task_id,
            "task_name": task_name,
            "seed": seed,
            "started_at": self.started_at,
            "capture": {
                "profile": "balanced_evidence" if capture_videos else "no_videos",
                "max_steps": max_steps,
                "camera_width": camera_width,
                "camera_height": camera_height,
                "videos": capture_videos,
            },
            "init_mode": init_mode,
            "providers": dict(providers or {}),
            "environment_sha256": os.environ.get("CAP_HARNESS_ENVIRONMENT_SHA256"),
        }
        atomic_json(self.root / "run.json", self.identity)

    def save_program(self, path: Path) -> None:
        raw = path.read_bytes()
        (self.root / "source/program.py").write_bytes(raw)
        dependency_lock = Path(__file__).resolve().parents[2] / "configs/dependency-lock.json"
        lock_raw = dependency_lock.read_bytes() if dependency_lock.is_file() else b""
        if lock_raw:
            (self.root / "source/dependency-lock.json").write_bytes(lock_raw)
        atomic_json(
            self.root / "source/provenance.json",
            {
                "generation_source": "program_file",
                "input_path": str(path.resolve()),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "harness_git": self._git(),
                "dependency_lock_sha256": (
                    hashlib.sha256(lock_raw).hexdigest() if lock_raw else None
                ),
            },
        )

    def import_model_trace(self, path: Path | None) -> None:
        if path is None:
            return
        records = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    records.append(redact(json.loads(line)))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"model trace line {number} is invalid JSON") from exc
        target = self.root / "trace/model/turns.jsonl"
        target.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in records))
        media = path.parent / "media"
        if media.is_dir():
            shutil.copytree(media, self.root / "trace/model/media", dirs_exist_ok=True)

    def save_protocol_evidence(self, evidence: Mapping[str, object]) -> None:
        """Persist summarized host-only evaluation without exposing native state."""
        atomic_json(self.root / "evaluation/protocol.json", evidence)

    @contextlib.contextmanager
    def span(self, name: str, *, category: str, inputs: object) -> Iterator[_Span]:
        self.call_index += 1
        span_id, parent = f"span-{self.call_index:06d}", _PARENT.get()
        directory = self.root / "trace/calls" / f"call-{self.call_index:06d}-{slug(name)}"
        with self.timing.measure("trace_write"):
            directory.mkdir()
            self.serializer.dump(directory / "input.json", inputs)
        self._event("span_start", span_id=span_id, parent_id=parent, name=name, category=category)
        token, state, started = _PARENT.set(span_id), {"ok": True}, time.monotonic()
        try:
            yield _Span(
                directory,
                self.serializer,
                state,
                self,
                capture_segmentation=category == "provider" and name.startswith("sam3."),
                capture_grasps=category == "provider" and name.startswith("contact_graspnet."),
            )
        except BaseException as exc:
            self.serializer.dump(
                directory / "error.json", {"type": type(exc).__name__, "message": str(exc)}
            )
            state["ok"] = False
            raise
        finally:
            self._event(
                "span_end",
                span_id=span_id,
                parent_id=parent,
                name=name,
                category=category,
                ok=state["ok"],
                duration_s=time.monotonic() - started,
            )
            _PARENT.reset(token)

    def before_step(self, action: RobotAction) -> None:
        del action
        if self.step_count >= self.max_steps:
            self.terminal_reason = "step_limit"
            raise StepLimitReached(f"run reached max_steps={self.max_steps}")
        self._step_entered = time.monotonic()

    def after_step(self, action: RobotAction, result: StepResult) -> None:
        if self._step_entered is not None:
            self.timing.add("adapter_step", time.monotonic() - self._step_entered)
            self._step_entered = None
        with self.timing.measure("record_step"):
            self._after_step(action, result)

    def _after_step(self, action: RobotAction, result: StepResult) -> None:
        if result.ok:
            self.step_count += 1
            self.cumulative_reward += result.reward or 0.0
            self.terminated, self.truncated = result.terminated, result.truncated
            if result.observation is not None:
                self.last_observation = result.observation
                if self.step_video:
                    self._video(result.observation, self.step_count)
                else:
                    self.keyframe(f"step-{self.step_count:04d}")
        record = {
            "step_index": self.step_count,
            "simulator_time_s": self.step_count / self.frequency,
            "action": self.serializer.encode(action, self.root / "episode/arrays"),
            "robot_state": (
                self.serializer.encode(result.observation.robot_state, self.root / "episode/arrays")
                if result.observation
                else None
            ),
            "ok": result.ok,
            "terminated": result.terminated,
            "truncated": result.truncated,
        }
        self.steps.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self.steps.flush()

    def on_reset(self, observation: Observation, metadata: Mapping[str, object]) -> None:
        self.last_observation = observation
        self.serializer.dump(
            self.root / "episode/reset.json", {"metadata": metadata, "observation": observation}
        )
        if self.step_video:
            self._video(observation, 0)
        else:
            self.keyframe("reset")

    def keyframe(self, label: str) -> None:
        if self.last_observation is None:
            return
        with self.timing.measure("keyframe"):
            self._keyframe(label)

    def _keyframe(self, label: str) -> None:
        self.keyframe_index += 1
        directory = self.root / "media/keyframes" / f"{self.keyframe_index:06d}-{slug(label)}"
        directory.mkdir()
        cameras = {}
        for name, camera in self.last_observation.cameras.items():
            rgb = directory / f"{_component(name)}.png"
            imageio.imwrite(
                rgb, np.asarray(camera.rgb, dtype=np.uint8), compress_level=PNG_COMPRESS_LEVEL
            )
            record = {"rgb": rgb.name, "frame": camera.frame}
            if not self.depth_saved:
                depth = directory / f"{_component(name)}-depth.npy"
                np.save(depth, camera.depth_m, allow_pickle=False)
                record["depth"] = depth.name
                self.depth_saved = True
            cameras[name] = record
        atomic_json(
            directory / "metadata.json",
            {"label": label, "step_index": self.step_count, "cameras": cameras},
        )

    def segmentation_overlay(self, result: SegmentationSet) -> None:
        """Save one annotated image containing every returned SAM candidate.

        One image per provider call keeps large campaigns tractable while still
        showing false positives, score ordering, boxes, and overlapping masks.
        Raw masks remain summarized rather than persisted.
        """
        if not result.ok or not result.segmentations or self.last_observation is None:
            return
        with self.timing.measure("overlay"):
            self._segmentation_overlay(result)

    def _segmentation_overlay(self, result: SegmentationSet) -> None:
        segmentations = result.segmentations[:5]
        first = segmentations[0]
        camera = next(
            (
                item
                for name, item in self.last_observation.cameras.items()
                if name == first.camera_name or item.frame == first.frame
            ),
            None,
        )
        if camera is None or camera.rgb.shape[:2] != first.mask.shape:
            return
        self.last_segmentations = tuple(segmentations)
        rgb = np.asarray(camera.rgb)
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = np.rint(rgb * 255.0)
        overlay = np.asarray(rgb, dtype=np.uint8).copy()
        colors = (
            np.array([255, 64, 64]),
            np.array([64, 255, 64]),
            np.array([64, 128, 255]),
            np.array([255, 192, 64]),
            np.array([224, 64, 255]),
        )
        records = []
        boxes = []
        for index, segmentation in enumerate(segmentations):
            mask = segmentation.mask
            if (
                mask.shape != overlay.shape[:2]
                or segmentation.camera_name != first.camera_name
                or segmentation.frame != first.frame
            ):
                continue
            color = colors[index]
            overlay[mask] = np.rint(0.68 * overlay[mask] + 0.32 * color).astype(np.uint8)
            box = None
            if segmentation.box_xyxy is not None:
                height, width = mask.shape
                x1, y1, x2, y2 = np.rint(segmentation.box_xyxy).astype(int)
                x1, x2 = np.clip((x1, x2), 0, width - 1)
                y1, y2 = np.clip((y1, y2), 0, height - 1)
                box = [int(x1), int(y1), int(x2), int(y2)]
                boxes.append((box, tuple(int(value) for value in color), index, segmentation))
            records.append(
                {
                    "index": index,
                    "label": segmentation.label,
                    "score": float(segmentation.score),
                    "box_xyxy": box,
                    "mask_pixels": int(np.count_nonzero(mask)),
                }
            )
        self.overlay_index += 1
        directory = self.root / "media/overlays"
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"sam3-{self.overlay_index:06d}-{_component(first.camera_name)}"
        from PIL import Image, ImageDraw

        image = Image.fromarray(overlay)
        draw = ImageDraw.Draw(image)
        for box, color, index, segmentation in boxes:
            draw.rectangle(box, outline=color, width=3)
            text = f"#{index + 1} {segmentation.label} {segmentation.score:.3f}"
            text_y = max(0, box[1] - 13)
            draw.rectangle(
                (box[0], text_y, min(image.width - 1, box[0] + 8 * len(text)), text_y + 12),
                fill=(0, 0, 0),
            )
            draw.text((box[0] + 2, text_y), text, fill=color)
        if image.width > 960:
            image.thumbnail((960, 960 * image.height // image.width))
        image.save(directory / f"{stem}.png", optimize=True)
        atomic_json(
            directory / f"{stem}.json",
            {
                "camera_name": first.camera_name,
                "camera_frame": first.frame,
                "image_shape": list(first.mask.shape),
                "candidates": records,
            },
        )
        self._write_scene3d(
            first.camera_name,
            camera,
            segmentations=tuple(segmentations),
            grasps=None,
        )

    def grasp_overlay(self, result: GraspSet) -> None:
        """Project grasp poses onto the current agentview RGB."""
        if not result.ok or not result.grasps or self.last_observation is None:
            return
        with self.timing.measure("overlay"):
            self._grasp_overlay(result)

    def _camera_for_grasp_frame(self, frame: str) -> tuple[str, CameraObservation] | None:
        cameras = self.last_observation.cameras
        for name in ("agentview", *cameras):
            camera = cameras.get(name)
            if camera is not None and camera.camera_pose.frame == frame:
                return name, camera
        return None

    def _project_base_to_pixel(
        self, point: np.ndarray, camera: CameraObservation
    ) -> tuple[int, int] | None:
        camera_from_base = invert_transform(camera.camera_pose.as_matrix())
        x_cam, y_cam, z_cam, _w = camera_from_base @ np.array(
            [float(point[0]), float(point[1]), float(point[2]), 1.0], dtype=np.float64
        )
        if z_cam <= 1e-6 or not np.isfinite([x_cam, y_cam, z_cam]).all():
            return None
        k = camera.intrinsics
        u = k[0, 0] * x_cam / z_cam + k[0, 2]
        v = k[1, 1] * y_cam / z_cam + k[1, 2]
        if not np.isfinite([u, v]).all():
            return None
        return int(round(u)), int(round(v))

    def _grasp_overlay(self, result: GraspSet) -> None:
        matched = self._camera_for_grasp_frame(result.grasps[0].frame)
        if matched is None:
            return
        camera_name, camera = matched
        rgb = np.asarray(camera.rgb)
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = np.rint(rgb * 255.0)
        canvas = np.asarray(rgb, dtype=np.uint8).copy()
        from PIL import Image, ImageDraw

        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        axis_colors = ((255, 64, 64), (64, 255, 64), (64, 128, 255))
        records = []
        for index, grasp in enumerate(result.grasps[:8]):
            origin = np.asarray(grasp.pose.position, dtype=np.float64)
            rotation = np.asarray(grasp.pose.as_matrix()[:3, :3], dtype=np.float64)
            length = 0.05
            pixel = self._project_base_to_pixel(origin, camera)
            if pixel is None:
                records.append(
                    {
                        "index": index,
                        "score": float(grasp.score),
                        "width_m": float(grasp.width_m),
                        "position": [round(float(v), 4) for v in origin],
                        "pixel_uv": None,
                    }
                )
                continue
            for axis, color in enumerate(axis_colors):
                tip = origin + length * rotation[:, axis]
                tip_pixel = self._project_base_to_pixel(tip, camera)
                if tip_pixel is not None:
                    draw.line([pixel, tip_pixel], fill=color, width=2)
            x, y = pixel
            draw.ellipse(
                (x - 4, y - 4, x + 4, y + 4),
                fill=(255, 255, 255),
                outline=(255, 220, 0),
                width=1,
            )
            draw.text((x + 6, y - 6), f"#{index + 1} {grasp.score:.2f}", fill=(255, 255, 255))
            records.append(
                {
                    "index": index,
                    "score": float(grasp.score),
                    "width_m": float(grasp.width_m),
                    "position": [round(float(v), 4) for v in origin],
                    "pixel_uv": [x, y],
                }
            )
        self.grasp_overlay_index += 1
        directory = self.root / "media/overlays"
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"grasp-{self.grasp_overlay_index:06d}-{_component(camera_name)}"
        if image.width > 960:
            image.thumbnail((960, 960 * image.height // image.width))
        image.save(directory / f"{stem}.png", optimize=True)
        atomic_json(
            directory / f"{stem}.json",
            {
                "camera_name": camera_name,
                "grasp_frame": result.grasps[0].frame,
                "candidates": records,
            },
        )
        self._write_scene3d(
            camera_name,
            camera,
            segmentations=self.last_segmentations,
            grasps=result,
        )

    def _lift_mask(
        self, camera: CameraObservation, segmentation: Segmentation
    ) -> PointCloud | None:
        try:
            return mask_to_point_cloud(
                camera, segmentation, target_frame=camera.camera_pose.frame
            )
        except ValueError:
            return None

    def _write_scene3d(
        self,
        camera_name: str,
        camera: CameraObservation,
        *,
        segmentations: tuple[Segmentation, ...],
        grasps: GraspSet | None,
    ) -> None:
        valid = np.isfinite(camera.depth_m) & (camera.depth_m > 0.0)
        scene = None
        if bool(np.any(valid)):
            scene = self._lift_mask(
                camera,
                Segmentation(
                    mask=valid,
                    label="scene",
                    score=1.0,
                    camera_name=camera_name,
                    frame=camera.frame,
                ),
            )
        object_layers: list[tuple[np.ndarray, tuple[int, int, int]]] = []
        for index, segmentation in enumerate(segmentations[:5]):
            if segmentation.mask.shape != camera.depth_m.shape:
                continue
            if segmentation.frame != camera.frame:
                continue
            lifted = self._lift_mask(camera, segmentation)
            if lifted is None:
                continue
            points, _colors = _downsample_points(
                lifted.points, lifted.colors, _MAX_OBJECT_POINTS
            )
            object_layers.append((points, _OBJECT_COLORS[index % len(_OBJECT_COLORS)]))
        grasp_origins: list[np.ndarray] = []
        grasp_axes: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]] = []
        if grasps is not None and grasps.ok:
            for grasp in grasps.grasps[:8]:
                if grasp.frame != camera.camera_pose.frame:
                    continue
                origin = np.asarray(grasp.pose.position, dtype=np.float64)
                rotation = np.asarray(grasp.pose.as_matrix()[:3, :3], dtype=np.float64)
                grasp_origins.append(origin)
                for axis, color in enumerate(((255, 64, 64), (64, 255, 64), (64, 128, 255))):
                    grasp_axes.append((origin, origin + 0.05 * rotation[:, axis], color))
        bound_points = []
        if scene is not None:
            scene_points, scene_colors = _downsample_points(
                scene.points, scene.colors, _MAX_SCENE_POINTS
            )
            bound_points.append(scene_points)
        else:
            scene_points, scene_colors = None, None
        for points, _color in object_layers:
            bound_points.append(points)
        if grasp_origins:
            bound_points.append(np.stack(grasp_origins))
            bound_points.append(np.stack([tip for _origin, tip, _color in grasp_axes]))
        if not bound_points:
            return
        all_points = np.concatenate(bound_points, axis=0)
        u_all, v_all, _depth_all = _view_uv(all_points)
        span = max(float(np.ptp(u_all)), float(np.ptp(v_all)), 1e-3)
        center_u = 0.5 * (float(np.min(u_all)) + float(np.max(u_all)))
        center_v = 0.5 * (float(np.min(v_all)) + float(np.max(v_all)))
        scale = (_SCENE3D_SIZE - 2 * _SCENE3D_MARGIN) / span

        def to_pixel(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            u, v, depth = _view_uv(points)
            x = np.rint((u - center_u) * scale + _SCENE3D_SIZE / 2.0).astype(int)
            y = np.rint(_SCENE3D_SIZE / 2.0 - (v - center_v) * scale).astype(int)
            return x, y, depth

        canvas = np.full((_SCENE3D_SIZE, _SCENE3D_SIZE, 3), 28, dtype=np.uint8)

        def splat(points: np.ndarray, colors: np.ndarray) -> None:
            x, y, depth = to_pixel(points)
            inside = (x >= 0) & (x < _SCENE3D_SIZE) & (y >= 0) & (y < _SCENE3D_SIZE)
            if not bool(np.any(inside)):
                return
            order = np.argsort(depth[inside])
            xs = x[inside][order]
            ys = y[inside][order]
            painted = colors[inside][order]
            canvas[ys, xs] = painted

        if scene_points is not None:
            if scene_colors is None:
                painted = np.full((scene_points.shape[0], 3), 120, dtype=np.uint8)
            else:
                painted = np.rint(_rgb8(scene_colors) * 0.55).astype(np.uint8)
            splat(scene_points, painted)
        for points, color in object_layers:
            painted = np.broadcast_to(
                np.asarray(color, dtype=np.uint8), (points.shape[0], 3)
            ).copy()
            splat(points, painted)

        from PIL import Image, ImageDraw

        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        for origin, tip, color in grasp_axes:
            x0, y0, _d0 = to_pixel(origin.reshape(1, 3))
            x1, y1, _d1 = to_pixel(tip.reshape(1, 3))
            draw.line(
                [(int(x0[0]), int(y0[0])), (int(x1[0]), int(y1[0]))],
                fill=color,
                width=2,
            )
        for origin in grasp_origins:
            x, y, _depth = to_pixel(origin.reshape(1, 3))
            px, py = int(x[0]), int(y[0])
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(255, 255, 255))
        draw.text((8, 8), "scene + lifted mask + grasps (3D)", fill=(220, 220, 220))
        self.scene3d_overlay_index += 1
        directory = self.root / "media/overlays"
        directory.mkdir(parents=True, exist_ok=True)
        kind = "grasp" if grasps is not None else "segment"
        stem = f"scene3d-{self.scene3d_overlay_index:06d}-{kind}-{_component(camera_name)}"
        image.save(directory / f"{stem}.png", optimize=True)
        atomic_json(
            directory / f"{stem}.json",
            {
                "kind": kind,
                "camera_name": camera_name,
                "frame": camera.camera_pose.frame,
                "n_scene_points": 0 if scene_points is None else int(scene_points.shape[0]),
                "n_object_points": int(sum(layer[0].shape[0] for layer in object_layers)),
                "n_grasps": len(grasp_origins),
            },
        )

    def finalize(
        self,
        *,
        program_ok: bool,
        task_success: bool | None,
        protocol_success: bool | None = None,
        termination_reason: str,
        program_result: object,
        error: object = None,
        cleanup_errors: tuple[str, ...] = (),
        success_observed_step: int | None = None,
    ) -> None:
        if self.finalized:
            return
        finalization_errors = list(cleanup_errors)
        with self.timing.measure("finalize"):
            # Closing an x264 writer flushes and muxes the whole file, so it is
            # video cost, not bookkeeping: charge it where a reader would look.
            with self.timing.measure("video_encode"):
                for name, writer in self.writers.items():
                    try:
                        writer.close()
                        (self.root / "media/videos" / f".{_component(name)}.partial.mp4").replace(
                            self.root / "media/videos" / f"{_component(name)}.mp4"
                        )
                    # Record video finalization errors without losing the episode result.
                    except Exception as exc:
                        finalization_errors.append(f"video {name}: {exc}")
            self.events.close()
            self.steps.close()
            self.serializer.dump(self.root / "source/program-result.json", program_result)
        wall_duration_s = time.monotonic() - self.started
        self.serializer.dump(
            self.root / "outcome.json",
            {
                "schema_version": SCHEMA_VERSION,
                "program_ok": program_ok,
                "task_success": task_success,
                "protocol_success": protocol_success,
                "cumulative_reward": self.cumulative_reward,
                "steps_executed": self.step_count,
                "simulator_duration_s": self.step_count / self.frequency,
                "wall_duration_s": wall_duration_s,
                # Where the wall clock went. Buckets are disjoint; the manifest
                # hash pass runs after this file is written and is not in them.
                "timing": self.timing.snapshot(wall_s=wall_duration_s),
                "termination_reason": termination_reason,
                "termination_description": TERMINATION_REASONS[termination_reason],
                # How the episode itself ended, independent of the reason above:
                # a latched success on a task without a success termination reads
                # task_succeeded while the environment never ended. These make the
                # two distinguishable after the fact.
                "environment_terminated": bool(self.terminated),
                "environment_truncated": bool(self.truncated),
                "success_observed_step": success_observed_step,
                "error": error,
                "finalization_errors": finalization_errors,
            },
        )
        atomic_json(
            self.root / "media/video-index.json",
            {"fps": self.frequency / self.video_frame_stride, "frames": self.video_frames},
        )
        self.identity.update(status="finalized", finished_at=_now())
        atomic_json(self.root / "run.json", self.identity)
        self._manifest()
        self.finalized = True

    def close_incomplete(
        self, error: BaseException, *, cleanup_errors: tuple[str, ...] = ()
    ) -> None:
        try:
            self.finalize(
                program_ok=False,
                task_success=None,
                termination_reason="harness_error",
                program_result=None,
                error={"type": type(error).__name__, "message": str(error)},
                cleanup_errors=cleanup_errors,
            )
        # Persist an incomplete run marker if finalization fails.
        except Exception:
            self.identity.update(status="incomplete", finished_at=_now())
            atomic_json(self.root / "run.json", self.identity)

    def _event(self, event: str, **fields: object) -> None:
        with self.timing.measure("trace_write"):
            self._write_event(event, **fields)

    def _write_event(self, event: str, **fields: object) -> None:
        self.event_index += 1
        record = {
            "event_index": self.event_index,
            "event": event,
            "timestamp": _now(),
            "simulator_time_s": self.step_count / self.frequency,
            **fields,
        }
        self.events.write(json.dumps(redact(record), sort_keys=True, allow_nan=False) + "\n")
        self.events.flush()

    def _video(self, observation: Observation, step: int) -> None:
        if not self.capture_videos or self.last_video_step == step:
            return
        if step % self.video_frame_stride:
            return
        with self.timing.measure("video_encode"):
            self._encode_video_frame(observation, step)

    def _encode_video_frame(self, observation: Observation, step: int) -> None:
        for name, camera in observation.cameras.items():
            if name not in self.writers:
                path = self.root / "media/videos" / f".{_component(name)}.partial.mp4"
                self.writers[name] = imageio.get_writer(
                    path,
                    fps=self.frequency / self.video_frame_stride,
                    codec="libx264",
                    macro_block_size=None,
                )
                self.video_frames[name] = 0
            self.writers[name].append_data(np.asarray(camera.rgb, dtype=np.uint8))
            self.video_frames[name] += 1
        self.last_video_step = step

    def _manifest(self) -> None:
        files = [
            {
                "path": str(path.relative_to(self.root)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(self.root.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        ]
        atomic_json(self.root / "manifest.json", {"schema_version": SCHEMA_VERSION, "files": files})

    @staticmethod
    def _git() -> Mapping[str, object]:
        repository = Path(__file__).resolve().parents[2]
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            dirty = bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=repository,
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
            )
            return {"commit": commit, "dirty": dirty}
        except (OSError, subprocess.CalledProcessError):
            return {"commit": None, "dirty": None}


class TracedProvider:
    """Tiny proxy adding provider child spans."""

    def __init__(self, provider: object, recorder: RunRecorder, name: str) -> None:
        self.provider, self.recorder, self.name = provider, recorder, name

    def __getattr__(self, name: str) -> object:
        attribute = getattr(self.provider, name)
        if not callable(attribute) or name == "health":
            return attribute

        def invoke(*args: object, **kwargs: object) -> object:
            with self.recorder.span(
                f"{self.name}.{name}", category="provider", inputs={"args": args, "kwargs": kwargs}
            ) as span:
                # Time the backend call alone. The span's own input/output
                # serialization is charged to trace_write, so a slow provider
                # and a slow trace never hide inside one another.
                with self.recorder.timing.measure(f"provider.{self.name}"):
                    result = attribute(*args, **kwargs)
                ok = getattr(result, "ok", True)
                return span.output(result, ok=ok if type(ok) is bool else True)

        return invoke


__all__ = [
    "TERMINATION_REASONS",
    "RunRecorder",
    "RunTiming",
    "StepLimitReached",
    "TracedProvider",
    "redact",
    "slug",
]
