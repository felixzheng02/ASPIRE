"""Shared plumbing for the out-of-sandbox RoboCasa API tool check.

Why out of the sandbox
----------------------
``api_tool_check.py`` in this directory is the same catalog written as a *generated program*:
it runs inside ``cap-harness run``, where the sandbox bans every import but ``math`` and bans
ndarray methods (trap 15). That is the right place to prove the agent-visible path works, and
it is run as a cross-check -- but it cannot produce a PNG, a point-cloud plot or a video,
because the sandbox cannot write files and has no numpy.

So this module assembles **the identical public tool namespace** the sandbox would be given --
``CapApi.register_tools(ToolRegistry(public_extension_allowlist=ROBOCASA_PUBLIC_TOOL_NAMES))``
-- in an ordinary Python process, and calls the tools through that mapping. Same callables,
same allowlist, same adapter and providers as ``cap_harness.run.run_program``; the only
difference is that here the caller may keep the arrays and draw them.

Run with::

    PYTHONPATH="" aspire/.venv-robocasa/bin/python examples/robocasa/api_tool_check/probe.py
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from cap_harness.api import CapApi
from cap_harness.contracts import (
    ArmCommand,
    CameraObservation,
    GraspCandidate,
    MotionStrategy,
    Observation,
    Pose,
    RobotAction,
    RobotState,
    Trajectory,
)
from cap_harness.providers.curobo.client import CuRoboProvider
from cap_harness.providers.graspnet.client import ContactGraspNetProvider
from cap_harness.providers.pyroki.client import PyRokiProvider
from cap_harness.providers.sam3.client import Sam3Provider
from cap_harness.registry import ROBOCASA_PUBLIC_TOOL_NAMES, ToolRegistry
from cap_harness.robocasa.adapter import RobocasaAdapter
from cap_harness.robocasa.registry import RobocasaTaskRegistry

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
RAW = OUT / "raw"

#: Every tool the RoboCasa agent is handed. Assembled, not typed out: the point of the check
#: is that this set is whatever ``register_tools`` actually produces.
EXPECTED_TOOL_COUNT = 31


# --------------------------------------------------------------------------- session


@dataclass
class StepRecord:
    """One control tick, as seen from outside the sandbox."""

    index: int
    commanded_joints: list[float]
    commanded_gripper: float | None
    joints: list[float]
    ee_position: list[float]
    ee_quaternion_wxyz: list[float]
    gripper: float
    ok: bool
    truncated: bool


class StepCapture:
    """``run_observer`` that keeps per-tick state and (optionally) RGB frames.

    The adapter calls ``before_step`` / ``after_step`` around every tick of every multi-tick
    primitive, so this sees the inside of ``move_to_joints``, ``execute_trajectory`` and the
    gripper settle loops -- which is what makes a tracking plot and a video possible.
    ``navigate_to_pose`` is the exception: it drives the base inside the driver's own loop and
    never reaches ``step()``, so base motion is captured from the simulator's video buffer
    instead (see ``probe.capture_base_video``).
    """

    def __init__(self) -> None:
        self.steps: list[StepRecord] = []
        self.frames: dict[str, list[np.ndarray]] = {"agentview": [], "robot0_eye_in_hand": []}
        self.capture_frames = False
        self.frame_stride = 1
        self._pending: tuple[list[float], float | None] | None = None
        self._index = 0

    # -- observer protocol ------------------------------------------------
    def on_reset(self, observation: Observation, metadata: Mapping[str, object]) -> None:
        self.reset_buffers()

    def before_step(self, action: RobotAction) -> None:
        command = next(iter(action.arms.values()))
        target = np.asarray(command.target, dtype=np.float64).reshape(-1).tolist()
        gripper = None if command.gripper_position is None else float(command.gripper_position)
        self._pending = (target, gripper)

    def after_step(self, action: RobotAction, result: Any) -> None:
        commanded, gripper = self._pending or ([], None)
        self._pending = None
        observation = getattr(result, "observation", None)
        if observation is None:
            return
        state = observation.robot_state
        pose = state.end_effector_poses["primary"]
        self.steps.append(
            StepRecord(
                index=self._index,
                commanded_joints=commanded,
                commanded_gripper=gripper,
                joints=np.asarray(state.joint_positions["primary"], dtype=float).tolist(),
                ee_position=np.asarray(pose.position, dtype=float).tolist(),
                ee_quaternion_wxyz=np.asarray(pose.quaternion_wxyz, dtype=float).tolist(),
                gripper=float(state.gripper_positions["primary"]),
                ok=bool(getattr(result, "ok", True)),
                truncated=bool(getattr(result, "truncated", False)),
            )
        )
        if self.capture_frames and self._index % self.frame_stride == 0:
            for name, camera in observation.cameras.items():
                if name in self.frames:
                    self.frames[name].append(np.asarray(camera.rgb, dtype=np.uint8).copy())
        self._index += 1

    # -- bookkeeping ------------------------------------------------------
    def reset_buffers(self) -> None:
        self.steps = []
        self.frames = {"agentview": [], "robot0_eye_in_hand": []}
        self._pending = None
        self._index = 0

    def take(self) -> tuple[list[StepRecord], dict[str, list[np.ndarray]]]:
        steps, frames = self.steps, self.frames
        self.steps = []
        self.frames = {"agentview": [], "robot0_eye_in_hand": []}
        self._index = 0
        return steps, frames


@dataclass
class Session:
    adapter: RobocasaAdapter
    api: CapApi
    tools: Mapping[str, Any]
    capture: StepCapture
    task: str
    seed: int

    def reset(self) -> Observation:
        observation = self.adapter.reset(self.task, self.seed)
        self.capture.reset_buffers()
        return observation

    @property
    def client(self) -> Any:
        """The driver handle. Used ONLY for the simulator video buffer, which the public
        tool surface has no equivalent of, and for the elapsed-tick bookkeeping printed in
        the log. No tool result on this report comes from it.
        """
        return self.adapter._client

    def elapsed(self) -> int:
        return int(self.adapter._elapsed_steps)

    def close(self) -> None:
        self.adapter.close()


def build_session(task: str, seed: int, *, horizon: int = 1000) -> Session:
    """Assemble exactly what ``cap_harness.run.run_program`` assembles for ``robocasa``."""
    capture = StepCapture()
    adapter = RobocasaAdapter(
        registry=RobocasaTaskRegistry(),
        horizon=horizon,
        run_observer=capture,
        camera_height=512,
        camera_width=512,
    )
    adapter.reset(task, seed)
    curobo = CuRoboProvider()
    api = CapApi(
        adapter,
        segmentation_provider=Sam3Provider(),
        grasp_providers={"contact-graspnet": ContactGraspNetProvider()},
        ik_providers={"pyroki": PyRokiProvider(), "curobo": curobo},
        trajectory_planning_providers={"curobo": curobo},
        integrated_pose_planning_providers={"curobo-integrated": curobo},
        default_camera="agentview",
    )
    registry = api.register_tools(
        ToolRegistry(public_extension_allowlist=ROBOCASA_PUBLIC_TOOL_NAMES)
    )
    tools = registry.public_tools()
    return Session(
        adapter=adapter, api=api, tools=tools, capture=capture, task=task, seed=seed
    )


# --------------------------------------------------------------------------- geometry


def pose_matrix(pose: Pose) -> np.ndarray:
    return np.asarray(pose.as_matrix(), dtype=np.float64)


def quat_to_matrix(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in quaternion_wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def yaw_of(pose: Pose) -> float:
    w, x, y, z = (float(v) for v in pose.quaternion_wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    """Geodesic angle between two orientations, in degrees."""
    rotation = quat_to_matrix(a).T @ quat_to_matrix(b)
    cos = (float(np.trace(rotation)) - 1.0) / 2.0
    return float(np.degrees(math.acos(max(-1.0, min(1.0, cos)))))


def project_points(points_base: np.ndarray, camera: CameraObservation) -> np.ndarray:
    """Project ``robot0_link0`` points to pixels through the public calibration.

    Exactly the inverse of what ``mask_to_point_cloud`` does, using only ``camera.intrinsics``
    and ``camera.camera_pose`` -- so a round trip that does not land back on the mask is a
    calibration bug, not a rendering choice.
    """
    points = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    transform = pose_matrix(camera.camera_pose)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    in_camera = (points - translation) @ rotation
    depth = in_camera[:, 2]
    intrinsics = np.asarray(camera.intrinsics, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = in_camera[:, 0] * intrinsics[0, 0] / depth + intrinsics[0, 2]
        v = in_camera[:, 1] * intrinsics[1, 1] / depth + intrinsics[1, 2]
    return np.column_stack((u, v, depth))


def panda_hand_to_eef(pose: Pose) -> Pose:
    """Inverse of the PyRoki eef -> panda_hand mapping, for grasp poses.

    ``PyRokiProvider`` solves ``hand_position = eef_position + R_eef @ [0, 0, -0.1]`` and
    ``R_hand = R_eef @ Rz(+90 deg)``. Contact-GraspNet poses come back already
    right-multiplied by that same local Z, so if they are *panda_hand* poses this undoes it;
    if they are already eef poses, this is the wrong direction. Both are reported.
    """
    rotation = quat_to_matrix(pose.quaternion_wxyz)
    rz_minus_90 = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    eef_rotation = rotation @ rz_minus_90
    eef_position = np.asarray(pose.position, dtype=np.float64) + rotation[:, 2] * 0.1
    transform = np.eye(4)
    transform[:3, :3] = eef_rotation
    transform[:3, 3] = eef_position
    from cap_harness.geometry import matrix_to_pose

    return matrix_to_pose(transform, frame=pose.frame)


def world_from_base(base_pose: Pose) -> np.ndarray:
    """``robocasa.get_base_pose`` as a 4x4. It reports the ``robot0_link0`` body in world."""
    return pose_matrix(base_pose)


# --------------------------------------------------------------------------- dumping


class NumpyJson(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, Mapping):
            return dict(o)
        if isinstance(o, Pose):
            return {
                "position": o.position.tolist(),
                "quaternion_wxyz": o.quaternion_wxyz.tolist(),
                "frame": o.frame,
            }
        if isinstance(o, (set, frozenset, tuple)):
            return list(o)
        return str(o)


@dataclass
class Stage:
    """One group of tool calls plus everything needed to draw them."""

    name: str
    records: list[dict[str, Any]] = field(default_factory=list)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def directory(self) -> Path:
        path = RAW / self.name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def record(
        self,
        tool: str,
        inputs: Mapping[str, Any],
        output: Mapping[str, Any],
        *,
        ok: bool,
        label: str = "",
        verdict: str = "",
        assets: Mapping[str, str] | None = None,
        elapsed_s: float | None = None,
    ) -> dict[str, Any]:
        entry = {
            "tool": tool,
            "label": label or tool,
            "ok": bool(ok),
            "verdict": verdict,
            "inputs": dict(inputs),
            "output": dict(output),
            "assets": dict(assets or {}),
            "elapsed_s": None if elapsed_s is None else round(float(elapsed_s), 3),
        }
        self.records.append(entry)
        status = "ok  " if ok else "FAIL"
        print(f"  [{status}] {entry['label']}" + (f"  -- {verdict}" if verdict else ""))
        return entry

    def array(self, name: str, value: np.ndarray) -> str:
        self.arrays[name] = np.asarray(value)
        return name

    def image(self, name: str, rgb: np.ndarray) -> str:
        import imageio.v2 as imageio

        path = self.directory / f"{name}.png"
        imageio.imwrite(path, np.asarray(rgb, dtype=np.uint8))
        return str(path.relative_to(OUT))

    def video(self, name: str, frames: Sequence[np.ndarray], fps: float = 20.0) -> str | None:
        if not len(frames):
            return None
        import imageio.v2 as imageio

        path = self.directory / f"{name}.mp4"
        with imageio.get_writer(path, fps=fps, macro_block_size=None, quality=7) as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame, dtype=np.uint8))
        return str(path.relative_to(OUT))

    def flush(self) -> None:
        directory = self.directory
        (directory / "records.json").write_text(
            json.dumps(
                {"stage": self.name, "notes": self.notes, "records": self.records},
                cls=NumpyJson,
                indent=2,
            ),
            encoding="utf-8",
        )
        if self.arrays:
            np.savez_compressed(directory / "arrays.npz", **self.arrays)
        print(f"  -> {directory}  ({len(self.records)} records, {len(self.arrays)} arrays)")


def dump_pose(pose: Pose) -> dict[str, Any]:
    return {
        "position_m": [round(float(v), 5) for v in pose.position],
        "quaternion_wxyz": [round(float(v), 5) for v in pose.quaternion_wxyz],
        "yaw_rad": round(yaw_of(pose), 5),
        "frame": pose.frame,
    }


def dump_execution(result: Any) -> dict[str, Any]:
    return {
        "ok": bool(result.ok),
        "status": str(result.status),
        "steps_executed": int(result.steps_executed),
        "terminated": bool(result.terminated),
        "truncated": bool(result.truncated),
        "final_errors": {k: round(float(v), 5) for k, v in dict(result.final_errors).items()},
        "error": error_text(result.error),
        "diagnostics": {k: str(v) for k, v in dict(result.diagnostics).items()},
    }


def error_text(error: Any) -> str:
    if error is None:
        return ""
    return f"{error.code}: {error.message}"


def dump_error(error: Any) -> dict[str, Any] | None:
    """The whole typed error, including the ``details`` a bare message drops."""
    if error is None:
        return None
    return {
        "code": str(error.code),
        "message": str(error.message),
        "details": {k: str(v) for k, v in dict(getattr(error, "details", {}) or {}).items()},
    }


def dump_segmentation(segmentation: Any) -> dict[str, Any]:
    mask = np.asarray(segmentation.mask, dtype=bool)
    return {
        "label": str(segmentation.label),
        "score": round(float(segmentation.score), 4),
        "camera_name": str(segmentation.camera_name),
        "frame": str(segmentation.frame),
        "box_xyxy": None
        if segmentation.box_xyxy is None
        else [round(float(v), 1) for v in segmentation.box_xyxy],
        "mask_shape": list(mask.shape),
        "mask_pixels": int(mask.sum()),
        "mask_fraction": round(float(mask.mean()), 5),
    }


def dump_cloud(cloud: Any) -> dict[str, Any]:
    points = np.asarray(cloud.points, dtype=np.float64)
    return {
        "n_points": int(len(points)),
        "frame": str(cloud.frame),
        "centroid_m": [round(float(v), 4) for v in points.mean(axis=0)],
        "aabb_min_m": [round(float(v), 4) for v in points.min(axis=0)],
        "aabb_max_m": [round(float(v), 4) for v in points.max(axis=0)],
        "has_colors": cloud.colors is not None,
    }


def dump_geometry(geometry: Any) -> dict[str, Any]:
    return {
        "pose": dump_pose(geometry.pose),
        "extents_m": [round(float(v), 4) for v in geometry.extents],
        "point_count": geometry.point_count,
    }


def dump_grasp(grasp: GraspCandidate) -> dict[str, Any]:
    return {
        "score": round(float(grasp.score), 4),
        "width_m": round(float(grasp.width_m), 4),
        "approach_distance_m": round(float(grasp.approach_distance_m), 4),
        "pose": dump_pose(grasp.pose),
        "metadata": {k: str(v) for k, v in dict(grasp.metadata).items()},
    }


def dump_state(state: RobotState) -> dict[str, Any]:
    return {
        "embodiment": str(state.embodiment),
        "base_frame": str(state.base_frame),
        "arms": list(state.joint_positions),
        "joint_names": [str(n) for n in state.joint_names["primary"]],
        "joint_positions_rad": [round(float(v), 5) for v in state.joint_positions["primary"]],
        "joint_velocities": [round(float(v), 5) for v in state.joint_velocities["primary"]],
        "gripper_position": round(float(state.gripper_positions["primary"]), 4),
        "ee_pose": dump_pose(state.end_effector_poses["primary"]),
        "timestamp_s": state.timestamp_s,
    }


def steps_to_arrays(steps: Sequence[StepRecord]) -> dict[str, np.ndarray]:
    if not steps:
        return {}
    width = max((len(s.commanded_joints) for s in steps), default=0)
    commanded = np.full((len(steps), width), np.nan)
    for index, step in enumerate(steps):
        if step.commanded_joints:
            commanded[index, : len(step.commanded_joints)] = step.commanded_joints
    return {
        "commanded_joints": commanded,
        "measured_joints": np.asarray([s.joints for s in steps], dtype=np.float64),
        "ee_position": np.asarray([s.ee_position for s in steps], dtype=np.float64),
        "ee_quaternion_wxyz": np.asarray([s.ee_quaternion_wxyz for s in steps], dtype=np.float64),
        "gripper": np.asarray([s.gripper for s in steps], dtype=np.float64),
        "commanded_gripper": np.asarray(
            [np.nan if s.commanded_gripper is None else s.commanded_gripper for s in steps],
            dtype=np.float64,
        ),
    }


class Timer:
    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self.start

    @property
    def elapsed(self) -> float:
        return getattr(self, "seconds", time.perf_counter() - self.start)


__all__ = [
    "EXPECTED_TOOL_COUNT",
    "HERE",
    "OUT",
    "RAW",
    "ArmCommand",
    "MotionStrategy",
    "NumpyJson",
    "Pose",
    "RobotAction",
    "Session",
    "Stage",
    "StepCapture",
    "Timer",
    "Trajectory",
    "build_session",
    "dump_cloud",
    "dump_error",
    "dump_execution",
    "dump_geometry",
    "dump_grasp",
    "dump_pose",
    "dump_segmentation",
    "dump_state",
    "error_text",
    "panda_hand_to_eef",
    "pose_matrix",
    "project_points",
    "quat_to_matrix",
    "quaternion_angle_deg",
    "steps_to_arrays",
    "world_from_base",
    "yaw_of",
]
