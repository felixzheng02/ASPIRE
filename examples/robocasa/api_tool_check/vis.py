"""Drawing primitives for the API tool-check report.

Runs under ``cap-x/.venv-client`` (matplotlib + PIL + imageio), NOT under the harness venv:
``.venv-robocasa`` deliberately has no plotting stack, and adding one to it would change a
venv whose correctness is partly an absence. Nothing here touches cap_harness; it only reads
the ``.npz`` / ``.json`` / ``.png`` that ``probe.py`` wrote.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "out"
RAW = OUT / "raw"
FIGURES = OUT / "figures"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 9,
        "axes.titlesize": 10,
        "figure.dpi": 110,
    }
)

#: One colour per named object, reused across every figure so the reader can track an object.
PALETTE = {
    "fish": "#d1495b",
    "plate": "#2e86ab",
    "pan": "#8a4fff",
    "counter": "#e0a458",
    "cabinet": "#4f9d69",
    "drawer": "#9d4f8c",
    "cabinet handle": "#1f7a8c",
    "stove": "#5c5c5c",
    "sink": "#7f9c3a",
    "scene": "#9aa0a6",
}
SERIES = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]


# --------------------------------------------------------------------------- loading


class StageData:
    def __init__(self, name: str) -> None:
        self.name = name
        self.directory = RAW / name
        payload = json.loads((self.directory / "records.json").read_text(encoding="utf-8"))
        self.records: list[dict[str, Any]] = payload["records"]
        self.notes: list[str] = payload.get("notes", [])
        path = self.directory / "arrays.npz"
        self.arrays = dict(np.load(path, allow_pickle=False)) if path.exists() else {}

    def exists(self) -> bool:
        return self.directory.exists()

    def get(self, key: str, default: Any = None) -> Any:
        return self.arrays.get(key, default)

    def image(self, name: str) -> np.ndarray:
        import imageio.v2 as imageio

        return np.asarray(imageio.imread(self.directory / f"{name}.png"))

    def records_for(self, tool: str) -> list[dict[str, Any]]:
        return [record for record in self.records if record["tool"] == tool]

    def videos(self) -> list[Path]:
        return sorted(self.directory.glob("*.mp4"))


def figure_path(stage: str, name: str) -> Path:
    directory = FIGURES / stage
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def save(fig: plt.Figure, stage: str, name: str) -> str:
    path = figure_path(stage, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(OUT))


# --------------------------------------------------------------------------- projection


def project(points: np.ndarray, intrinsics: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
    """``robot0_link0`` XYZ -> (u, v, depth), the inverse of the API's back-projection."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    rotation = camera_pose[:3, :3]
    translation = camera_pose[:3, 3]
    in_camera = (points - translation) @ rotation
    depth = in_camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = in_camera[:, 0] * intrinsics[0, 0] / depth + intrinsics[0, 2]
        v = in_camera[:, 1] * intrinsics[1, 1] / depth + intrinsics[1, 2]
    return np.column_stack((u, v, depth))


def draw_segments_2d(
    axis: plt.Axes,
    segments: Sequence[tuple[np.ndarray, np.ndarray]],
    intrinsics: np.ndarray,
    camera_pose: np.ndarray,
    *,
    color: str,
    linewidth: float = 1.6,
    alpha: float = 1.0,
) -> None:
    """Project 3D line segments into the image and draw the ones in front of the camera."""
    for start, end in segments:
        pixels = project(np.stack([start, end]), intrinsics, camera_pose)
        if np.any(pixels[:, 2] <= 0.01) or not np.all(np.isfinite(pixels[:, :2])):
            continue
        axis.plot(
            pixels[:, 0], pixels[:, 1], color=color, linewidth=linewidth, alpha=alpha,
            solid_capstyle="round",
        )


# --------------------------------------------------------------------------- shapes


def box_segments(transform: np.ndarray, extents: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """The twelve edges of an oriented bounding box."""
    half = np.asarray(extents, dtype=np.float64) / 2.0
    corners = np.array(
        [
            [sx * half[0], sy * half[1], sz * half[2]]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
    )
    world = corners @ transform[:3, :3].T + transform[:3, 3]
    edges = [
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    ]
    return [(world[a], world[b]) for a, b in edges]


#: Panda gripper, drawn from its ``panda_hand`` origin: approach is local +Z, the jaws open
#: along local +/-Y, and the fingertips sit 0.1034 m out -- the same depth Contact-GraspNet
#: puts between a candidate pose and its contact point.
FINGER_LENGTH_M = 0.1034


def gripper_segments(
    transform: np.ndarray, width_m: float = 0.08, finger_length_m: float = FINGER_LENGTH_M
) -> list[tuple[np.ndarray, np.ndarray]]:
    half = width_m / 2.0
    local = [
        ([0, 0, -0.05], [0, 0, 0]),                      # wrist stem
        ([0, -half, 0], [0, half, 0]),                   # the bar the fingers slide on
        ([0, -half, 0], [0, -half, finger_length_m]),    # left finger
        ([0, half, 0], [0, half, finger_length_m]),      # right finger
    ]
    rotation, translation = transform[:3, :3], transform[:3, 3]
    return [
        (rotation @ np.asarray(a, dtype=float) + translation,
         rotation @ np.asarray(b, dtype=float) + translation)
        for a, b in local
    ]


def frame_segments(transform: np.ndarray, length: float = 0.05) -> list[tuple[np.ndarray, np.ndarray, str]]:
    origin = transform[:3, 3]
    return [
        (origin, origin + transform[:3, axis] * length, color)
        for axis, color in enumerate(("#d62728", "#2ca02c", "#1f77b4"))
    ]


# --------------------------------------------------------------------------- 2D overlays


def mask_overlay(
    rgb: np.ndarray, mask: np.ndarray, color: str, alpha: float = 0.45
) -> np.ndarray:
    """Tint the masked pixels and draw a one-pixel outline, returning a new RGB array."""
    tint = np.asarray(matplotlib.colors.to_rgb(color)) * 255.0
    out = np.asarray(rgb, dtype=np.float64).copy()
    mask = np.asarray(mask, dtype=bool)
    out[mask] = (1 - alpha) * out[mask] + alpha * tint
    edge = mask ^ _erode(mask)
    out[edge] = tint
    return np.clip(out, 0, 255).astype(np.uint8)


def _erode(mask: np.ndarray) -> np.ndarray:
    shrunk = mask.copy()
    shrunk[1:, :] &= mask[:-1, :]
    shrunk[:-1, :] &= mask[1:, :]
    shrunk[:, 1:] &= mask[:, :-1]
    shrunk[:, :-1] &= mask[:, 1:]
    return shrunk


def show_image(axis: plt.Axes, rgb: np.ndarray, title: str = "") -> None:
    axis.imshow(rgb)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(False)
    if title:
        axis.set_title(title)


def draw_box(axis: plt.Axes, box_xyxy: Sequence[float], color: str, label: str = "") -> None:
    x0, y0, x1, y1 = (float(v) for v in box_xyxy)
    axis.add_patch(
        Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=1.6)
    )
    if label:
        axis.text(
            x0 + 2,
            max(y0 - 4, 8),
            label,
            color="white",
            fontsize=7,
            bbox={"facecolor": color, "edgecolor": "none", "pad": 1.2},
        )


# --------------------------------------------------------------------------- 3D


def axes3d(fig: plt.Figure, position: Any, title: str = "") -> plt.Axes:
    axis = fig.add_subplot(position, projection="3d")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.grid(True)
    if title:
        axis.set_title(title)
    return axis


def equalize(axis: plt.Axes, points: np.ndarray, pad: float = 0.05) -> None:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    centre = (points.max(axis=0) + points.min(axis=0)) / 2.0
    span = float(np.max(points.max(axis=0) - points.min(axis=0))) / 2.0 + pad
    axis.set_xlim(centre[0] - span, centre[0] + span)
    axis.set_ylim(centre[1] - span, centre[1] + span)
    axis.set_zlim(centre[2] - span, centre[2] + span)
    axis.set_box_aspect((1, 1, 1))


def draw_segments_3d(
    axis: plt.Axes,
    segments: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    color: str,
    linewidth: float = 1.5,
    alpha: float = 1.0,
    label: str | None = None,
) -> None:
    for index, (start, end) in enumerate(segments):
        axis.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=label if index == 0 else None,
        )


def scatter_cloud(
    axis: plt.Axes,
    points: np.ndarray,
    *,
    colors: np.ndarray | None = None,
    color: str = "#9aa0a6",
    size: float = 1.0,
    alpha: float = 0.5,
    label: str | None = None,
    stride: int = 1,
) -> None:
    points = np.asarray(points)[::stride]
    if colors is not None:
        rgb = np.asarray(colors)[::stride]
        rgb = rgb / 255.0 if rgb.max() > 1.5 else rgb
        axis.scatter(
            points[:, 0], points[:, 1], points[:, 2], c=rgb, s=size, alpha=alpha,
            linewidths=0, label=label,
        )
    else:
        axis.scatter(
            points[:, 0], points[:, 1], points[:, 2], c=color, s=size, alpha=alpha,
            linewidths=0, label=label,
        )


def turntable_gif(
    draw: Any,
    stage: str,
    name: str,
    *,
    frames: int = 36,
    elevation: float = 22.0,
    figsize: tuple[float, float] = (6.0, 5.2),
    fps: int = 12,
) -> str:
    """Render ``draw(axis)`` from ``frames`` azimuths into an animated GIF.

    A still 3D scatter is genuinely hard to read -- depth is ambiguous without motion -- so
    every point-cloud figure also gets a turntable.
    """
    import imageio.v2 as imageio

    images = []
    for index in range(frames):
        fig = plt.figure(figsize=figsize)
        axis = axes3d(fig, 111)
        draw(axis)
        axis.view_init(elev=elevation, azim=-60 + 360.0 * index / frames)
        fig.canvas.draw()
        buffer = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        images.append(buffer)
        plt.close(fig)
    path = figure_path(stage, name)
    imageio.mimsave(path, images, duration=1.0 / fps, loop=0)
    return str(path.relative_to(OUT))


def filmstrip(
    images: Sequence[tuple[str, np.ndarray]], stage: str, name: str, *, columns: int = 5
) -> str:
    rows = int(np.ceil(len(images) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(2.7 * columns, 2.9 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for index, (title, rgb) in enumerate(images):
        axis = axes[index // columns][index % columns]
        axis.axis("on")
        show_image(axis, rgb, title)
    fig.tight_layout()
    return save(fig, stage, name)
