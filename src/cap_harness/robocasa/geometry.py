"""Pure XY standoff / approach geometry and base-offset kinematics for RoboCasa.

The standoff / approach hull helpers are copied from ``cap_harness.behavior.adapter``
rather than imported: the harness must not grow a dependency from this embodiment onto
BEHAVIOR. Both clouds share one frame, and that frame is the frame of the answer.
``base_pose_is_free`` is world-frame on the wire; transform before mixing the two.

The base-offset helpers at the bottom are this embodiment's own. They are pure so that the
one thing a silent error would invert -- the sign and frame of the shift -- is testable
without a simulator or an IK service.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from cap_harness.contracts import Pose


def _closest_point_on_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    segment = b - a
    length = float(np.dot(segment, segment))
    if length <= 1e-12:
        return a.copy()
    t = float(np.clip(np.dot(point - a, segment) / length, 0.0, 1.0))
    return a + t * segment


def _hull_vertices(points_xy: np.ndarray) -> np.ndarray:
    """Convex hull vertices in order; falls back to the axis-aligned box for degenerate input."""
    unique = np.unique(points_xy, axis=0)
    if len(unique) >= 3:
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(unique)
            return unique[hull.vertices]
        except Exception:  # pragma: no cover - collinear or numerically degenerate clouds
            pass
    lower, upper = unique.min(axis=0), unique.max(axis=0)
    return np.array(
        [[lower[0], lower[1]], [upper[0], lower[1]], [upper[0], upper[1]], [lower[0], upper[1]]]
    )


def plan_standoff_pose(
    support_cloud: Any, object_cloud: Any, standoff_m: float = 0.3
) -> tuple[float, float, float]:
    """Pure geometry: a base pose ``standoff_m`` outside the support surface's nearest edge.

    The support cloud's convex hull in XY gives the edges; the edge closest to the object's
    median XY is chosen; the outward normal is the one pointing away from the hull centroid;
    yaw faces the object. Both clouds must share one frame, which is the frame of the answer.
    """
    standoff = float(standoff_m)
    if not np.isfinite(standoff) or standoff <= 0.0:
        raise ValueError("standoff_m must be positive")
    support = np.asarray(getattr(support_cloud, "points", support_cloud), dtype=np.float64)
    target = np.asarray(getattr(object_cloud, "points", object_cloud), dtype=np.float64)
    if support.ndim != 2 or support.shape[1] != 3 or len(support) == 0:
        raise ValueError("support_cloud must contain (N, 3) points")
    if target.ndim != 2 or target.shape[1] != 3 or len(target) == 0:
        raise ValueError("object_cloud must contain (N, 3) points")
    support_frame = getattr(support_cloud, "frame", None)
    object_frame = getattr(object_cloud, "frame", None)
    if support_frame is not None and object_frame is not None and support_frame != object_frame:
        raise ValueError("support and object clouds must share a frame")
    object_xy = np.median(target[:, :2], axis=0)
    vertices = _hull_vertices(support[:, :2])
    centroid = vertices.mean(axis=0)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for index in range(len(vertices)):
        a, b = vertices[index], vertices[(index + 1) % len(vertices)]
        closest = _closest_point_on_segment(object_xy, a, b)
        distance = float(np.linalg.norm(object_xy - closest))
        if best is None or distance < best[0]:
            edge = b - a
            normal = np.array([edge[1], -edge[0]])
            norm = float(np.linalg.norm(normal))
            normal = normal / norm if norm > 1e-9 else np.array([1.0, 0.0])
            if np.dot(normal, closest - centroid) < 0.0:
                normal = -normal
            best = (distance, closest, normal)
    assert best is not None
    _, closest, normal = best
    standoff_xy = closest + normal * standoff
    yaw = math.atan2(object_xy[1] - standoff_xy[1], object_xy[0] - standoff_xy[0])
    return float(standoff_xy[0]), float(standoff_xy[1]), float(yaw)


def plan_approach_pose(
    object_cloud: Any,
    base_pose: tuple[float, float, float],
    distance_m: float = 0.7,
) -> tuple[float, float, float]:
    """Pure geometry: a base pose ``distance_m`` short of the object along the line from the base.

    Useful for free-standing objects (a can on the floor) where there is no support surface to
    stand off from. Yaw faces the object. The answer is in the cloud's frame.
    """
    distance = float(distance_m)
    if not np.isfinite(distance) or distance < 0.0:
        raise ValueError("distance_m must be non-negative")
    points = np.asarray(getattr(object_cloud, "points", object_cloud), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("object_cloud must contain (N, 3) points")
    object_xy = np.median(points[:, :2], axis=0)
    base_xy = np.asarray(base_pose[:2], dtype=np.float64)
    direction = object_xy - base_xy
    span = float(np.linalg.norm(direction))
    if span <= 1e-9:
        return float(base_xy[0]), float(base_xy[1]), float(base_pose[2])
    unit = direction / span
    goal = object_xy - unit * min(distance, span)
    yaw = math.atan2(unit[1], unit[0])
    return float(goal[0]), float(goal[1]), float(yaw)


def lateral_base_offsets(max_offset_m: float = 0.40, step_m: float = 0.05) -> tuple[float, ...]:
    """Candidate lateral base offsets in metres, nearest first, excluding zero.

    Lateral only, and that is a measured choice rather than a simplification: on this kitchen
    yaw alone rescues no scene, and forward is the direction the counter blocks (5.5 cm
    achieved of 30 cm commanded). An axis that cannot move the robot does not belong in a
    sweep that costs an IK call per entry.

    Nearest first so that a caller breaking ties on sweep order automatically prefers the
    smallest move, which is the one that perturbs the robot-mounted camera least.
    """
    span = float(max_offset_m)
    step = float(step_m)
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError("max_offset_m must be positive")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step_m must be positive")
    if step > span:
        raise ValueError("step_m must not exceed max_offset_m")
    count = int(math.floor(span / step + 1e-9))
    magnitudes = [round(step * index, 9) for index in range(1, count + 1)]
    offsets: list[float] = []
    for magnitude in magnitudes:
        offsets.extend((-magnitude, magnitude))
    return tuple(offsets)


def pose_after_base_offset(pose: Pose, dx: float, dy: float) -> Pose:
    """Re-express ``pose`` in the base frame the robot would have after moving ``(dx, dy)``.

    The simulator reports ``robot0_link0`` as the base pose, and ``drive_base`` servos a
    relative command in that same frame, so a pure translation of the base by ``(dx, dy)``
    with no yaw change moves a fixed world point to ``p - (dx, dy, 0)`` in the new frame and
    leaves its orientation alone. The frame label is unchanged because it names the same link.
    """
    if not isinstance(pose, Pose):
        raise ValueError("pose must be a Pose")
    shift = np.array([float(dx), float(dy), 0.0], dtype=np.float64)
    if not bool(np.all(np.isfinite(shift))):
        raise ValueError("dx and dy must be finite")
    return Pose(
        position=np.asarray(pose.position, dtype=np.float64) - shift,
        quaternion_wxyz=pose.quaternion_wxyz,
        frame=pose.frame,
    )


def retreat_pose(pose: Pose, standoff_m: float) -> Pose:
    """``standoff_m`` back along the gripper's own approach axis (the pose's local +z).

    The arm cannot arrive at a closed-gripper pose from an arbitrary direction -- the fingers
    have to straddle the object -- so an executable candidate needs this pose to solve as well
    as the grasp itself. Same convention as ``examples/robocasa/grasp_bench``.
    """
    if not isinstance(pose, Pose):
        raise ValueError("pose must be a Pose")
    distance = float(standoff_m)
    if not np.isfinite(distance) or distance < 0.0:
        raise ValueError("standoff_m must be non-negative")
    approach = np.asarray(pose.as_matrix(), dtype=np.float64)[:3, 2]
    return Pose(
        position=np.asarray(pose.position, dtype=np.float64) - approach * distance,
        quaternion_wxyz=pose.quaternion_wxyz,
        frame=pose.frame,
    )
