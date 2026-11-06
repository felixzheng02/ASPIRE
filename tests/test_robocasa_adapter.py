from __future__ import annotations

import math

import numpy as np
import pytest

from cap_harness.contracts import PointCloud
from cap_harness.robocasa.adapter import (
    PANDA_JOINT_NAMES,
    TRAJECTORY_MAX_JOINT_STEP_RAD,
    RobocasaAdapter,
    densify_joint_waypoints,
)
from cap_harness.robocasa.geometry import plan_approach_pose, plan_standoff_pose


def test_densify_joint_waypoints_caps_each_step() -> None:
    start = np.zeros(7)
    goal = np.zeros(7)
    goal[0] = 0.20
    out = densify_joint_waypoints(np.vstack([start, goal]), max_step_rad=0.05)
    deltas = np.abs(np.diff(out, axis=0))
    assert np.all(deltas <= 0.05 + 1e-12)
    np.testing.assert_allclose(out[0], start)
    np.testing.assert_allclose(out[-1], goal)
    assert out.shape[0] >= 5


def test_densify_joint_waypoints_keeps_already_dense_paths() -> None:
    points = np.vstack([np.zeros(7), np.full(7, 0.04)])
    out = densify_joint_waypoints(points, max_step_rad=TRAJECTORY_MAX_JOINT_STEP_RAD)
    assert out.shape[0] == 2
    np.testing.assert_allclose(out, points)


def test_planning_context_is_panda_in_robot0_link0() -> None:
    context = RobocasaAdapter().get_planning_context()
    assert context.embodiment == "robocasa"
    assert context.model == "panda"
    assert context.end_effector_links == {"primary": "panda_hand"}
    assert context.joint_names["primary"] == PANDA_JOINT_NAMES
    np.testing.assert_array_equal(context.base_transforms["primary"], np.eye(4))


def test_plan_standoff_pose_stands_outside_the_nearest_edge() -> None:
    table = np.array(
        [[x, y, 0.7] for x in np.linspace(1.0, 2.0, 11) for y in np.linspace(-0.5, 0.5, 11)]
    )
    radio = np.array([[1.8, 0.1, 0.8], [1.85, 0.12, 0.8], [1.82, 0.08, 0.8]])
    x, y, yaw = plan_standoff_pose(
        PointCloud(points=table, frame="world"),
        PointCloud(points=radio, frame="world"),
        standoff_m=0.3,
    )
    assert x == pytest.approx(2.3, abs=1e-6)
    assert y == pytest.approx(0.1, abs=1e-6)
    assert yaw == pytest.approx(math.pi, abs=1e-6)
    with pytest.raises(ValueError):
        plan_standoff_pose(
            PointCloud(points=table, frame="world"),
            PointCloud(points=radio, frame="robot0_link0"),
        )


def test_plan_approach_pose_uses_cloud_frame_origin_for_link0() -> None:
    target = np.array([[0.7, 0.0, 0.9], [0.72, 0.02, 0.9], [0.68, -0.01, 0.9]])
    x, y, yaw = plan_approach_pose(
        PointCloud(points=target, frame="robot0_link0"),
        (0.0, 0.0, 0.0),
        distance_m=0.3,
    )
    assert math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw)
    assert x == pytest.approx(0.4, abs=0.05)


class _FakeFootprintClient:
    def __init__(self) -> None:
        self.calls: list[tuple[float, float, float]] = []
        self.last = {
            "base_pos": np.array([1.0, 2.0, 0.7]),
            "base_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
        }

    def base_pose_is_free(self, x: float, y: float, yaw: float = 0.0) -> bool:
        self.calls.append((float(x), float(y), float(yaw)))
        return True


def test_adapter_base_pose_is_free_is_world_yes_no() -> None:
    adapter = RobocasaAdapter()
    fake = _FakeFootprintClient()
    adapter._client = fake
    adapter._closed = False
    assert adapter.base_pose_is_free(1.5, 2.5) is True
    assert fake.calls == [(1.5, 2.5, 0.0)]


def test_adapter_plan_approach_pose_in_link0_does_not_need_client() -> None:
    adapter = RobocasaAdapter()
    cloud = PointCloud(
        points=np.array([[0.8, 0.0, 0.9], [0.82, 0.01, 0.9], [0.79, -0.01, 0.9]]),
        frame="robot0_link0",
    )
    x, y, yaw = adapter.plan_approach_pose(cloud, 0.3)
    assert math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw)
