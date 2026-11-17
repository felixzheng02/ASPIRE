from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from cap_harness.contracts import (
    CameraObservation,
    Observation,
    PointCloud,
    Pose,
    RobotState,
    StepResult,
    SynchronizedTrajectory,
    Trajectory,
)
from cap_harness.robocasa.adapter import (
    JOINT_LIMIT_SLACK_RAD,
    PANDA_JOINT_NAMES,
    TRAJECTORY_MAX_JOINT_STEP_RAD,
    RobocasaAdapter,
    densify_joint_waypoints,
    validate_joint_limits,
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


def test_densify_does_not_add_a_tick_for_float_noise_at_an_exact_multiple() -> None:
    """0.1 rad at a 0.05 rad step is 2 ticks, even when the subtraction leaves noise."""
    start = np.full(7, 0.2)
    goal = np.full(7, 0.2)
    goal[0] = 0.1 + 0.2  # 0.30000000000000004
    span = goal[0] - start[0]
    assert span / 0.05 > 2.0, "the test must exercise the noisy ratio"
    out = densify_joint_waypoints(np.vstack([start, goal]), max_step_rad=0.05)
    assert out.shape[0] == 3  # start + 2 ticks
    np.testing.assert_allclose(out[1, 0], 0.25)
    np.testing.assert_allclose(out[-1], goal)


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


def _pose() -> Pose:
    return Pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), "robot0_link0")


def _state(joints: np.ndarray) -> RobotState:
    target = np.asarray(joints, dtype=np.float64).reshape(7)
    return RobotState(
        joint_positions={"primary": target},
        joint_velocities={"primary": np.zeros(7)},
        end_effector_poses={"primary": _pose()},
        gripper_positions={"primary": 1.0},
        base_frame="robot0_link0",
        embodiment="robocasa",
    )


def _observation(joints: np.ndarray) -> Observation:
    pose = _pose()
    return Observation(
        cameras={
            "agentview": CameraObservation(
                rgb=np.zeros((8, 8, 3), dtype=np.uint8),
                depth_m=np.ones((8, 8)),
                intrinsics=np.eye(3),
                frame="camera/agentview",
                camera_pose=pose,
            )
        },
        robot_state=_state(joints),
    )


def _trajectory(points: np.ndarray) -> Trajectory:
    start = _state(points[0])
    return Trajectory(
        joint_positions=np.asarray(points, dtype=np.float64),
        dt_s=0.05,
        joint_names=tuple(start.joint_names["primary"]),
        planner="test",
        collision_aware=False,
        expected_start=start,
        arm="primary",
        embodiment="robocasa",
    )


def _adapter_with_recording_step() -> tuple[RobocasaAdapter, list]:
    adapter = RobocasaAdapter()
    adapter._metadata = adapter.registry.resolve("OpenDrawer")
    calls: list = []

    def fake_step(action):
        target = np.asarray(action.arms["primary"].target, dtype=np.float64)
        calls.append(action)
        adapter._elapsed_steps += 1
        return StepResult(ok=True, observation=_observation(target))

    adapter.step = fake_step  # type: ignore[method-assign]
    return adapter, calls


def test_execute_trajectory_plays_waypoints_through_step() -> None:
    """Video frames are recorded in ``step()``; the driver loop used to skip it."""
    adapter, calls = _adapter_with_recording_step()
    points = np.vstack([np.zeros(7), np.array([0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])])
    dense = densify_joint_waypoints(points)
    result = adapter.execute_trajectory(_trajectory(points), settle_steps=4)
    assert result.ok
    assert len(calls) == dense.shape[0]
    assert result.steps_executed == dense.shape[0]
    np.testing.assert_allclose(calls[-1].arms["primary"].target, dense[-1])


def test_execute_trajectory_ramps_from_the_start_state_to_a_far_first_waypoint() -> None:
    """A hand-built path whose waypoint 0 is 0.2 rad from the start is not crossed in one tick."""
    adapter, calls = _adapter_with_recording_step()
    first = np.array([0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    points = np.vstack([first, first])
    start = _state(np.zeros(7))
    trajectory = Trajectory(
        joint_positions=points,
        dt_s=0.05,
        joint_names=tuple(start.joint_names["primary"]),
        planner="test",
        collision_aware=False,
        expected_start=start,
        arm="primary",
        embodiment="robocasa",
    )
    result = adapter.execute_trajectory(trajectory, settle_steps=4)
    assert result.ok
    targets = np.vstack([np.zeros(7)] + [c.arms["primary"].target for c in calls])
    assert np.max(np.abs(np.diff(targets, axis=0))) <= TRAJECTORY_MAX_JOINT_STEP_RAD + 1e-12
    # 4 ramp ticks reach waypoint 0, then waypoint 1 (no move) is played as its own tick.
    assert result.steps_executed == 5
    np.testing.assert_allclose(calls[3].arms["primary"].target, first)


def test_execute_trajectory_keeps_the_tick_count_of_a_path_that_starts_at_the_measured_state() -> (
    None
):
    """Planner paths begin at ``expected_start``; ramping from it must add nothing."""
    adapter, calls = _adapter_with_recording_step()
    points = np.vstack([np.zeros(7), np.full(7, 0.04), np.full(7, 0.08)])
    result = adapter.execute_trajectory(_trajectory(points), settle_steps=4)
    assert result.ok
    assert result.steps_executed == points.shape[0]
    np.testing.assert_allclose(calls[0].arms["primary"].target, points[0])


def _sync_trajectory(points: np.ndarray, arm: str = "primary") -> SynchronizedTrajectory:
    """What ``CapApi.move_synchronized`` hands the adapter after planning."""
    start = _state(points[0]) if arm == "primary" else _state_for_arm(points[0], arm)
    return SynchronizedTrajectory(
        joint_positions={arm: np.asarray(points, dtype=np.float64)},
        dt_s=0.05,
        joint_names={arm: tuple(start.joint_names[arm])},
        planner="test",
        collision_aware=False,
        expected_start=start,
        embodiment="robocasa",
    )


def _state_for_arm(joints: np.ndarray, arm: str) -> RobotState:
    target = np.asarray(joints, dtype=np.float64).reshape(7)
    return RobotState(
        joint_positions={arm: target},
        joint_velocities={arm: np.zeros(7)},
        end_effector_poses={arm: _pose()},
        gripper_positions={arm: 1.0},
        base_frame="robot0_link0",
        embodiment="robocasa",
    )


def test_execute_trajectory_accepts_a_single_arm_synchronized_trajectory() -> None:
    """``move_synchronized`` plans a SynchronizedTrajectory and executes it unchanged.

    Rejecting the type here made the tool structurally impossible on this one-arm
    embodiment even though ``plan_synchronized_motion`` succeeded.
    """
    adapter, calls = _adapter_with_recording_step()
    points = np.vstack([np.zeros(7), np.array([0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])])
    dense = densify_joint_waypoints(points)
    result = adapter.execute_trajectory(_sync_trajectory(points), settle_steps=4)
    assert result.ok
    assert len(calls) == dense.shape[0]
    assert result.steps_executed == dense.shape[0]
    np.testing.assert_allclose(calls[-1].arms["primary"].target, dense[-1])


def test_execute_trajectory_matches_the_single_arm_path_waypoint_for_waypoint() -> None:
    points = np.vstack([np.zeros(7), np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0, 0.0])])
    plain_adapter, plain_calls = _adapter_with_recording_step()
    plain_adapter.execute_trajectory(_trajectory(points), settle_steps=4)
    sync_adapter, sync_calls = _adapter_with_recording_step()
    sync_adapter.execute_trajectory(_sync_trajectory(points), settle_steps=4)
    assert len(sync_calls) == len(plain_calls)
    for sync_call, plain_call in zip(sync_calls, plain_calls, strict=True):
        np.testing.assert_allclose(
            sync_call.arms["primary"].target, plain_call.arms["primary"].target
        )


def test_execute_trajectory_carries_synchronized_gripper_positions() -> None:
    adapter, calls = _adapter_with_recording_step()
    points = np.vstack([np.zeros(7), np.array([0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])])
    trajectory = _sync_trajectory(points)
    trajectory = SynchronizedTrajectory(
        joint_positions=trajectory.joint_positions,
        dt_s=trajectory.dt_s,
        joint_names=trajectory.joint_names,
        planner=trajectory.planner,
        collision_aware=trajectory.collision_aware,
        expected_start=trajectory.expected_start,
        gripper_positions={"primary": np.array([1.0, 0.0])},
        embodiment="robocasa",
    )
    result = adapter.execute_trajectory(trajectory, settle_steps=4)
    assert result.ok
    # The single-arm path holds the final commanded width for every densified tick.
    assert all(call.arms["primary"].gripper_position == 0.0 for call in calls)


def test_execute_trajectory_rejects_a_synchronized_trajectory_for_another_arm() -> None:
    adapter, _ = _adapter_with_recording_step()
    points = np.vstack([np.zeros(7), np.full(7, 0.02)])
    with pytest.raises(ValueError, match="must name exactly the arms"):
        adapter.execute_trajectory(_sync_trajectory(points, arm="left"))


def test_execute_trajectory_still_rejects_other_types() -> None:
    adapter, _ = _adapter_with_recording_step()
    with pytest.raises(ValueError, match="Trajectory or SynchronizedTrajectory"):
        adapter.execute_trajectory(object())  # type: ignore[arg-type]


def test_close_gripper_settles_through_step() -> None:
    adapter, calls = _adapter_with_recording_step()
    adapter._client = SimpleNamespace(profile=SimpleNamespace(gripper_settle_steps=4))
    adapter._closed = False
    adapter.get_robot_state = lambda: _state(np.zeros(7))  # type: ignore[method-assign]
    adapter._measured_gripper = lambda: 0.0  # type: ignore[method-assign]
    result = adapter.close_gripper()
    assert result.ok
    assert len(calls) == 4
    assert all(call.arms["primary"].gripper_position == 0.0 for call in calls)


# --------------------------------------------------------------- joint limits

#: What the simulator published before the fix: ``model.jnt_range[:7]``, i.e. the first seven
#: joints of the whole kitchen model -- three mobile-base joints (zero-width, MuJoCo's
#: unlimited convention), the torso, then Panda joints 1-3.
_SLICED_LIMITS = {
    "low": [0.0, 0.0, 0.0, 0.0, -2.8973, -1.7628, -2.8973],
    "high": [0.0, 0.0, 0.0, 0.34, 2.8973, 1.7628, 2.8973],
}
#: The arm's own limits, resolved from its named joints.
_PANDA_LIMITS = {
    "low": [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    "high": [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    "names": [f"robot0_joint{index}" for index in range(1, 8)],
}
#: Measured on PickPlaceCounterToStove, seed 101.
_MEASURED_JOINTS = np.array([0.0023, -0.2233, -0.0054, -0.5032, 0.0088, 0.3746, 0.1989])


def test_validate_joint_limits_accepts_the_arms_own_limits() -> None:
    out = validate_joint_limits(_PANDA_LIMITS, arm_dof=7, measured=_MEASURED_JOINTS)
    assert out["low"] == _PANDA_LIMITS["low"]
    assert out["high"] == _PANDA_LIMITS["high"]
    assert out["names"] == _PANDA_LIMITS["names"]


def test_validate_joint_limits_rejects_the_leading_model_slice() -> None:
    """The original bug: four of seven measured joints outside their published range."""
    with pytest.raises(ValueError, match="empty range"):
        validate_joint_limits(_SLICED_LIMITS, arm_dof=7, measured=_MEASURED_JOINTS)


def test_validate_joint_limits_rejects_a_range_missing_the_measurement() -> None:
    shifted = {
        "low": _PANDA_LIMITS["low"][3:] + _PANDA_LIMITS["low"][:3],
        "high": _PANDA_LIMITS["high"][3:] + _PANDA_LIMITS["high"][:3],
        "names": _PANDA_LIMITS["names"],
    }
    with pytest.raises(ValueError, match="measured configuration"):
        validate_joint_limits(shifted, arm_dof=7, measured=_MEASURED_JOINTS)


def test_validate_joint_limits_tolerates_solver_drift_past_a_limit() -> None:
    measured = np.array(_PANDA_LIMITS["high"], dtype=np.float64)
    measured[0] += 0.5 * JOINT_LIMIT_SLACK_RAD
    assert validate_joint_limits(_PANDA_LIMITS, arm_dof=7, measured=measured)


def test_validate_joint_limits_rejects_wrong_shapes_and_types() -> None:
    with pytest.raises(ValueError, match="mapping"):
        validate_joint_limits(None, arm_dof=7)
    with pytest.raises(ValueError, match="missing 'high'"):
        validate_joint_limits({"low": _PANDA_LIMITS["low"]}, arm_dof=7)
    with pytest.raises(ValueError, match="describes"):
        validate_joint_limits({"low": [0.0], "high": [1.0]}, arm_dof=7)
    with pytest.raises(ValueError, match="non-finite"):
        validate_joint_limits({"low": [-math.inf] * 7, "high": _PANDA_LIMITS["high"]}, arm_dof=7)


def _adapter_with_spec(limits: dict) -> RobocasaAdapter:
    adapter = RobocasaAdapter()
    adapter._client = SimpleNamespace(
        spec={"action_dim": 13, "arm_dof": 7, "joint_limits": limits},
        last={"joint_pos": _MEASURED_JOINTS},
    )
    adapter._closed = False
    return adapter


def test_adapter_publishes_checked_arm_limits() -> None:
    adapter = _adapter_with_spec(_PANDA_LIMITS)
    limits = adapter._published_joint_limits(adapter._client.spec)
    assert limits["names"] == _PANDA_LIMITS["names"]
    assert all(limits["low"][i] <= _MEASURED_JOINTS[i] <= limits["high"][i] for i in range(7))


def test_adapter_refuses_limits_that_exclude_the_arm() -> None:
    """get_controller_metadata() must not hand a program the old sliced array."""
    adapter = _adapter_with_spec(_SLICED_LIMITS)
    with pytest.raises(ValueError, match="joint_limits"):
        adapter._published_joint_limits(adapter._client.spec)


def test_adapter_checks_structure_before_the_first_observation() -> None:
    """Metadata is legal to ask for before any observation; only containment is skipped."""
    adapter = _adapter_with_spec(_PANDA_LIMITS)
    adapter._client = SimpleNamespace(spec=adapter._client.spec, last=None)
    limits = adapter._published_joint_limits(adapter._client.spec)
    assert limits["low"] == _PANDA_LIMITS["low"]
    adapter = _adapter_with_spec(_SLICED_LIMITS)
    adapter._client = SimpleNamespace(spec=adapter._client.spec, last=None)
    with pytest.raises(ValueError, match="empty range"):
        adapter._published_joint_limits(adapter._client.spec)
