from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from cap_harness.api import CapApi
from cap_harness.contracts import (
    CameraObservation,
    IKResult,
    Observation,
    PointCloud,
    Pose,
    RobotState,
    StepResult,
    SynchronizedTrajectory,
    Trajectory,
)
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.registry import ROBOCASA_PUBLIC_TOOL_NAMES, ToolRegistry
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


#: The real robot spawns near this yaw, which is what made a zero default wrong.
SPAWN_YAW = math.pi / 2


class _FakeFootprintClient:
    """Enough of the driver for the base queries: a cached state and a footprint oracle.

    ``blocked_x`` lists world x values whose footprint is occupied, so a test can place a
    counter exactly where the sweep wants to stand.
    """

    def __init__(self, yaw: float = 0.0, blocked_x: tuple[float, ...] = ()) -> None:
        self.calls: list[tuple[float, float, float]] = []
        self.blocked_x = {round(value, 2) for value in blocked_x}
        self.steps = 0
        self.last = {
            "base_pos": np.array([1.0, 2.0, 0.7]),
            "base_quat_wxyz": np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]),
            "joint_pos": np.zeros(7),
            "gripper_qpos": np.array([0.04]),
            "eef_pos": np.array([0.4, 0.0, 1.0]),
            "eef_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
        }

    def base_pose_is_free(self, x: float, y: float, yaw: float = 0.0) -> bool:
        self.calls.append((float(x), float(y), float(yaw)))
        return round(float(x), 2) not in self.blocked_x

    def sim_step_count(self) -> int:
        return self.steps

    def step(self, action: object) -> object:  # pragma: no cover - must never be reached
        raise AssertionError("the base queries must not drive the simulator")


class _StubIkProvider:
    """IK whose feasibility is a pure function of the target's y in the arm's frame.

    The arm reaches ``|y| <= reach_m``, which is the whole point of the sweep: a target
    outside that band becomes reachable only once the base has moved under it.
    """

    def __init__(self, reach_m: float = 0.30) -> None:
        self.reach_m = float(reach_m)
        self.requested: list[np.ndarray] = []

    def solve_ik(
        self, target_pose: Pose, robot_state: RobotState, arm: str = "primary"
    ) -> IKResult:
        self.requested.append(np.asarray(target_pose.position, dtype=np.float64).copy())
        if abs(float(target_pose.position[1])) <= self.reach_m:
            return IKResult(ok=True, arm=arm, embodiment="robocasa", joint_positions=np.zeros(7))
        return IKResult(
            ok=False,
            arm=arm,
            embodiment="robocasa",
            error=ApiError(code=ErrorCode.IK_FAILED, message="outside the reach band"),
        )


def _base_adapter(
    *, yaw: float = SPAWN_YAW, blocked_x: tuple[float, ...] = (), reach_m: float = 0.30
) -> tuple[RobocasaAdapter, _FakeFootprintClient, _StubIkProvider]:
    provider = _StubIkProvider(reach_m)
    adapter = RobocasaAdapter(ik_provider=provider)
    adapter._metadata = adapter.registry.resolve("PickPlaceCounterToStove")
    client = _FakeFootprintClient(yaw=yaw, blocked_x=blocked_x)
    adapter._client = client
    adapter._closed = False
    return adapter, client, provider


def _target(y: float) -> Pose:
    return Pose(
        position=np.array([0.5, float(y), 0.9]),
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        frame="robot0_link0",
    )


def test_adapter_base_pose_is_free_is_world_yes_no() -> None:
    adapter, client, _ = _base_adapter(yaw=0.0)
    assert adapter.base_pose_is_free(1.5, 2.5) is True
    assert client.calls == [(1.5, 2.5, 0.0)]


def test_base_pose_is_free_defaults_to_the_current_base_yaw() -> None:
    """A zero default tested a box turned across the counter and blocked every valid pose."""
    adapter, client, _ = _base_adapter(yaw=SPAWN_YAW)
    assert adapter.base_pose_is_free(1.5, 2.5) is True
    assert client.calls[-1][2] == pytest.approx(SPAWN_YAW, abs=1e-9)


def test_base_pose_is_free_still_accepts_an_explicit_yaw() -> None:
    adapter, client, _ = _base_adapter(yaw=SPAWN_YAW)
    adapter.base_pose_is_free(1.5, 2.5, 0.0)
    assert client.calls[-1] == (1.5, 2.5, 0.0)


def test_plan_base_offset_shifts_targets_opposite_the_base_move() -> None:
    """The one error that would be silent: the sign and frame of the shift.

    A target at y=+0.55 is outside the arm's band. The base must move toward it -- positive
    dy -- which brings the target's y down in ``robot0_link0``. An inverted sign would sweep
    away from every target and rescue nothing.
    """
    adapter, _, provider = _base_adapter()
    result = adapter.plan_base_offset([_target(0.55)], standoff_m=0.0)
    assert result["ok"] is True
    assert result["reachable_now"] == 0
    assert result["reachable_after"] == 1
    assert float(result["dy"]) > 0.0
    assert float(result["dy"]) == pytest.approx(0.30, abs=0.051)
    assert result["dx"] == 0.0 and result["dyaw"] == 0.0
    # Every pose actually handed to IK kept the target's x and z and moved only y.
    for position in provider.requested:
        assert position[0] == pytest.approx(0.5)
        assert position[2] == pytest.approx(0.9)


def test_plan_base_offset_stands_still_when_targets_are_already_reachable() -> None:
    """Moving costs ticks and shifts the robot-mounted camera, so it must not be gratuitous."""
    adapter, client, provider = _base_adapter()
    result = adapter.plan_base_offset([_target(0.10)], standoff_m=0.0)
    assert result["ok"] is True
    assert result["dy"] == 0.0
    assert result["reachable_now"] == 1 and result["reachable_after"] == 1
    assert result["offsets_tested"] == 1
    assert len(provider.requested) == 1, "no sweep should have been run"
    assert client.calls == [], "no footprint query should have been needed"


def test_plan_base_offset_never_returns_an_occupied_offset() -> None:
    """Do not tell a program to stand inside a counter.

    The base sits at world (1.0, 2.0) facing +y, so a lateral dy maps to world x = 1.0 - dy.
    Blocking the x values of every rescuing offset leaves the sweep with nothing to offer.
    """
    rescuing_x = tuple(round(1.0 - dy / 100.0, 2) for dy in range(25, 41))
    adapter, client, _ = _base_adapter(blocked_x=rescuing_x)
    result = adapter.plan_base_offset([_target(0.55)], standoff_m=0.0)
    assert result["ok"] is False
    assert result["dy"] == 0.0, "a blocked offset must never be the answer"
    assert client.calls, "the sweep must have consulted the footprint"
    blocked = [entry for entry in result["offsets"] if entry["free"] is False]
    assert blocked, "blocked offsets stay visible in the diagnostics"
    assert all(entry["reachable"] is None for entry in blocked), "no IK spent on a blocked pose"


def test_plan_base_offset_spends_no_simulator_ticks() -> None:
    """The whole point: a program can ask where to stand without paying from its horizon."""
    adapter, client, _ = _base_adapter()
    before = adapter._elapsed_steps
    adapter.plan_base_offset([_target(0.55)], standoff_m=0.0)
    assert adapter._elapsed_steps == before
    assert client.steps == 0


def test_plan_base_offset_counts_the_retreat_pose_too() -> None:
    """A candidate is executable only if the approach it must pass through also solves."""
    adapter, _, _ = _base_adapter(reach_m=0.30)
    # Rotated 90 degrees about x, so the gripper's approach axis (local +z) is -y: the
    # retreat pose sits 0.20 m further out in y than the grasp, which puts it outside the
    # band while the grasp itself is inside it.
    rotated = Pose(
        position=np.array([0.5, 0.25, 0.9]),
        quaternion_wxyz=np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]),
        frame="robot0_link0",
    )
    bare = adapter.plan_base_offset([rotated], standoff_m=0.0)
    assert bare["reachable_now"] == 1

    adapter2, _, _ = _base_adapter(reach_m=0.30)
    strict = adapter2.plan_base_offset([rotated], standoff_m=0.20)
    assert strict["reachable_now"] == 0, "the retreat pose is out of reach, so this is not usable"
    assert strict["standoff_m"] == pytest.approx(0.20)


def test_plan_base_offset_rejects_bad_requests() -> None:
    adapter, _, _ = _base_adapter()
    world = Pose(
        position=np.array([1.0, 2.0, 0.9]),
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        frame="world",
    )
    with pytest.raises(ValueError, match="robot0_link0"):
        adapter.plan_base_offset([world])
    with pytest.raises(ValueError):
        adapter.plan_base_offset([])
    with pytest.raises(ValueError):
        adapter.plan_base_offset([_target(0.1)], min_reachable=0)


def test_plan_base_offset_is_a_public_robocasa_extension() -> None:
    """Allowlisted, and reachable through the mapping a generated program is handed."""
    assert "robocasa.plan_base_offset" in ROBOCASA_PUBLIC_TOOL_NAMES
    adapter, _, _ = _base_adapter()
    registry = CapApi(adapter).register_tools(
        ToolRegistry(public_extension_allowlist=ROBOCASA_PUBLIC_TOOL_NAMES)
    )
    assert "robocasa.plan_base_offset" in dict(registry.public_tools())


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
