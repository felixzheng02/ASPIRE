"""RoboCasa environment adapter: ASPIRE's typed contracts over the CaP-X robot driver.

Unlike every other adapter in this package, this one owns **no simulator**. RoboCasa runs in its
own process with its own MuJoCo/robosuite/robocasa environment and is reached over a ZeroMQ
``step(action)`` / observation loop, so the boundary here is the same one a real robot would
present. That is why this file is short where ``robosuite/adapter.py`` is long: everything the
Robosuite adapter derives from ``env.sim`` -- camera intrinsics, extrinsics, frame transforms --
arrives over the wire already computed by the side that legitimately has ``sim``.

Consequences worth knowing before reading on:

* ``native_env`` does not exist. Nothing on this side may reach into simulator state.
* The simulator's scoring channel (``evaluate``) is reachable from the adapter, which is
  host-side, and is carried on ``StepResult.reward``. ``public_view()`` strips it, exactly as
  for the in-process adapters, so generated programs never see it.
* ``joint_velocities`` are reported as zeros: the wire protocol carries no velocity channel.
  See the note on :meth:`get_robot_state`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
from types import MappingProxyType
from typing import Any

import numpy as np

from cap_harness.contracts import (
    ArmCommand,
    CameraObservation,
    ExecutionResult,
    Observation,
    Pose,
    RobotAction,
    RobotPlanningContext,
    RobotState,
    StepResult,
    SynchronizedTrajectory,
    TaskContext,
    Trajectory,
)
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.geometry import invert_transform, matrix_to_pose, quaternion_wxyz_to_matrix
from cap_harness.robocasa.codec import RobocasaActionCodec
from cap_harness.robocasa.geometry import (
    plan_approach_pose as approach_from_cloud,
)
from cap_harness.robocasa.geometry import (
    plan_standoff_pose as standoff_from_clouds,
)
from cap_harness.robocasa.registry import (
    RobocasaTaskMetadata,
    RobocasaTaskRegistry,
    scene_group_for_seed,
)

#: CapApi interpolates at this scale; two sparse waypoints 0.10 rad apart miss
#: the 0.02 rad execute_trajectory tolerance (trap 17).
TRAJECTORY_MAX_JOINT_STEP_RAD = 0.05
#: Extra ``step()`` ticks holding the last waypoint until joint error is inside tolerance.
TRAJECTORY_GOAL_MAX_STEPS = 120
#: cuRobo's panda model names. The public RobotState still reports joint_0..6.
PANDA_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
#: MuJoCo's solver lets a joint settle a hair past a limit, so an exact containment test on a
#: measured configuration would fire on physics rather than on a bad limit array. This slack is
#: wide enough never to trip on drift and far narrower than the errors worth catching (the
#: original bug published a zero-width range and a 0.5 rad violation).
JOINT_LIMIT_SLACK_RAD = 0.05


def validate_joint_limits(
    limits: object,
    *,
    arm_dof: int,
    measured: np.ndarray | None = None,
    slack_rad: float = JOINT_LIMIT_SLACK_RAD,
) -> dict[str, list[float] | list[str]]:
    """Return the simulator's ``joint_limits`` only if they can describe this arm.

    The simulator is the only producer of this array and it lives outside any repository, so
    this is the one reviewable place a regression can be caught. It has happened: the simulator
    published ``model.jnt_range[:7]``, the first seven joints of the whole kitchen model, which
    on a PandaOmron are three mobile-base joints (zero-width, MuJoCo's unlimited convention),
    the torso, and then Panda joints 1-3. Four of the seven measured joints sat outside their
    own published range, and a program clamping its targets to it would have driven the first
    four joints to zero.

    Raises ``ValueError`` rather than dropping the key: a program reading no limits writes a
    different (and safer) policy than one reading wrong limits.
    """
    if not isinstance(limits, Mapping):
        raise ValueError(f"joint_limits must be a mapping, got {type(limits).__name__}")
    try:
        low = np.asarray(limits["low"], dtype=np.float64).reshape(-1)
        high = np.asarray(limits["high"], dtype=np.float64).reshape(-1)
    except KeyError as exc:
        raise ValueError(f"joint_limits is missing {exc.args[0]!r}") from exc
    if low.size != arm_dof or high.size != arm_dof:
        raise ValueError(
            f"joint_limits describes {low.size}/{high.size} joints, the arm has {arm_dof}"
        )
    if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
        raise ValueError("joint_limits contains non-finite bounds")
    names = [str(n) for n in limits.get("names", ())] or [
        f"joint{index}" for index in range(1, arm_dof + 1)
    ]
    if len(names) != arm_dof:
        raise ValueError(f"joint_limits names {len(names)} joints, the arm has {arm_dof}")
    empty = [names[i] for i in np.flatnonzero(high - low <= 0.0)]
    if empty:
        raise ValueError(f"joint_limits has an empty range for {empty}")
    if measured is not None:
        values = np.asarray(measured, dtype=np.float64).reshape(-1)
        if values.size == arm_dof:
            outside = np.flatnonzero((values < low - slack_rad) | (values > high + slack_rad))
            if outside.size:
                detail = ", ".join(
                    f"{names[i]}={values[i]:.4f} not in [{low[i]:.4f}, {high[i]:.4f}]"
                    for i in outside
                )
                raise ValueError(
                    f"joint_limits does not contain the measured configuration: {detail}"
                )
    out: dict[str, list[float] | list[str]] = {
        "low": [float(x) for x in low],
        "high": [float(x) for x in high],
    }
    if limits.get("names"):
        out["names"] = names
    return out


def densify_joint_waypoints(
    waypoints: np.ndarray,
    *,
    max_step_rad: float = TRAJECTORY_MAX_JOINT_STEP_RAD,
) -> np.ndarray:
    """Insert linearly interpolated joints so no step exceeds ``max_step_rad``."""
    points = np.asarray(waypoints, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0:
        return points
    if not np.isfinite(max_step_rad) or max_step_rad <= 0.0:
        raise ValueError("max_step_rad must be positive and finite")
    # A segment that is an exact multiple of the step, plus float noise (0.30000000000000004 -
    # 0.2), must not gain a tick: a bare ceil turns 0.1 rad at 0.05 into 3 ticks, not 2.
    ratio_eps = 1e-9
    densified = [points[0]]
    for nxt in points[1:]:
        prev = densified[-1]
        delta = nxt - prev
        span = float(np.max(np.abs(delta)))
        count = max(1, int(np.ceil(span / max_step_rad - ratio_eps)))
        for index in range(1, count + 1):
            densified.append(prev + (index / count) * delta)
        densified[-1] = nxt
    return np.asarray(densified, dtype=np.float64)


def _transform_from_quaternion_and_position(quaternion_wxyz: Any, position: Any) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = quaternion_wxyz_to_matrix(quaternion_wxyz)
    transform[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return transform


def _yaw_from_wxyz(quaternion_wxyz: Any) -> float:
    w, x, y, z = (float(v) for v in np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class RobocasaAdapter:
    """``EnvironmentAdapter`` for RoboCasa over the CaP-X wire protocol."""

    #: PandaOmron's nominal root ``robot0_base`` is a dummy parked at world [10, 10, 0]; the
    #: real arm base is ``robot0_link0``. Using the decoy puts every pose ~14 m away, silently.
    BASE_FRAME = "robot0_link0"

    #: This gripper is commanded fully open or fully closed; intermediate targets are not held.
    gripper_holds_intermediate = False

    def __init__(
        self,
        *,
        registry: RobocasaTaskRegistry | None = None,
        client_factory: Callable[..., Any] | None = None,
        endpoint: str | None = None,
        timeout_ms: int = 120_000,
        control_frequency: float = 20.0,
        camera_height: int = 128,
        camera_width: int = 128,
        horizon: int = 1000,
        run_observer: Any | None = None,
    ) -> None:
        if camera_height <= 0 or camera_width <= 0 or horizon <= 0:
            raise ValueError("camera dimensions and horizon must be positive")
        self.registry = registry or RobocasaTaskRegistry()
        self._client_factory = client_factory
        self._endpoint = endpoint
        self._timeout_ms = int(timeout_ms)
        self.control_frequency = float(control_frequency)
        if not np.isfinite(self.control_frequency) or self.control_frequency <= 0:
            raise ValueError("control_frequency must be positive and finite")
        self.control_period_s = 1.0 / self.control_frequency
        # Recorded for the manifest only. The simulator renders at the size it was started
        # with; a client cannot resize another process's framebuffer, and get_spec reports
        # the truth, which is what _camera_observation uses.
        self.camera_height = int(camera_height)
        self.camera_width = int(camera_width)
        self.horizon = int(horizon)
        self.controller = "JOINT_POSITION"
        self._run_observer = run_observer
        self._client: Any | None = None
        self._metadata: RobocasaTaskMetadata | None = None
        self._task_context: TaskContext | None = None
        self._codec: RobocasaActionCodec | None = None
        self._camera_map: dict[str, str] = {}
        self._elapsed_steps = 0
        self._seed: int | None = None
        self._last_reward: float | None = None
        self._closed = False

    # ------------------------------------------------------------------ identity

    @property
    def embodiment(self) -> str:
        return "robocasa"

    @property
    def current_time_s(self) -> float:
        return self._elapsed_steps * self.control_period_s

    @property
    def metadata(self) -> RobocasaTaskMetadata:
        if self._metadata is None:
            raise RuntimeError("reset() must be called before reading metadata")
        return self._metadata

    @property
    def arms(self) -> tuple[str, ...]:
        return self.metadata.arms

    @property
    def native_env(self) -> Any:
        raise AttributeError(
            "RoboCasa runs in another process: there is no native environment on this side, "
            "which is the point of the boundary. Host-side scoring goes through evaluate()."
        )

    # ------------------------------------------------------------------ plumbing

    def _require_client(self) -> Any:
        if self._client is None or self._closed:
            raise RuntimeError("reset() must be called before using the adapter")
        return self._client

    def _make_client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory(endpoint=self._endpoint, timeout_ms=self._timeout_ms)
        from robot_driver import RobotClient
        from robot_driver import protocol as wire

        return RobotClient(self._endpoint or wire.DEFAULT_ENDPOINT, timeout_ms=self._timeout_ms)

    def _build_camera_map(self, metadata: RobocasaTaskMetadata) -> dict[str, str]:
        """Map ASPIRE's portable camera names onto whatever this simulator advertises.

        ``get_spec`` reports the simulator's own names under the roles ``scene`` and ``wrist``,
        so the mapping is queried rather than hardcoded -- a different kitchen, robot or
        render size cannot silently break it.
        """
        cameras = self._require_client().spec["cameras"]
        resolved: dict[str, str] = {}
        for name in metadata.camera_names:
            if name == "agentview":
                resolved[name] = cameras["scene"]
            elif name in ("robot0_eye_in_hand", "eye_in_hand"):
                resolved[name] = cameras["wrist"]
            else:
                raise ValueError(
                    f"camera {name!r} is not advertised by the simulator; "
                    f"it reports scene={cameras['scene']!r} wrist={cameras['wrist']!r}"
                )
        return resolved

    # --------------------------------------------------------------------- reset

    def reset(
        self,
        task_ref: str | tuple[str, int] | RobocasaTaskMetadata,
        seed: int,
    ) -> Observation:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        metadata = self.registry.resolve(task_ref)
        if self._client is None:
            self._client = self._make_client()
            self._closed = False
        client = self._require_client()

        # The seed crosses the wire with the group it belongs to: only the simulator can act
        # on a seed, but which group a seed draws from is this harness's partition, not its.
        episode = client.reset(seed=seed, scene_group=scene_group_for_seed(seed))
        self._seed = seed
        self._metadata = metadata
        self._elapsed_steps = 0
        self._last_reward = None
        self._camera_map = self._build_camera_map(metadata)
        self._codec = RobocasaActionCodec(metadata.arms, control_frequency=self.control_frequency)
        # Per-episode instruction from the simulator is authoritative; RoboCasa composes it from
        # the sampled objects, so it is not knowable from the static registry.
        language = str(episode.get("instruction") or metadata.language)
        self._task_context = TaskContext(
            suite=metadata.suite_name,
            task_id=metadata.task_id,
            task_name=metadata.task_name,
            language=language,
            family=metadata.family,
            metadata={"task_ref": metadata.task_ref},
        )
        observation = self.get_observation()
        if self._run_observer is not None:
            self._run_observer.on_reset(observation, metadata.to_manifest_record())
        return observation

    # -------------------------------------------------------------- observations

    def _base_from_world(self) -> np.ndarray:
        last = self._require_client().last
        return invert_transform(
            _transform_from_quaternion_and_position(last["base_quat_wxyz"], last["base_pos"])
        )

    def _camera_observation(self, name: str) -> CameraObservation:
        client = self._require_client()
        native = self._camera_map[name]
        depth = client.get_depth(native)
        if name == "agentview" and "rgb" in (client.last or {}):
            # reset()/step() already carry the scene frame; re-rendering it would be a second
            # round trip and a second render for an image we already hold.
            rgb = np.asarray(client.last["rgb"])
        else:
            rgb = np.asarray(client.render(native))
        depth_m = np.asarray(depth["depth"], dtype=np.float64)
        if depth_m.ndim == 3:
            depth_m = depth_m[..., 0]
        return CameraObservation(
            rgb=np.ascontiguousarray(rgb),
            depth_m=np.ascontiguousarray(depth_m),
            intrinsics=np.asarray(depth["intrinsics"], dtype=np.float64),
            frame=f"camera/{native}",
            # Already expressed in the robot base frame by the simulator, including the
            # OpenGL->vision axis correction robosuite's cameras need.
            camera_pose=matrix_to_pose(
                np.asarray(depth["cam_pose_mat"], dtype=np.float64), frame=self.BASE_FRAME
            ),
            timestamp_s=self.current_time_s,
        )

    def _measured_gripper(self) -> float:
        qpos = np.asarray(self._require_client().last["gripper_qpos"], dtype=np.float64).reshape(-1)
        if qpos.size == 0:
            return 1.0
        return float(np.clip(qpos[0] / 0.04, 0.0, 1.0))

    def get_robot_state(self) -> RobotState:
        client = self._require_client()
        last = client.last
        arm = self.arms[0]
        joints = np.asarray(last["joint_pos"], dtype=np.float64).reshape(-1)[:7]
        eef_in_base = self._base_from_world() @ _transform_from_quaternion_and_position(
            last["eef_quat_wxyz"], last["eef_pos"]
        )
        return RobotState(
            joint_positions={arm: joints},
            # The driver protocol carries no velocity channel, so these are zeros rather than a
            # measurement. Nothing in the RoboCasa tool surface consumes them; if a planner ever
            # does, the honest fix is a velocity field on the wire, not a finite difference here.
            joint_velocities={arm: np.zeros_like(joints)},
            end_effector_poses={arm: matrix_to_pose(eef_in_base, frame=self.BASE_FRAME)},
            gripper_positions={arm: self._measured_gripper()},
            base_frame=self.BASE_FRAME,
            timestamp_s=self.current_time_s,
            embodiment=self.embodiment,
        )

    def get_task_context(self) -> TaskContext:
        if self._task_context is None:
            raise RuntimeError("reset() must be called before reading the task context")
        return self._task_context

    def get_planning_context(self) -> RobotPlanningContext:
        """Panda calibration in ``robot0_link0``. Identity because the fused
        planning scene already lives in ``BASE_FRAME`` — never ``robot0_base``.
        """
        arm = "primary"
        return RobotPlanningContext(
            embodiment=self.embodiment,
            model="panda",
            joint_names={arm: PANDA_JOINT_NAMES},
            base_transforms={arm: np.eye(4, dtype=np.float64)},
            end_effector_links={arm: "panda_hand"},
        )

    def get_observation(self) -> Observation:
        return Observation(
            cameras={name: self._camera_observation(name) for name in self.metadata.camera_names},
            robot_state=self.get_robot_state(),
            task_context=self.get_task_context(),
            timestamp_s=self.current_time_s,
        )

    # ---------------------------------------------------------------- base motion

    def get_base_pose(self) -> Pose:
        """The mobile base in the world frame.

        Exposed to generated programs as ``robocasa.get_base_pose``. It is a proprioceptive
        reading a real mobile robot also has, not simulator state.
        """
        last = self._require_client().last
        return matrix_to_pose(
            _transform_from_quaternion_and_position(last["base_quat_wxyz"], last["base_pos"]),
            frame="world",
        )

    def navigate_to_pose(self, dx: float, dy: float, dyaw: float) -> ExecutionResult:
        """Relative base motion in the current base frame, closed-loop.

        There is no collision checking, by design: a blocked path simply fails to converge and
        reports ``ok=False``. Forward motion from the spawn pose is usually blocked by a
        counter.
        """
        client = self._require_client()
        before = np.asarray(client.base_pos(), dtype=np.float64)[:2].copy()
        start_steps = client.sim_step_count()
        ok = bool(client.drive_base(float(dx), float(dy), float(dyaw)))
        moved = float(np.linalg.norm(np.asarray(client.base_pos(), dtype=np.float64)[:2] - before))
        steps = max(0, client.sim_step_count() - start_steps)
        self._elapsed_steps += steps
        return ExecutionResult(
            ok=ok,
            steps_executed=steps,
            final_observation=self.get_observation() if ok else None,
            error=None
            if ok
            else ApiError(
                code=ErrorCode.EXECUTION_FAILED,
                message=(
                    "base did not converge; the path is probably blocked "
                    f"(moved {moved:.3f} m of the requested motion)"
                ),
            ),
            final_errors={"base_translation_m": moved},
        )

    def _base_yaw(self) -> float:
        """World yaw of ``robot0_link0`` right now. Reads the cached state; costs no ticks."""
        return _yaw_from_wxyz(self.get_base_pose().quaternion_wxyz)

    def base_pose_is_free(self, x: float, y: float, yaw: float | None = None) -> bool:
        """Whether a PandaOmron footprint at world ``(x, y)`` is empty of scene geometry.

        ``yaw`` defaults to the yaw the base has **now**, not to zero. The footprint is a
        rectangle offset 0.20 m behind ``robot0_link0``, both rotated by ``yaw``, and this
        robot spawns near 90 degrees -- so a zero default tested a box turned across the
        counter and reported the robot's own pose blocked in every scene. A relative
        ``navigate_to_pose`` keeps the yaw it started with, so the current yaw is the right
        one for every pose a program reached by driving laterally. Pass ``yaw`` explicitly to
        ask about a turned pose.

        Same frame as :meth:`get_base_pose`. Arm poses and default SAM clouds are
        ``robot0_link0``; transform those into world before calling this. The
        answer is a boolean, not a map.
        """
        angle = self._base_yaw() if yaw is None else float(yaw)
        return bool(self._require_client().base_pose_is_free(float(x), float(y), angle))

    @staticmethod
    def plan_standoff_pose(
        support_cloud: Any, object_cloud: Any, standoff_m: float = 0.3
    ) -> tuple[float, float, float]:
        """Pure geometry: ``(x, y, yaw)`` outside the support hull, in the clouds' frame."""
        return standoff_from_clouds(support_cloud, object_cloud, standoff_m)

    def plan_approach_pose(
        self, object_cloud: Any, distance_m: float = 0.7
    ) -> tuple[float, float, float]:
        """Pure geometry: ``distance_m`` short of the object along the current line of sight.

        The answer is in the cloud's frame. ``world`` clouds use :meth:`get_base_pose`;
        ``robot0_link0`` clouds use the origin of that frame as the current base.
        """
        frame = getattr(object_cloud, "frame", None)
        if frame == self.BASE_FRAME:
            base = (0.0, 0.0, 0.0)
        else:
            pose = self.get_base_pose()
            base = (
                float(pose.position[0]),
                float(pose.position[1]),
                _yaw_from_wxyz(pose.quaternion_wxyz),
            )
        return approach_from_cloud(object_cloud, base, distance_m)

    # ---------------------------------------------------------------------- step

    def _truncated(self) -> bool:
        return self._elapsed_steps >= self.horizon

    def _reward(self) -> float | None:
        """Host-side scoring. Stripped by ``StepResult.public_view()``."""
        try:
            return float(self._require_client().evaluate()["reward"])
        except Exception:
            return None

    def step(self, action: RobotAction) -> StepResult:
        client = self._require_client()
        if self._codec is None:
            raise RuntimeError("reset() must be called before stepping")
        if self._run_observer is not None:
            self._run_observer.before_step(action)
        try:
            command = self._codec.encode(action, self.get_robot_state())
        except ValueError as exc:
            result = StepResult(
                ok=False,
                error=ApiError(code=ErrorCode.INVALID_REQUEST, message=str(exc)),
            )
            if self._run_observer is not None:
                self._run_observer.after_step(action, result)
            return result

        if command.gripper_position is not None:
            client.set_gripper_target(command.gripper_position)
        client.step(client.build_action(command.joint_target))
        self._elapsed_steps += 1
        self._last_reward = self._reward()
        truncated = self._truncated()
        result = StepResult(
            ok=True,
            observation=self.get_observation(),
            terminated=False,
            truncated=truncated,
            reward=self._last_reward,
            diagnostics={"elapsed_steps": self._elapsed_steps},
        )
        if self._run_observer is not None:
            self._run_observer.after_step(action, result)
        return result

    def _arm_action(self, joints: np.ndarray, gripper: float | None) -> RobotAction:
        return RobotAction(
            arms={
                self.arms[0]: ArmCommand(
                    mode="joint_position",
                    target=np.asarray(joints, dtype=np.float64).reshape(-1)[:7],
                    gripper_position=gripper,
                    embodiment=self.embodiment,
                )
            }
        )

    def execute_trajectory(
        self,
        trajectory: Trajectory | SynchronizedTrajectory,
        *,
        tolerance: float = 0.02,
        settle_steps: int = TRAJECTORY_GOAL_MAX_STEPS,
    ) -> ExecutionResult:
        # ``EnvironmentAdapter`` (protocols.py) declares both trajectory types, and
        # ``CapApi.move_synchronized`` hands the ``SynchronizedTrajectory`` it just planned
        # straight to the adapter. Accepting only ``Trajectory`` here made that whole tool
        # structurally impossible on this embodiment -- its planning half
        # (``plan_synchronized_motion``) succeeds, then execution raised
        # "trajectory must be a Trajectory", surfaced as INVALID_REQUEST. PandaOmron has one
        # arm, so a synchronized trajectory naming exactly that arm *is* a single-arm
        # trajectory; robosuite/adapter.py and behavior/adapter.py take the same branch.
        if not isinstance(trajectory, Trajectory | SynchronizedTrajectory):
            # Preserve the invalid-data ValueError contract.
            raise ValueError("trajectory must be a Trajectory or SynchronizedTrajectory")
        if isinstance(trajectory, SynchronizedTrajectory):
            if set(trajectory.joint_positions) != set(self.arms):
                raise ValueError(
                    f"synchronized trajectory must name exactly the arms {list(self.arms)}"
                )
            arm = self.arms[0]
            joint_positions = trajectory.joint_positions[arm]
            gripper_positions = (
                None if trajectory.gripper_positions is None else trajectory.gripper_positions[arm]
            )
        else:
            arm = trajectory.arm
            joint_positions = trajectory.joint_positions
            gripper_positions = trajectory.gripper_positions
        if arm not in self.arms:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(code=ErrorCode.INVALID_REQUEST, message=f"unknown arm {arm!r}"),
            )
        # Each tick must go through ``step()`` so the run recorder gets RGB.
        # ``client.step`` / ``move_to_joints`` move the arm but leave the
        # video at the reset frame (one 0.05 s mp4).
        # Densify from the start state, not from waypoint 0, so a hand-built trajectory whose
        # first waypoint is far away is ramped up to rather than crossed in one tick. This is
        # safe only behind CapApi's stale-plan check, which rejects any ``expected_start``
        # more than 0.02 rad from the measured joints before execution reaches here. Planner
        # paths start at the measured state, so their tick count is unchanged.
        start = np.asarray(trajectory.expected_start.joint_positions[arm], dtype=np.float64)
        planned = np.asarray(joint_positions, dtype=np.float64)
        waypoints = densify_joint_waypoints(np.vstack([start.reshape(1, -1), planned]))[1:]
        raw_grips = (
            None
            if gripper_positions is None
            else np.asarray(gripper_positions, dtype=np.float64).reshape(-1)
        )
        grips = None
        if raw_grips is not None and raw_grips.size and waypoints.size:
            grips = np.full(waypoints.shape[0], float(raw_grips[-1]))
        executed = 0
        final_observation: Observation | None = None
        for index, waypoint in enumerate(waypoints):
            grip = None if grips is None else float(grips[min(index, grips.size - 1)])
            result = self.step(self._arm_action(waypoint, grip))
            executed += 1
            if result.observation is not None:
                final_observation = result.observation
            if not result.ok:
                return ExecutionResult(
                    ok=False,
                    steps_executed=executed,
                    final_observation=final_observation,
                    error=result.error,
                    truncated=result.truncated,
                    terminated=result.terminated,
                )
            if result.terminated or result.truncated:
                return ExecutionResult(
                    ok=False,
                    steps_executed=executed,
                    final_observation=final_observation,
                    truncated=result.truncated,
                    terminated=result.terminated,
                    error=ApiError(
                        code=ErrorCode.EXECUTION_FAILED,
                        message=(
                            f"trajectory truncated at the {self.horizon}-tick horizon"
                            if result.truncated
                            else "episode ended before the trajectory finished"
                        ),
                    ),
                )
        goal = waypoints[-1][:7]
        last_grip = None if grips is None else float(grips[-1])
        error = float("inf")
        for _ in range(max(1, int(settle_steps))):
            state = (
                final_observation.robot_state
                if final_observation is not None
                else self.get_robot_state()
            )
            measured = np.asarray(state.joint_positions[arm], dtype=np.float64).reshape(-1)[:7]
            error = float(np.linalg.norm(measured - goal))
            if error <= float(tolerance):
                break
            result = self.step(self._arm_action(goal, last_grip))
            executed += 1
            if result.observation is not None:
                final_observation = result.observation
            if not result.ok or result.terminated or result.truncated:
                truncated = bool(result.truncated)
                return ExecutionResult(
                    ok=False,
                    steps_executed=executed,
                    final_observation=final_observation,
                    truncated=truncated,
                    terminated=result.terminated,
                    error=result.error
                    or ApiError(
                        code=ErrorCode.EXECUTION_FAILED,
                        message=(
                            f"trajectory truncated at the {self.horizon}-tick horizon"
                            if truncated
                            else "episode ended before the trajectory finished"
                        ),
                    ),
                    final_errors={"joint_position": error},
                )
        truncated = self._truncated()
        converged = error <= float(tolerance)
        ok = converged and not truncated
        return ExecutionResult(
            ok=ok,
            steps_executed=executed,
            final_observation=final_observation if ok else None,
            truncated=truncated,
            error=None
            if ok
            else ApiError(
                code=ErrorCode.EXECUTION_FAILED,
                message=(
                    f"trajectory truncated at the {self.horizon}-tick horizon"
                    if truncated
                    else f"final joint error {error:.4f} exceeds tolerance {tolerance}"
                ),
            ),
            final_errors={"joint_position": error},
        )

    # ------------------------------------------------------------------- gripper

    def _run_gripper(self, position: float, arm: str) -> ExecutionResult:
        if arm not in self.arms:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(code=ErrorCode.INVALID_REQUEST, message=f"unknown arm {arm!r}"),
            )
        # Same as execute_trajectory: the driver's own settle loop never
        # notifies the run recorder, so gripper motion would be invisible.
        settle = 30
        client = self._client
        if client is not None:
            settle = int(getattr(getattr(client, "profile", None), "gripper_settle_steps", settle))
        joints = self.get_robot_state().joint_positions[arm]
        final_observation: Observation | None = None
        executed = 0
        for _ in range(max(1, settle)):
            result = self.step(self._arm_action(joints, float(position)))
            executed += 1
            if result.observation is not None:
                final_observation = result.observation
            if not result.ok or result.terminated or result.truncated:
                return ExecutionResult(
                    ok=False,
                    steps_executed=executed,
                    final_observation=final_observation,
                    truncated=result.truncated,
                    terminated=result.terminated,
                    error=result.error,
                )
        measured = self._measured_gripper()
        return ExecutionResult(
            ok=True,
            steps_executed=executed,
            final_observation=final_observation or self.get_observation(),
            final_errors={"gripper_position": abs(measured - float(position))},
        )

    def set_gripper(self, position: float, *, arm: str = "primary") -> ExecutionResult:
        return self._run_gripper(position, arm)

    def close_gripper(self, *, arm: str = "primary") -> ExecutionResult:
        return self._run_gripper(0.0, arm)

    def set_grippers(self, positions: Mapping[str, float]) -> ExecutionResult:
        if set(positions) != set(self.arms):
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message=f"expected exactly the arms {list(self.arms)}",
                ),
            )
        arm = self.arms[0]
        return self._run_gripper(positions[arm], arm)

    # ------------------------------------------------------- host-side scoring

    def check_success(self) -> bool:
        """Task success, for the host only.

        ``run.py`` scores the episode with this. It is not registered as a tool, and the
        simulator's ``evaluate`` channel is unreachable from generated code by construction:
        a program holds no client, only the ``CapApi`` surface.
        """
        return bool(self._require_client().evaluate()["success"])

    def compute_reward(self) -> float | None:
        return self._last_reward

    # ----------------------------------------------------- metadata extensions

    def get_task_metadata(self) -> Mapping[str, object]:
        return MappingProxyType(self.metadata.to_manifest_record())

    def _published_joint_limits(self, spec: Mapping[str, Any]) -> Mapping[str, object]:
        """The simulator's arm joint limits, checked against the measured configuration."""
        measured: np.ndarray | None = None
        try:
            measured = np.asarray(
                self._require_client().last["joint_pos"], dtype=np.float64
            ).reshape(-1)
        except (ApiError, KeyError, TypeError, ValueError):
            # Metadata is legal to ask for before the first observation lands; the structural
            # checks still run, only the containment check is skipped.
            measured = None
        return MappingProxyType(
            validate_joint_limits(
                spec.get("joint_limits"),
                arm_dof=int(spec.get("arm_dof", len(PANDA_JOINT_NAMES))),
                measured=measured,
            )
        )

    def get_controller_metadata(self) -> Mapping[str, object]:
        spec = self._require_client().spec
        return MappingProxyType(
            {
                "action_dimension": int(spec["action_dim"]),
                "arm_names": self.arms,
                "base_frame": self.BASE_FRAME,
                "camera_names": self.metadata.camera_names,
                "control_frequency_hz": float(spec.get("control_freq", self.control_frequency)),
                "control_period_s": self.control_period_s,
                "controller": self.controller,
                "gripper_semantics": {
                    "normalized": "0=closed, 1=open",
                    "native": "1=closed, -1=open",
                },
                "controllable_gripper_arms": self.arms,
                "supported_action_modes": ("joint_position",),
                # A mobile manipulator: the shared tool set cannot move the base, so the
                # program has to use the robocasa.* extensions to get within reach.
                "mobile_base": True,
                "joint_limits": self._published_joint_limits(spec),
            }
        )

    # --------------------------------------------------------------------- close

    def close(self) -> None:
        """Disconnect from the simulator without shutting it down.

        ``cap-harness run`` closes the adapter at the end of every episode, and a campaign is
        hundreds of episodes against one simulator process. Sending the protocol's ``CLOSE``
        here would terminate that process, so episode 2 of a 125-seed sweep would find nothing
        listening. Tearing the simulator down is the supervisor's job, not an episode's.
        """
        if self._client is not None and not self._closed:
            try:
                self._client.close(shutdown_server=False)
            finally:
                self._closed = True
                self._client = None


__all__ = [
    "TRAJECTORY_GOAL_MAX_STEPS",
    "TRAJECTORY_MAX_JOINT_STEP_RAD",
    "RobocasaAdapter",
    "densify_joint_waypoints",
]
