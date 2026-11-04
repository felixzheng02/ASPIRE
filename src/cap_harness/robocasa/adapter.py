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
from types import MappingProxyType
from typing import Any

import numpy as np

from cap_harness.contracts import (
    CameraObservation,
    ExecutionResult,
    Observation,
    Pose,
    RobotAction,
    RobotState,
    StepResult,
    TaskContext,
    Trajectory,
)
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.geometry import invert_transform, matrix_to_pose, quaternion_wxyz_to_matrix
from cap_harness.robocasa.codec import RobocasaActionCodec
from cap_harness.robocasa.registry import (
    RobocasaTaskMetadata,
    RobocasaTaskRegistry,
    scene_group_for_seed,
)


def _transform_from_quaternion_and_position(quaternion_wxyz: Any, position: Any) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = quaternion_wxyz_to_matrix(quaternion_wxyz)
    transform[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return transform


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

        return RobotClient(
            self._endpoint or wire.DEFAULT_ENDPOINT, timeout_ms=self._timeout_ms
        )

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
        self._codec = RobocasaActionCodec(
            metadata.arms, control_frequency=self.control_frequency
        )
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

    def execute_trajectory(
        self,
        trajectory: Trajectory,
        *,
        tolerance: float = 0.02,
        settle_steps: int = 0,
    ) -> ExecutionResult:
        if not isinstance(trajectory, Trajectory):
            raise ValueError("trajectory must be a Trajectory")
        client = self._require_client()
        arm = trajectory.arm
        if arm not in self.arms:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST, message=f"unknown arm {arm!r}"
                ),
            )
        waypoints = np.asarray(trajectory.joint_positions, dtype=np.float64)
        grips = (
            None
            if trajectory.gripper_positions is None
            else np.asarray(trajectory.gripper_positions, dtype=np.float64).reshape(-1)
        )
        executed = 0
        for index, waypoint in enumerate(waypoints):
            if grips is not None and index < grips.size:
                client.set_gripper_target(float(grips[index]))
            client.step(client.build_action(waypoint[:7]))
            executed += 1
            self._elapsed_steps += 1
            if self._truncated():
                break
        for _ in range(max(0, int(settle_steps))):
            client.step(client.build_action())
            executed += 1
            self._elapsed_steps += 1
        final = np.asarray(client.joint_pos(), dtype=np.float64).reshape(-1)[:7]
        error = float(np.linalg.norm(final - waypoints[-1][:7])) if waypoints.size else 0.0
        converged = error <= float(tolerance)
        truncated = self._truncated()
        self._last_reward = self._reward()
        ok = converged and not truncated
        return ExecutionResult(
            ok=ok,
            steps_executed=executed,
            final_observation=self.get_observation() if ok else None,
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
        client = self._require_client()
        start = client.sim_step_count()
        # The driver's settling loop, not a single tick: a grip that is commanded but not
        # settled reports success while the fingers are still moving.
        client.set_gripper(float(position))
        steps = max(0, client.sim_step_count() - start)
        self._elapsed_steps += steps
        measured = self._measured_gripper()
        self._last_reward = self._reward()
        return ExecutionResult(
            ok=True,
            steps_executed=steps,
            final_observation=self.get_observation(),
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
                "joint_limits": spec.get("joint_limits"),
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


__all__ = ["RobocasaAdapter"]
