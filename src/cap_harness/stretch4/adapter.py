"""``EnvironmentAdapter`` for the Stretch 4, modelled on ``YamRealAdapter``.

One arm, ``primary``, with eight joints: base_x, base_y, base_theta (odom),
lift, arm, wrist_yaw, wrist_pitch, wrist_roll. Gripper is 0-1, 1 = open.

**Motion interlock.** As on YAM: reads always work; every command path is
refused before any bridge call unless ``allow_physical_motion`` is set.

**Trajectories.** The robot executor takes one relative move at a time and
waits for it to finish, so it cannot stream ASPIRE's interpolated waypoints.
``execute_trajectory`` sends only the final waypoint as one absolute target;
the executor then moves each joint on its own profile, so the path between
start and goal is not the straight joint-space line the planner produced.
There is no collision checking on this path.
"""

from __future__ import annotations

from collections.abc import Mapping
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
from cap_harness.yam_real.codec import rotation_to_wxyz

from .kinematics import DOF, JOINT_NAMES
from .station import BASE_FRAME, WRIST_CAMERA, Stretch4Station

EMBODIMENT = "stretch4"
ARM = "primary"
ARMS = (ARM,)
#: The base has no joint limit; this finite bound only feeds the controller
#: metadata the interpolation planner reads (metres / radians).
BASE_BOUND = 100.0
#: Same as the executor's completion tolerances (0.5 cm, 2 deg).
LINEAR_TOLERANCE_M = 0.005
ANGULAR_TOLERANCE_RAD = np.radians(2.0)
_ANGULAR = np.array(
    [name in {"base_theta", "wrist_yaw", "wrist_pitch", "wrist_roll"} for name in JOINT_NAMES]
)
HOME_GRIPPER_OPEN = 0.95
#: A trajectory's gripper column is only sent when it differs by this much
#: from the last gripper command, so a held squeeze is not re-commanded.
GRIPPER_CHANGE = 0.02


class Stretch4Adapter:
    embodiment = EMBODIMENT
    registry_prefix = EMBODIMENT

    def __init__(
        self,
        station: Stretch4Station,
        *,
        task_context: TaskContext | None = None,
        allow_physical_motion: bool = False,
    ) -> None:
        self._env = station
        self._task_context = task_context
        self._allow_physical_motion = bool(allow_physical_motion)
        self._command_rpc_count = 0
        self._last_gripper_command: float | None = None
        self._closed = False
        self.control_frequency = station.config.controller.control_frequency_hz
        self.control_period_s = 1.0 / self.control_frequency
        kinematics = station.kinematics
        self._lower = np.where(np.isfinite(kinematics.lower), kinematics.lower, -BASE_BOUND)
        self._upper = np.where(np.isfinite(kinematics.upper), kinematics.upper, BASE_BOUND)

    # -- identity ----------------------------------------------------------

    @property
    def native_env(self) -> Stretch4Station:
        return self._env

    @property
    def arms(self) -> tuple[str, ...]:
        return ARMS

    def resolve_arm(self, arm: str) -> str:
        return str(arm)

    @property
    def dof(self) -> int:
        return DOF

    @property
    def command_rpc_count(self) -> int:
        return self._command_rpc_count

    @property
    def physical_motion_authorized(self) -> bool:
        return self._allow_physical_motion

    def _motion_denied(self) -> ApiError | None:
        if self._allow_physical_motion:
            return None
        return ApiError(
            code=ErrorCode.SAFETY_INTERLOCK,
            message="physical motion is not authorized for this Stretch 4 adapter; start with --allow-motion",
            recoverable=False,
            details={"command_rpc_count": self._command_rpc_count},
        )

    # -- lifecycle ---------------------------------------------------------

    def reset(self, task_ref: object = None, seed: int = 0) -> Observation:
        """Set the task context and observe. Never moves; homing is ``go_home``."""
        if isinstance(task_ref, TaskContext):
            self._task_context = task_ref
        return self.get_observation()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._env.close()

    # -- observation -------------------------------------------------------

    def get_task_context(self) -> TaskContext:
        if self._task_context is None:
            raise RuntimeError("no task context set; pass one to the adapter or to reset()")
        return self._task_context

    def get_robot_state(self) -> RobotState:
        joints, gripper, _ = self._env.joint_state()
        position, quaternion = self._env.kinematics.forward_kinematics(joints)
        return RobotState(
            joint_positions={ARM: joints},
            joint_velocities={ARM: np.zeros(DOF)},
            end_effector_poses={
                ARM: Pose(position=position, quaternion_wxyz=quaternion, frame=BASE_FRAME)
            },
            gripper_positions={ARM: gripper},
            base_frame=BASE_FRAME,
            joint_names={ARM: JOINT_NAMES},
            embodiment=EMBODIMENT,
        )

    def get_observation(self) -> Observation:
        frame = self._env.wrist_frame()
        pose = frame.odom_from_camera
        camera = CameraObservation(
            rgb=frame.rgb,
            depth_m=frame.depth_m,
            intrinsics=frame.intrinsics,
            frame=f"{WRIST_CAMERA}_optical",
            camera_pose=Pose(
                position=pose[:3, 3],
                quaternion_wxyz=rotation_to_wxyz(pose[:3, :3]),
                frame=BASE_FRAME,
            ),
            timestamp_s=frame.timestamp_s,
        )
        return Observation(
            cameras={WRIST_CAMERA: camera},
            robot_state=self.get_robot_state(),
            task_context=self._task_context,
        )

    def check_success(self) -> bool:
        """No task success predicate exists on hardware."""
        return False

    # -- actuation ---------------------------------------------------------

    def _result(
        self, reply: Mapping[str, Any], operation: str, **diagnostics: Any
    ) -> ExecutionResult:
        status = str(reply.get("status", ""))
        error = None
        if status != "completed":
            error = ApiError(
                code=ErrorCode.INVALID_REQUEST
                if status == "rejected"
                else ErrorCode.EXECUTION_FAILED,
                message=f"{operation} {status or 'failed'} on the Stretch: {reply.get('reason') or 'no reason given'}",
                details={"status": status, "reason": reply.get("reason", "")},
            )
        return ExecutionResult(
            ok=error is None,
            steps_executed=1,
            final_observation=self.get_observation(),
            error=error,
            diagnostics={
                "bridge": dict(reply),
                "command_rpc_count": self._command_rpc_count,
                **diagnostics,
            },
        )

    def _failure(self, exc: Exception, operation: str) -> ExecutionResult:
        return ExecutionResult(
            ok=False,
            steps_executed=0,
            error=ApiError(
                code=ErrorCode.EXECUTION_FAILED,
                message=f"{operation} failed: {exc}",
                details={"command_rpc_count": self._command_rpc_count},
            ),
        )

    def _move(self, target: np.ndarray, gripper: float | None, operation: str) -> ExecutionResult:
        """One absolute joint target, then the gripper if it changes."""
        target = np.asarray(target, dtype=np.float64).reshape(DOF)
        try:
            reply = self._env.client.move(dict(zip(JOINT_NAMES, target.tolist())))
            self._command_rpc_count += 1
        except Exception as exc:
            return self._failure(exc, operation)
        measured, measured_gripper, _ = self._env.joint_state()
        if reply.get("status") == "completed" and gripper is not None:
            previous = self._last_gripper_command
            if abs((measured_gripper if previous is None else previous) - gripper) > GRIPPER_CHANGE:
                return self.set_gripper(gripper)
        residual = np.abs(measured - target)
        residual[_ANGULAR] = np.abs((residual[_ANGULAR] + np.pi) % (2 * np.pi) - np.pi)
        return self._result(
            reply,
            operation,
            joint_residual={name: float(value) for name, value in zip(JOINT_NAMES, residual)},
        )

    def step(self, action: RobotAction) -> StepResult:
        denied = self._motion_denied()
        if denied is not None:
            return StepResult(ok=False, error=denied)
        unknown = set(action.arms) - set(ARMS)
        if unknown:
            return StepResult(
                ok=False,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message=f"action names unknown Stretch arms: {sorted(unknown)}",
                    details={"known_arms": list(ARMS)},
                ),
            )
        command = action.arms[ARM]
        if not isinstance(command, ArmCommand):
            raise ValueError("arms['primary'] must be an ArmCommand")
        result = self._move(
            np.asarray(command.target, dtype=np.float64).reshape(-1)[:DOF],
            None if command.gripper_position is None else float(command.gripper_position),
            "step",
        )
        return StepResult(
            ok=result.ok,
            error=result.error,
            observation=result.final_observation,
            diagnostics=dict(result.diagnostics),
        )

    def execute_trajectory(
        self, trajectory: Trajectory | SynchronizedTrajectory
    ) -> ExecutionResult:
        """Send the trajectory's final waypoint as one absolute target (see module doc)."""
        denied = self._motion_denied()
        if denied is not None:
            return ExecutionResult(ok=False, steps_executed=0, error=denied)
        if isinstance(trajectory, Trajectory):
            joints = np.asarray(trajectory.joint_positions, dtype=np.float64)
            grippers = trajectory.gripper_positions
        else:
            joints = np.asarray(trajectory.joint_positions[ARM], dtype=np.float64)
            grippers = (
                None
                if trajectory.gripper_positions is None
                else trajectory.gripper_positions.get(ARM)
            )
        gripper = (
            None
            if grippers is None
            else float(np.asarray(grippers, dtype=np.float64).reshape(-1)[-1])
        )
        return self._move(joints[-1], gripper, "execute_trajectory")

    def set_gripper(self, position: float, *, arm: str = ARM) -> ExecutionResult:
        denied = self._motion_denied()
        if denied is not None:
            return ExecutionResult(ok=False, steps_executed=0, error=denied)
        if arm != ARM:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST, message=f"unknown Stretch arm {arm!r}"
                ),
            )
        try:
            reply = self._env.client.gripper(float(position))
            self._command_rpc_count += 1
        except Exception as exc:
            return self._failure(exc, "set_gripper")
        if reply.get("status") == "completed":
            self._last_gripper_command = float(position)
        return self._result(reply, "set_gripper")

    def set_grippers(self, positions: Mapping[str, float]) -> ExecutionResult:
        unknown = set(positions) - set(ARMS)
        if unknown:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message=f"unknown Stretch arms {sorted(unknown)}",
                ),
            )
        if ARM not in positions:
            return ExecutionResult(
                ok=True, steps_executed=0, final_observation=self.get_observation()
            )
        return self.set_gripper(float(positions[ARM]))

    def go_home(self, *, arm: str = "both", duration: float | None = None) -> ExecutionResult:
        """The bridge's Home: ``goto_pose kitchen_manipulation`` for lift, arm and wrists. Never moves the base.

        ``arm`` and ``duration`` are accepted for API compatibility and ignored;
        the executor sets its own speeds.
        """
        denied = self._motion_denied()
        if denied is not None:
            return ExecutionResult(ok=False, steps_executed=0, error=denied)
        try:
            reply = self._env.client.home()
            self._command_rpc_count += 1
        except Exception as exc:
            return self._failure(exc, "go_home")
        result = self._result(reply, "go_home")
        state = result.final_observation.robot_state
        joints = dict(zip(JOINT_NAMES, state.joint_positions[ARM]))
        target = reply.get("target") or {}
        residuals = {name: abs(joints[name] - float(value)) for name, value in target.items()}
        verified = bool(target) and all(
            value <= (ANGULAR_TOLERANCE_RAD if name.startswith("wrist") else LINEAR_TOLERANCE_M)
            for name, value in residuals.items()
        )
        return ExecutionResult(
            ok=result.ok,
            steps_executed=result.steps_executed,
            final_observation=result.final_observation,
            error=result.error,
            diagnostics={
                **result.diagnostics,
                "joint_residual": residuals,
                "home_verified": result.ok and verified,
                "grippers_open": {ARM: float(state.gripper_positions[ARM]) >= HOME_GRIPPER_OPEN},
                "requested_arm": str(arm),
            },
        )

    # -- planning and metadata --------------------------------------------

    def get_planning_context(self) -> RobotPlanningContext:
        return RobotPlanningContext(
            embodiment=EMBODIMENT,
            model=EMBODIMENT,
            joint_names={ARM: JOINT_NAMES},
            base_transforms={ARM: np.eye(4)},
            end_effector_links={ARM: "grasp_center_link"},
        )

    def get_task_metadata(self) -> dict[str, Any]:
        context = self._task_context
        return {
            "embodiment": EMBODIMENT,
            "station": self._env.config.station,
            "arms": list(ARMS),
            "dof": DOF,
            "joint_names": list(JOINT_NAMES),
            "cameras": [WRIST_CAMERA],
            "view_only_cameras": [
                name for name in self._env.camera_aliases if name != WRIST_CAMERA
            ],
            "language": "" if context is None else context.language,
        }

    def get_controller_metadata(self) -> dict[str, Any]:
        return {
            "control_frequency_hz": self.control_frequency,
            "control_period_s": self.control_period_s,
            "arm_names": list(ARMS),
            "joint_names": list(JOINT_NAMES),
            "action_lower_bounds": self._lower.tolist(),
            "action_upper_bounds": self._upper.tolist(),
            "trajectory_execution": "final waypoint only, one executor move; no collision checking",
            "physical_motion_authorized": self._allow_physical_motion,
            "command_rpc_count": self._command_rpc_count,
        }


__all__ = ["ARM", "EMBODIMENT", "Stretch4Adapter"]
