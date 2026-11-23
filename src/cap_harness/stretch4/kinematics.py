"""FK, IK and the wrist-camera pose for the Stretch 4, with mink over MuJoCo.

The same approach as :mod:`cap_harness.yam_real.kinematics`, on the generated
model in ``description/stretch4.xml``. That model's qpos is exactly the eight
ASPIRE joints in order: ``base_x, base_y, base_theta`` (planar pose in odom),
``lift, arm, wrist_yaw, wrist_pitch, wrist_roll``. Poses are in odom.

Two additions over YAM's solver, both because this arm is redundant (the base
adds three joints): ``mink.ConfigurationLimit`` keeps lift, arm and wrists in
range, and a light posture task toward the seed keeps the solution near the
current configuration instead of drifting the base.

Quaternions at this module's boundary are **wxyz**, the harness convention.
Not thread safe: one mutable ``mink.Configuration`` per instance.
"""

from __future__ import annotations

from pathlib import Path

import mink
import mujoco
import numpy as np

from cap_harness.contracts import IKResult, Pose, RobotState
from cap_harness.errors import ApiError, ErrorCode

MODEL_PATH = Path(__file__).resolve().parent / "description" / "stretch4.xml"
JOINT_NAMES = (
    "base_x",
    "base_y",
    "base_theta",
    "lift",
    "arm",
    "wrist_yaw",
    "wrist_pitch",
    "wrist_roll",
)
DOF = len(JOINT_NAMES)
GRASP_SITE = "grasp_site"
WRIST_CAMERA_SITE = "wrist_camera_optical"
#: FK of a returned solution must land this close to the target to count as
#: solved; matches the YAM provider and CapApi.move_to_pose's tolerance.
IK_TOLERANCE_M = 0.01
#: Posture cost relative to the end-effector task: small enough not to pull
#: the gripper off target, large enough to stop the base wandering.
POSTURE_COST = 1e-2


class Stretch4Kinematics:
    """FK/IK over the Stretch 4 kinematic model."""

    def __init__(self, model_path: str | Path = MODEL_PATH, *, max_iters: int = 200) -> None:
        path = Path(model_path)
        self.model_path = path
        self.model = mujoco.MjModel.from_xml_path(str(path))
        names = tuple(self.model.joint(index).name for index in range(self.model.njnt))
        if names != JOINT_NAMES or self.model.nq != DOF:
            raise ValueError(f"{path} joints are {names}, expected {JOINT_NAMES}")
        self.configuration = mink.Configuration(self.model)
        self.ee_task = mink.FrameTask(
            frame_name=GRASP_SITE,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        self.posture_task = mink.PostureTask(self.model, cost=POSTURE_COST)
        self.limits = [mink.ConfigurationLimit(self.model)]
        self.max_iters = int(max_iters)
        self.lower = self.model.jnt_range[:, 0].copy()
        self.upper = self.model.jnt_range[:, 1].copy()
        limited = self.model.jnt_limited.astype(bool)
        self.lower[~limited] = -np.inf
        self.upper[~limited] = np.inf

    def _set(self, joints: np.ndarray) -> None:
        self.configuration.update(np.asarray(joints, dtype=np.float64).reshape(DOF))

    def _site_pose(self, site: str) -> tuple[np.ndarray, np.ndarray]:
        transform = self.configuration.get_transform_frame_to_world(site, "site")
        return transform.translation().copy(), transform.rotation().wxyz.copy()

    def forward_kinematics(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Grasp-site ``(position, quaternion_wxyz)`` in odom."""
        self._set(joints)
        return self._site_pose(GRASP_SITE)

    def camera_from_joints(self, joints: np.ndarray) -> np.ndarray:
        """4x4 odom-from-wrist-camera-optical transform."""
        self._set(joints)
        return self.configuration.get_transform_frame_to_world(
            WRIST_CAMERA_SITE, "site"
        ).as_matrix()

    def inverse_kinematics(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        seed: np.ndarray,
        *,
        dt: float = 0.05,
        solver: str = "daqp",
        damping: float = 1e-3,
        err_threshold: float = 1e-4,
    ) -> np.ndarray:
        """Best configuration toward the target, starting from ``seed``.

        Convergence is not guaranteed; check the result with
        :meth:`forward_kinematics` (the provider below does).
        """
        seed = np.asarray(seed, dtype=np.float64).reshape(DOF)
        self._set(np.clip(seed, self.lower, self.upper))
        self.posture_task.set_target(self.configuration.q.copy())
        self.ee_task.set_target(
            mink.SE3.from_rotation_and_translation(
                mink.SO3(wxyz=np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)),
                np.asarray(position, dtype=np.float64).reshape(3),
            )
        )
        tasks = [self.ee_task, self.posture_task]
        for _ in range(self.max_iters):
            velocity = mink.solve_ik(
                self.configuration, tasks, dt, solver, damping, limits=self.limits
            )
            self.configuration.integrate_inplace(velocity, dt)
            if np.linalg.norm(self.ee_task.compute_error(self.configuration)) <= err_threshold:
                break
        return self.configuration.q.copy()


class Stretch4IKProvider:
    """``IKProvider`` for the single Stretch arm, seeded from measured state."""

    def __init__(self, kinematics: Stretch4Kinematics, *, tolerance_m: float = IK_TOLERANCE_M):
        self._kin = kinematics
        self._tolerance_m = float(tolerance_m)

    def solve_ik(
        self, target_pose: Pose, robot_state: RobotState, *, arm: str = "primary"
    ) -> IKResult:
        def failure(code: ErrorCode, message: str, **details: object) -> IKResult:
            return IKResult(
                ok=False,
                embodiment=robot_state.embodiment,
                error=ApiError(code=code, message=message, details=details),
            )

        if target_pose.frame != robot_state.base_frame:
            return failure(
                ErrorCode.INVALID_REQUEST,
                f"target frame {target_pose.frame!r} is not {robot_state.base_frame!r}",
            )
        try:
            seed = np.asarray(robot_state.joint_positions[arm], dtype=float).reshape(DOF)
        except (KeyError, TypeError, ValueError) as exc:
            return failure(
                ErrorCode.INVALID_REQUEST, f"robot state has no 8-joint {arm!r} arm: {exc}"
            )
        try:
            joints = self._kin.inverse_kinematics(
                target_pose.position, target_pose.quaternion_wxyz, seed
            )
            achieved, _ = self._kin.forward_kinematics(joints)
        except Exception as exc:
            return failure(
                ErrorCode.IK_FAILED, f"mink IK failed: {exc}", exception_type=type(exc).__name__
            )
        error_m = float(np.linalg.norm(achieved - np.asarray(target_pose.position, dtype=float)))
        if (
            not np.all(np.isfinite(joints))
            or not np.isfinite(error_m)
            or error_m > self._tolerance_m
        ):
            return failure(
                ErrorCode.IK_UNREACHABLE,
                f"mink did not converge: best configuration is {error_m:.3f} m from the target "
                f"(tolerance {self._tolerance_m:.3f} m)",
                position_error_m=error_m,
                arm=arm,
            )
        return IKResult(
            ok=True,
            embodiment=robot_state.embodiment,
            joint_positions=joints,
            arm=arm,
            diagnostics={"solver": "mink", "position_error_m": error_m},
        )


__all__ = [
    "DOF",
    "IK_TOLERANCE_M",
    "JOINT_NAMES",
    "MODEL_PATH",
    "Stretch4IKProvider",
    "Stretch4Kinematics",
]
