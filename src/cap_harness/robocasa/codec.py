"""Portable ``RobotAction`` to RoboCasa PandaOmron driver commands.

Unlike :mod:`cap_harness.robosuite.codec`, this codec does **not** build the native action
vector. The part order of a RoboCasa action is not stable -- RoboCasa orders the composite
controller's parts ``right, right_gripper, base, torso`` while robosuite's own ``Lift`` orders
them ``right, torso, base, right_gripper``, and the base-mode flag sits at its own index -- so
the layout is read from the simulator's ``get_spec`` and assembled by
``robot_driver.RobotClient.build_action``. Reimplementing that here would be a second copy of
the one thing most likely to be silently wrong.

It also does **not** convert targets to velocities. ``pandaomron_joint_ctrl.json`` configures the
arm as ``JOINT_POSITION`` with ``"input_type": "absolute"``, so the controller consumes absolute
joint targets; the ``(target - current) * control_frequency`` form the Robosuite codec needs
would make the arm drift instead of track. (The *base* is ``JOINT_VELOCITY``, which is why base
motion is a closed loop in the driver rather than a command computed here.)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cap_harness.contracts import RobotAction, RobotState


@dataclass(frozen=True, slots=True)
class DriverCommand:
    """One control period, in the terms ``RobotClient.build_action`` accepts."""

    joint_target: np.ndarray
    """Absolute arm joint positions, radians."""

    gripper_position: float | None
    """Normalized grip in [0, 1], 1.0 fully open -- or None to hold the current grip.

    ASPIRE and the driver already agree here: ``open_gripper`` is ``set_gripper(1.0)`` on both
    sides and both map to native via ``1 - 2p``, so no conversion is applied.
    """


class RobocasaActionCodec:
    """Translate one synchronized tick into absolute driver commands."""

    def __init__(
        self,
        arms: Sequence[str] = ("primary",),
        *,
        control_frequency: float = 20.0,
        arm_dimension: int = 7,
    ) -> None:
        self.arms = tuple(arms)
        if self.arms != ("primary",):
            raise ValueError("RoboCasa's PandaOmron is single-armed; arms must be ('primary',)")
        self.control_frequency = float(control_frequency)
        if not np.isfinite(self.control_frequency) or self.control_frequency <= 0:
            raise ValueError("control_frequency must be positive and finite")
        self.arm_dimension = int(arm_dimension)
        if self.arm_dimension != 7:
            raise ValueError("PandaOmron's arm has 7 joints")

    def encode(self, action: RobotAction, current_state: RobotState) -> DriverCommand:
        if not isinstance(action, RobotAction) or not isinstance(current_state, RobotState):
            raise ValueError("action and current_state must be typed contract values")
        unknown = set(action.arms) - set(self.arms)
        if unknown:
            raise ValueError(f"unknown RoboCasa arms: {sorted(unknown)}")

        arm = self.arms[0]
        current = np.asarray(current_state.joint_positions[arm], dtype=np.float64)
        command = action.arms.get(arm)
        if command is None:
            # No command for the arm this tick: hold position rather than drift.
            return DriverCommand(joint_target=current.copy(), gripper_position=None)

        target = np.asarray(command.target, dtype=np.float64).reshape(-1)
        if target.shape != (self.arm_dimension,):
            raise ValueError(f"arm target must have shape ({self.arm_dimension},)")
        if not np.all(np.isfinite(target)):
            raise ValueError("arm target must be finite")
        return DriverCommand(joint_target=target, gripper_position=command.gripper_position)


__all__ = ["DriverCommand", "RobocasaActionCodec"]
