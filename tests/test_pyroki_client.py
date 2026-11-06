from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import requests

from cap_harness.contracts import Pose, RobotState
from cap_harness.errors import ErrorCode
from cap_harness.geometry import quaternion_wxyz_to_matrix
from cap_harness.providers.pyroki import PyRokiProvider
from cap_harness.providers.pyroki.client import _panda_hand_target


class FakeResponse:
    def __init__(self, status_code: int, payload: object, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> object:
        if isinstance(self._payload, BaseException):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, FakeResponse)
        return outcome


def _robocasa_state() -> RobotState:
    pose = Pose(
        position=np.array([0.4, 0.0, 0.45]),
        quaternion_wxyz=np.array([0.0, 1.0, 0.0, 0.0]),
        frame="robot0_link0",
    )
    return RobotState(
        joint_positions=np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
        joint_velocities=np.zeros(7),
        end_effector_poses=pose,
        gripper_positions=1.0,
        base_frame="robot0_link0",
        embodiment="robocasa",
    )


def _pyroki_healthz() -> bool:
    try:
        response = requests.get("http://127.0.0.1:8116/healthz", timeout=0.5)
    except requests.RequestException:
        return False
    return response.status_code == 200


def _ok_ik_session() -> FakeSession:
    return FakeSession(
        [
            FakeResponse(
                200,
                {
                    "joint_positions": [0.0] * 8,
                    "position_error_m": 0.001,
                    "orientation_error_rad": 0.01,
                },
            )
        ]
    )


def test_pyroki_allowlists_robocasa_and_reaches_http() -> None:
    session = _ok_ik_session()
    state = _robocasa_state()
    result = PyRokiProvider(session=session).solve_ik(  # type: ignore[arg-type]
        state.end_effector_poses["primary"], state
    )
    assert result.ok
    assert result.embodiment == "robocasa"
    assert session.calls[0]["url"].endswith("/ik")


def test_pyroki_maps_robocasa_eef_with_local_z_90() -> None:
    session = _ok_ik_session()
    state = _robocasa_state()
    pose = state.end_effector_poses["primary"]
    result = PyRokiProvider(session=session).solve_ik(pose, state)  # type: ignore[arg-type]
    assert result.ok
    sent = np.asarray(session.calls[0]["json"]["target_pose_wxyz_xyz"], dtype=np.float64)
    np.testing.assert_allclose(sent, _panda_hand_target(pose, "robocasa"))
    public_rotation = quaternion_wxyz_to_matrix(pose.quaternion_wxyz)
    sent_rotation = quaternion_wxyz_to_matrix(sent[:4])
    assert not np.allclose(sent_rotation, public_rotation)


def test_pyroki_maps_libero_eef_without_wrist_rotation() -> None:
    session = _ok_ik_session()
    pose = Pose(
        position=np.array([0.4, 0.0, 0.45]),
        quaternion_wxyz=np.array([0.0, 1.0, 0.0, 0.0]),
        frame="robot0_base",
    )
    state = RobotState(
        joint_positions=np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
        joint_velocities=np.zeros(7),
        end_effector_poses=pose,
        gripper_positions=1.0,
        base_frame="robot0_base",
        embodiment="libero",
    )
    result = PyRokiProvider(session=session).solve_ik(pose, state)  # type: ignore[arg-type]
    assert result.ok
    sent = np.asarray(session.calls[0]["json"]["target_pose_wxyz_xyz"], dtype=np.float64)
    np.testing.assert_allclose(sent, _panda_hand_target(pose, "libero"))
    np.testing.assert_allclose(
        quaternion_wxyz_to_matrix(sent[:4]),
        quaternion_wxyz_to_matrix(pose.quaternion_wxyz),
    )


def test_pyroki_rejects_embodiments_outside_the_panda_allowlist() -> None:
    state = RobotState(
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        end_effector_poses=Pose(
            position=np.zeros(3),
            quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            frame="robot_base",
        ),
        gripper_positions=0.5,
        base_frame="robot_base",
        embodiment="behavior",
    )
    result = PyRokiProvider(session=FakeSession([])).solve_ik(  # type: ignore[arg-type]
        state.end_effector_poses["primary"], state
    )
    assert not result.ok
    assert result.error is not None
    assert result.error.code is ErrorCode.UNSUPPORTED
    assert "behavior" in result.error.message


@pytest.mark.skipif(not _pyroki_healthz(), reason="PyRoki is not running on :8116")
def test_pyroki_live_solves_a_robocasa_reachable_pose() -> None:
    state = _robocasa_state()
    result = PyRokiProvider().solve_ik(state.end_effector_poses["primary"], state)
    assert result.ok, result.error
    assert result.joint_positions is not None
    assert result.joint_positions.shape == (7,)
