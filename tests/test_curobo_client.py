"""cuRobo client behaviour that needs no FastAPI, so it runs in every harness venv.

``tests/test_providers.py`` skips as a whole where the service module cannot import
(no FastAPI in ``.venv-robocasa``), which is why these live in their own file.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from cap_harness.contracts import PlanningScene, PointCloud, Pose, RobotPlanningContext, RobotState
from cap_harness.errors import ErrorCode
from cap_harness.providers.curobo import CuRoboProvider
from cap_harness.providers.http import HttpProviderClient, ProviderHttpError


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
        assert isinstance(outcome, FakeResponse)
        return outcome


def _state() -> RobotState:
    return RobotState(
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        end_effector_poses=Pose(
            position=np.zeros(3),
            quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            frame="robot_base",
        ),
        gripper_positions=0.5,
        base_frame="robot_base",
    )


def _context(state: RobotState) -> RobotPlanningContext:
    return RobotPlanningContext(
        embodiment=state.embodiment,
        model="panda",
        joint_names=state.joint_names,
        base_transforms={arm: np.eye(4) for arm in state.arms},
        end_effector_links=dict.fromkeys(state.arms, "panda_hand"),
    )


def test_curobo_refusal_reason_survives_a_long_service_detail() -> None:
    """The service's reason must reach the program even when its 422 body is long.

    The transport keeps only the first 512 characters of an error body as text. A
    cuRobo refusal puts per-constraint diagnostics after its message. Measured on
    RoboCasa PickPlaceCounterToStove seed 101 (standoff -> grasp move): the body was
    605 characters, the cut text was not valid JSON, and the program read
    "cuRobo refused the request" instead of the reason below.
    """
    zero = {"maximum": 0.0, "sum": 0.0}
    hit = {"maximum": 0.8770264387130737, "sum": 0.8770264387130737}
    refusal = {
        "detail": {
            "message": "cuRobo goal state infeasible (scene_collision)",
            "constraints_start": {
                "constraint:cspace": zero,
                "constraint:self_collision": zero,
                "constraint:scene_collision": zero,
                "sum_constraint": 0.0,
            },
            "start_state_feasible": True,
            "constraints_goal": {
                "constraint:cspace": zero,
                "constraint:self_collision": zero,
                "constraint:scene_collision": hit,
                "sum_constraint": 0.8770264387130737,
            },
            "goal_state_feasible": False,
            "failure_kind": "goal_scene_collision",
        }
    }
    body = json.dumps(refusal)
    assert len(body) > 512  # longer than the transport's text preview, as measured
    state = _state()
    provider = CuRoboProvider(session=FakeSession([FakeResponse(422, refusal, body)]))  # type: ignore[arg-type]
    scene = PlanningScene(PointCloud(points=np.array([[0.5, 0.0, 0.0]]), frame="robot_base"))

    plan = provider.plan_to_joints(state, np.zeros(7), scene=scene, context=_context(state))

    assert not plan.ok and plan.error is not None
    assert plan.error.code is ErrorCode.PLANNING_FAILED
    assert plan.error.message == "cuRobo goal state infeasible (scene_collision)"
    assert plan.error.details["status_code"] == 422


def test_http_error_keeps_the_decoded_body_next_to_the_text_preview() -> None:
    long_detail = {"detail": {"message": "why", "padding": "x" * 1000}}
    text = json.dumps(long_detail)
    client = HttpProviderClient(
        "http://service.test",
        max_retries=0,
        session=FakeSession(  # type: ignore[arg-type]
            [FakeResponse(422, long_detail, text), FakeResponse(400, ValueError("no json"), "oops")]
        ),
    )

    errors = []
    for _ in range(2):
        try:
            client.post_json("/plan", {})
        except ProviderHttpError as exc:
            errors.append(exc)

    long_error, plain_error = errors
    assert long_error.error.details["response"] == text[:512]  # preview unchanged
    assert long_error.response_body == long_detail
    assert plain_error.error.details["response"] == "oops"
    assert plain_error.response_body is None
