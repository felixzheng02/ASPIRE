"""What ``Segmentation.score`` means, and what ``ok=True`` does not promise.

The service runs SAM3 at a 0.0 detection threshold, so ``/segment`` answers every noun
with five ranked masks. These tests pin the two things that make that legible to a
caller: the score the model actually produced is reported even when nothing survives
filtering, and the default client keeps behaving exactly as it always has.

No live service: every response is a stub.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np
import pytest

from cap_harness.contracts import CameraObservation, Pose
from cap_harness.errors import ErrorCode
from cap_harness.providers.sam3.client import (
    SUGGESTED_PRESENCE_THRESHOLD,
    Sam3Provider,
)


class FakeResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = ""

    def json(self) -> object:
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
        return outcome  # type: ignore[return-value]


def _camera() -> CameraObservation:
    return CameraObservation(
        rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        depth_m=np.ones((2, 3), dtype=np.float64),
        intrinsics=np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]]),
        frame="camera/agentview",
        camera_pose=Pose(
            position=np.zeros(3),
            quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            frame="robot_base",
        ),
    )


def _mask_result(score: float, label: str = "sink") -> dict[str, Any]:
    mask = np.ones((2, 3), dtype=np.uint8)
    return {
        "mask_base64": base64.b64encode(mask.tobytes()).decode("ascii"),
        "shape": [2, 3],
        "box": [0.0, 0.0, 2.0, 2.0],
        "score": score,
        "label": label,
    }


def _payload(scores: list[float], **extra: object) -> dict[str, Any]:
    return {"results": [_mask_result(score) for score in scores], **extra}


# --------------------------------------------------------------------------- defaults


def test_default_provider_still_returns_masks_for_a_low_scoring_prompt() -> None:
    """The long-standing contract. ``connectivity_check.py`` and the tool-check catalog
    both read ``ok`` for scene nouns, including nouns that score badly on the wrist
    camera, so the default must not start rejecting them.
    """
    session = FakeSession([FakeResponse(200, _payload([0.0347], threshold=0.0, best_score=0.0347))])

    result = Sam3Provider(session=session).segment_text(_camera(), "pan")  # type: ignore[arg-type]

    assert result.ok
    assert len(result.segmentations) == 1
    assert result.segmentations[0].score == pytest.approx(0.0347)


def test_diagnostics_report_the_models_best_candidate_and_the_service_threshold() -> None:
    session = FakeSession(
        [FakeResponse(200, _payload([0.7232, 0.1494], threshold=0.0, best_score=0.7232))]
    )

    result = Sam3Provider(session=session).segment_text(_camera(), "sink")  # type: ignore[arg-type]

    assert result.ok
    assert result.diagnostics["top_score"] == pytest.approx(0.7232)
    assert result.diagnostics["best_candidate_score"] == pytest.approx(0.7232)
    assert result.diagnostics["service_threshold"] == pytest.approx(0.0)
    assert result.diagnostics["min_score"] == pytest.approx(0.0)
    assert "presence test" in str(result.diagnostics["score_meaning"])


def test_no_threshold_is_sent_unless_one_is_configured() -> None:
    session = FakeSession([FakeResponse(200, _payload([0.9], threshold=0.0, best_score=0.9))])

    Sam3Provider(session=session).segment_text(_camera(), "plate")  # type: ignore[arg-type]

    payload = session.calls[0]["json"]
    assert isinstance(payload, dict)
    assert set(payload) == {"image_base64", "text_prompt"}


# --------------------------------------------------------------------------- min_score


def test_min_score_turns_an_absent_prompt_into_a_typed_no_segmentation() -> None:
    """The measured RoboCasa case: "pan" on an agentview with no pan scores 0.035."""
    session = FakeSession([FakeResponse(200, _payload([0.0347], threshold=0.0, best_score=0.0347))])
    provider = Sam3Provider(session=session, min_score=SUGGESTED_PRESENCE_THRESHOLD)  # type: ignore[arg-type]

    result = provider.segment_text(_camera(), "pan")

    assert not result.ok
    assert result.error is not None
    assert result.error.code is ErrorCode.NO_SEGMENTATION
    # The rejected score is in the message, so a trace says *why* it was rejected.
    assert "0.035" in result.error.message
    assert result.error.details["best_candidate_score"] == pytest.approx(0.0347)


def test_min_score_keeps_a_prompt_the_model_is_confident_about() -> None:
    session = FakeSession(
        [FakeResponse(200, _payload([0.9444, 0.1074], threshold=0.0, best_score=0.9444))]
    )
    provider = Sam3Provider(session=session, min_score=SUGGESTED_PRESENCE_THRESHOLD)  # type: ignore[arg-type]

    result = provider.segment_text(_camera(), "fish")

    assert result.ok
    # The weak second candidate is dropped; the confident one survives.
    assert [round(item.score, 4) for item in result.segmentations] == [0.9444]


def test_min_score_does_not_rescue_a_semantically_close_absent_noun() -> None:
    """Honest about the limit. "sink" on a kitchen with no sink scores 0.72 and returns
    the plate, so no threshold that keeps the present counter (0.75) rejects it.
    """
    session = FakeSession([FakeResponse(200, _payload([0.7232], threshold=0.0, best_score=0.7232))])
    provider = Sam3Provider(session=session, min_score=SUGGESTED_PRESENCE_THRESHOLD)  # type: ignore[arg-type]

    result = provider.segment_text(_camera(), "sink")

    assert result.ok, "a 0.5 cut cannot separate 'sink' from a present 'counter'"


def test_min_score_outside_the_unit_interval_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="min_score"):
        Sam3Provider(min_score=1.5)


# --------------------------------------------------------------------- compatibility


def test_a_service_without_the_additive_fields_still_works() -> None:
    """``threshold`` and ``best_score`` are additive. An older service omits them."""
    session = FakeSession([FakeResponse(200, {"results": [_mask_result(0.82)]})])

    result = Sam3Provider(session=session).segment_text(_camera(), "plate")  # type: ignore[arg-type]

    assert result.ok
    assert "service_threshold" not in result.diagnostics
    # Recovered from the results rather than reported.
    assert result.diagnostics["best_candidate_score"] == pytest.approx(0.82)


def test_a_non_numeric_threshold_is_a_typed_perception_failure() -> None:
    session = FakeSession([FakeResponse(200, _payload([0.5], threshold="half"))])

    result = Sam3Provider(session=session).segment_text(_camera(), "plate")  # type: ignore[arg-type]

    assert not result.ok
    assert result.error is not None
    assert result.error.code is ErrorCode.PERCEPTION_FAILED


def test_an_empty_results_list_names_the_prompt_and_the_best_score() -> None:
    session = FakeSession(
        [FakeResponse(200, {"results": [], "threshold": 0.5, "best_score": 0.02})]
    )

    result = Sam3Provider(session=session).segment_text(_camera(), "bicycle")  # type: ignore[arg-type]

    assert not result.ok
    assert result.error is not None
    assert result.error.code is ErrorCode.NO_SEGMENTATION
    assert "bicycle" in result.error.message
    assert result.error.details["best_candidate_score"] == pytest.approx(0.02)
