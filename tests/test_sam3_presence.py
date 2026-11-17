"""What ``Segmentation.score`` means, and which benchmarks act on it.

The service runs SAM3 at a 0.0 detection threshold, so ``/segment`` answers every noun
with five ranked masks. These tests pin three things: the score the model actually
produced is reported even when nothing survives filtering, the bare client keeps
behaving exactly as it always has, and the RoboCasa path alone turns that score into a
present/absent answer.

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
from cap_harness.run import (
    SAM3_MIN_SCORE_ENV,
    SEGMENTATION_MIN_SCORE,
    segmentation_min_score,
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
    """The bare provider stays permissive.

    The presence floor is applied per benchmark in ``run.py``, not here, so a caller that
    constructs ``Sam3Provider()`` itself -- the agent server, a scratch script -- keeps
    ranking candidates for itself and never sees a new ``ok=False``.
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


# ------------------------------------------------------------------ per-benchmark default


def test_robocasa_enforces_the_presence_floor_and_other_benchmarks_do_not() -> None:
    """The scope of this change, stated as a test.

    :8114 is shared with LIBERO, BEHAVIOR, Robosuite and CaP-X, whose prompts are long
    descriptive phrases nobody has scored. Only RoboCasa's bare scene nouns were measured,
    so only RoboCasa acts on the number.
    """
    assert segmentation_min_score("robocasa") == pytest.approx(SUGGESTED_PRESENCE_THRESHOLD)
    for benchmark in ("libero-pro", "behavior", "robosuite", "yam_real"):
        assert segmentation_min_score(benchmark) == pytest.approx(0.0)


def test_an_unknown_benchmark_keeps_the_permissive_default() -> None:
    assert segmentation_min_score("a-benchmark-that-does-not-exist-yet") == pytest.approx(0.0)


def test_the_floor_tracks_sam3s_own_threshold_rather_than_a_fitted_constant() -> None:
    """0.55 was this kitchen's sweep argmax, better by one prompt out of 73. The cut
    deliberately follows ``Sam3Processor``'s own default instead.
    """
    assert SEGMENTATION_MIN_SCORE["robocasa"] is SUGGESTED_PRESENCE_THRESHOLD


def test_the_environment_can_restore_the_old_permissive_behaviour(monkeypatch) -> None:
    monkeypatch.setenv(SAM3_MIN_SCORE_ENV, "0.0")

    assert segmentation_min_score("robocasa") == pytest.approx(0.0)


def test_the_environment_can_raise_the_floor_for_a_benchmark_without_one(monkeypatch) -> None:
    monkeypatch.setenv(SAM3_MIN_SCORE_ENV, "0.55")

    assert segmentation_min_score("libero-pro") == pytest.approx(0.55)


def test_a_blank_override_is_ignored_rather_than_read_as_zero(monkeypatch) -> None:
    """An exported-but-empty variable is a shell accident, not a request for 0.0."""
    monkeypatch.setenv(SAM3_MIN_SCORE_ENV, "   ")

    assert segmentation_min_score("robocasa") == pytest.approx(SUGGESTED_PRESENCE_THRESHOLD)


@pytest.mark.parametrize("value", ["1.5", "-0.1", "half", "nan", "inf"])
def test_an_unusable_override_names_the_variable_instead_of_silently_passing(
    monkeypatch, value: str
) -> None:
    monkeypatch.setenv(SAM3_MIN_SCORE_ENV, value)

    with pytest.raises(ValueError, match=SAM3_MIN_SCORE_ENV):
        segmentation_min_score("robocasa")


def test_the_robocasa_floor_rejects_the_measured_absent_pan() -> None:
    """End to end through the provider, at the default a RoboCasa run now gets.

    "pan" on seed 101 agentview scored 0.0347 and returned the *plate's* box. Before this
    change that was ``ok=True`` and a program reached for it.
    """
    session = FakeSession([FakeResponse(200, _payload([0.0347], threshold=0.0, best_score=0.0347))])
    provider = Sam3Provider(session=session, min_score=segmentation_min_score("robocasa"))  # type: ignore[arg-type]

    result = provider.segment_text(_camera(), "pan")

    assert not result.ok
    assert result.error is not None
    assert result.error.code is ErrorCode.NO_SEGMENTATION
    assert result.error.details["min_score"] == pytest.approx(0.5)


def test_the_libero_floor_still_returns_that_same_weak_mask() -> None:
    """The other side of the scope: an identical response, a different benchmark."""
    session = FakeSession([FakeResponse(200, _payload([0.0347], threshold=0.0, best_score=0.0347))])
    provider = Sam3Provider(session=session, min_score=segmentation_min_score("libero-pro"))  # type: ignore[arg-type]

    result = provider.segment_text(_camera(), "pan")

    assert result.ok
    assert len(result.segmentations) == 1
