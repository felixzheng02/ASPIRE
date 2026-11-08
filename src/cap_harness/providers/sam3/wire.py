"""SAM3 HTTP paths and scoring rules shared by the lightweight client and model service.

This module imports nothing: the service venv carries torch and sam3, the harness venv
carries neither, and both need to agree on what a score means and how results are
ranked. Keeping the rule here rather than writing it twice is what stops the two sides
drifting silently.
"""

SEGMENT_PATH = "/segment"
SEGMENT_POINT_PATH = "/segment_point"

#: Most masks either side will return for one prompt.
MAX_SEGMENTATIONS = 5

#: SAM3's own detection threshold, from ``Sam3Processor.__init__``.
#:
#: SAM3 scores a candidate as ``sigmoid(pred_logits) * sigmoid(presence_logit_dec)`` and
#: keeps the ones above this value. That product is a *concept-presence* confidence --
#: the second factor is the model's answer to "is this concept in this image at all" --
#: so a crisp, well-formed mask of the wrong object scores low, not high. It is not a
#: mask-quality or predicted-IoU score.
SAM3_DEFAULT_CONFIDENCE_THRESHOLD = 0.5

#: What the cap-harness service runs at, which is deliberately *not* SAM3's default.
#:
#: At 0.0 every candidate survives, so ``/segment`` always returns ranked masks and
#: never an empty ``results`` list -- for "sink" in a kitchen with no sink exactly as
#: much as for "plate". That suits a caller that wants to rank candidates itself, and
#: misleads one that reads a non-empty ``results`` as a detection.
#:
#: Measured on RoboCasa ``PickPlaceCounterToStove`` (seeds 101/102/103, ``agentview``
#: and ``robot0_eye_in_hand``, 73 labelled prompts): thresholding the published score at
#: 0.5 classifies present-vs-absent at 94.5%, against 39.7% at 0.0. It is a filter, not
#: a presence test -- see ``docs/api-reference.md`` for the residual failures.
DEFAULT_CONFIDENCE_THRESHOLD = 0.0


def rank_and_filter(items, score_of, threshold, limit=MAX_SEGMENTATIONS):
    """Rank candidates by score, drop those below ``threshold``, cap at ``limit``.

    Returns ``(kept, best_score)``. ``best_score`` is the highest score seen *before*
    filtering, so a caller handed an empty ``kept`` can still say whether the model
    found nothing (``best_score`` near 0.0) or the threshold rejected what it found.
    """
    ranked = sorted(items, key=score_of, reverse=True)
    best_score = float(score_of(ranked[0])) if ranked else 0.0
    kept = [item for item in ranked if float(score_of(item)) >= threshold]
    return kept[:limit], best_score


__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "MAX_SEGMENTATIONS",
    "SAM3_DEFAULT_CONFIDENCE_THRESHOLD",
    "SEGMENT_PATH",
    "SEGMENT_POINT_PATH",
    "rank_and_filter",
]
