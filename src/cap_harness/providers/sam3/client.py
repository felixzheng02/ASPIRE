"""Typed SAM3 client for the provider service wire format."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
import struct
from typing import Any
import zlib

import numpy as np
import requests

from cap_harness.contracts import CameraObservation, Segmentation, SegmentationSet
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.providers.http import HttpProviderClient, ProviderHttpError
from cap_harness.providers.sam3.wire import (
    MAX_SEGMENTATIONS,
    SAM3_DEFAULT_CONFIDENCE_THRESHOLD,
    SEGMENT_PATH,
    SEGMENT_POINT_PATH,
    rank_and_filter,
)

DEFAULT_SAM3_URL = "http://127.0.0.1:8114"

#: The cut to compare ``Segmentation.score`` against when a caller wants a yes/no.
#:
#: ``score`` is a concept-presence confidence, not mask quality (see ``wire.py``), so it
#: does carry a present/absent signal -- but an imperfect one. Measured on RoboCasa
#: ``PickPlaceCounterToStove`` (seeds 101/102/103, ``agentview`` and
#: ``robot0_eye_in_hand``, 73 labelled prompts) a sweep put the accuracy peak at 0.55
#: (95.9%); 0.5 is SAM3's own default and scores 94.5%, within noise of the peak, so
#: this tracks SAM3 rather than a number fitted to one kitchen.
#:
#: It is a filter, not a presence test, and the residual errors are semantic rather than
#: numeric: "sink" scores 0.72 on a kitchen with no sink (it returns the plate, the same
#: box "plate" returns), while a zucchini that *is* present scores 0.17 when prompted as
#: "squash" and 0.87 as "green vegetable". Nothing here fixes either case.
SUGGESTED_PRESENCE_THRESHOLD = SAM3_DEFAULT_CONFIDENCE_THRESHOLD


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = binascii.crc32(kind)
    checksum = binascii.crc32(payload, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _encode_rgb_png(image: np.ndarray) -> str:
    """Encode an RGB array without adding Pillow as a runtime dependency."""
    if np.issubdtype(image.dtype, np.floating):
        rgb = np.rint(image * 255.0).astype(np.uint8)
    else:
        rgb = image.astype(np.uint8, copy=False)
    height, width, _ = rgb.shape
    scanlines = b"".join(b"\x00" + row.tobytes(order="C") for row in rgb)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanlines))
        + _png_chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode("ascii")


def _optional_float(value: object, name: str) -> float | None:
    """Read an additive numeric response field, tolerating a service that omits it."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        # Preserve the invalid-data ValueError contract.
        raise ValueError(f"response field {name!r} must be a number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"response field {name!r} must be finite")
    return result


def _decode_raw_array(value: object, *, dtype: np.dtype[Any], shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(value, str):
        # Preserve the invalid-data ValueError contract.
        raise ValueError("base64 array payload must be a string")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid base64 array payload") from exc
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(raw) != expected_bytes:
        raise ValueError(
            f"decoded array has {len(raw)} bytes, expected {expected_bytes} for shape {shape}"
        )
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


class Sam3Provider:
    """Typed client for SAM3 on port 8114.

    ``CameraObservation`` does not carry a separate dictionary key, so its stable
    camera frame is also used as the segmentation's camera identity.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_SAM3_URL,
        *,
        timeout_s: float | tuple[float, float] = 120.0,
        max_retries: int = 2,
        retry_backoff_s: float = 0.1,
        session: requests.Session | None = None,
        min_score: float = 0.0,
    ) -> None:
        #: Client-side floor on ``Segmentation.score``. Defaults to 0.0, which keeps the
        #: long-standing contract that a scene noun yields masks and ``ok=True``. Raising
        #: it turns ``segment_text`` into a detector: prompts whose best candidate falls
        #: below it come back ``ok=False`` with ``NO_SEGMENTATION``. See
        #: ``SUGGESTED_PRESENCE_THRESHOLD`` for the measured value, and note that callers
        #: such as ``examples/robocasa/connectivity_check.py`` currently assume 0.0.
        self._min_score = float(min_score)
        if not 0.0 <= self._min_score <= 1.0:
            raise ValueError("min_score must be in [0, 1]")
        self._http = HttpProviderClient(
            base_url,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            session=session,
        )

    @property
    def base_url(self) -> str:
        return self._http.base_url

    def health(self) -> bool:
        return self._http.health()

    def segment_text(
        self,
        observation: CameraObservation,
        query: str,
    ) -> SegmentationSet:
        """Segment ``query`` in one camera image.

        ``ok=True`` is **not** evidence that ``query`` is in the frame. The service runs
        SAM3 at a 0.0 detection threshold, so it answers every noun with its five
        best-ranked regions; prompt a kitchen with no sink for "sink" and it returns the
        plate, at score 0.72. Read ``Segmentation.score`` -- it is a concept-presence
        confidence, not a mask-quality score (see ``wire.py``) -- and compare it against
        ``SUGGESTED_PRESENCE_THRESHOLD`` rather than trusting ``ok``.

        ``diagnostics`` carries ``top_score``, ``best_candidate_score`` and
        ``service_threshold`` so a run's trace records the judgement call either way.
        """
        if not isinstance(observation, CameraObservation):
            return self._failure(ErrorCode.INVALID_REQUEST, "observation must be CameraObservation")
        if not isinstance(query, str) or not query.strip():
            return self._failure(ErrorCode.INVALID_REQUEST, "query must be a non-empty string")
        query = query.strip()
        try:
            response = self._http.post_json(
                SEGMENT_PATH,
                {"image_base64": _encode_rgb_png(observation.rgb), "text_prompt": query},
            )
            raw_results = response.get("results")
            if not isinstance(raw_results, list):
                # Preserve the invalid-data ValueError contract.
                raise ValueError("response field 'results' must be a list")
            # Both fields are additive; a service predating them reports nothing here and
            # best_candidate_score is recovered from the results below.
            service_threshold = _optional_float(response.get("threshold"), "threshold")
            best_candidate = _optional_float(response.get("best_score"), "best_score")

            segmentations: list[Segmentation] = []
            for item in raw_results:
                if not isinstance(item, Mapping):
                    # Preserve the invalid-data ValueError contract.
                    raise ValueError("each result must be an object")
                shape = self._shape(item.get("shape"), dimensions=2)
                if shape != observation.rgb.shape[:2]:
                    raise ValueError(
                        f"mask shape {shape} does not match image shape {observation.rgb.shape[:2]}"
                    )
                mask = _decode_raw_array(
                    item.get("mask_base64"), dtype=np.dtype(np.uint8), shape=shape
                ).astype(bool)
                label = item.get("label", query)
                if not isinstance(label, str) or not label.strip():
                    label = query
                segmentations.append(
                    Segmentation(
                        mask=mask,
                        label=label,
                        score=float(item.get("score")),
                        camera_name=observation.frame,
                        frame=observation.frame,
                        box_xyxy=np.asarray(item.get("box"), dtype=np.float64),
                    )
                )
        except ProviderHttpError as exc:
            return SegmentationSet(ok=False, error=exc.error)
        except (TypeError, ValueError, KeyError) as exc:
            return self._failure(
                ErrorCode.PERCEPTION_FAILED,
                "SAM3 returned an invalid text-segmentation response",
                details={"provider": "sam3", "error": str(exc)},
            )

        kept, observed_best = rank_and_filter(
            segmentations, lambda item: item.score, self._min_score
        )
        # The service reports the score it saw before its own filtering; without that
        # field the best we can know is the best survivor.
        if best_candidate is None:
            best_candidate = observed_best

        diagnostics: dict[str, object] = {
            "provider": "sam3",
            "query": query,
            # The model's top candidate before any filtering -- the number that says
            # whether SAM3 believed the concept was in frame.
            "best_candidate_score": best_candidate,
            "min_score": self._min_score,
            "score_meaning": (
                "sam3 concept-presence confidence; a filter, not a presence test "
                "(semantically close nouns score high on the wrong object)"
            ),
        }
        if service_threshold is not None:
            diagnostics["service_threshold"] = service_threshold

        if not kept:
            return self._failure(
                ErrorCode.NO_SEGMENTATION,
                f"SAM3 found no segmentation for {query!r} scoring at or above "
                f"{self._min_score:.2f} (best candidate scored {best_candidate:.3f})",
                details=diagnostics,
            )
        diagnostics["top_score"] = kept[0].score
        return SegmentationSet(ok=True, segmentations=kept, diagnostics=diagnostics)

    def segment_points(
        self,
        observation: CameraObservation,
        points_px: np.ndarray,
        *,
        point_labels: np.ndarray | None = None,
    ) -> SegmentationSet:
        if not isinstance(observation, CameraObservation):
            return self._failure(ErrorCode.INVALID_REQUEST, "observation must be CameraObservation")
        try:
            points = np.asarray(points_px, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            return self._failure(
                ErrorCode.INVALID_REQUEST,
                "points_px must be a finite Nx2 array",
                details={"error": str(exc)},
            )
        if points.shape != (1, 2) or not bool(np.all(np.isfinite(points))):
            return self._failure(
                ErrorCode.UNSUPPORTED,
                "the SAM3 service wire format supports exactly one finite point prompt",
            )
        if point_labels is not None:
            labels = np.asarray(point_labels)
            if labels.shape != (1,) or labels[0] not in (1, True):
                return self._failure(
                    ErrorCode.UNSUPPORTED,
                    "the SAM3 service wire format supports one positive point only",
                )
        x, y = (float(points[0, 0]), float(points[0, 1]))
        height, width = observation.rgb.shape[:2]
        if not (0.0 <= x < width and 0.0 <= y < height):
            return self._failure(ErrorCode.INVALID_REQUEST, "point prompt is outside the image")

        try:
            response = self._http.post_json(
                SEGMENT_POINT_PATH,
                {"image_base64": _encode_rgb_png(observation.rgb), "point_coords": [x, y]},
            )
            scores_raw = response.get("scores")
            if not isinstance(scores_raw, list):
                # Preserve the invalid-data ValueError contract.
                raise ValueError("response field 'scores' must be a list")
            scores = np.asarray(scores_raw, dtype=np.float64)
            if scores.ndim != 1 or not bool(np.all(np.isfinite(scores))):
                raise ValueError("scores must be a finite vector")
            shape = self._shape(response.get("masks_shape"), dimensions=3)
            if shape[0] != len(scores) or shape[1:] != observation.rgb.shape[:2]:
                raise ValueError("point-mask shape does not match scores and image")
            dtype_name = response.get("masks_dtype", "float32")
            if dtype_name not in {"bool", "uint8", "float32", "float64"}:
                raise ValueError(f"unsupported masks_dtype {dtype_name!r}")
            masks = _decode_raw_array(
                response.get("masks_base64"), dtype=np.dtype(dtype_name), shape=shape
            )
            segmentations = [
                Segmentation(
                    mask=mask > 0,
                    label=f"point({x:g}, {y:g})",
                    score=float(score),
                    camera_name=observation.frame,
                    frame=observation.frame,
                )
                for mask, score in zip(masks, scores, strict=False)
            ]
        except ProviderHttpError as exc:
            return SegmentationSet(ok=False, error=exc.error)
        except (TypeError, ValueError, KeyError) as exc:
            return self._failure(
                ErrorCode.PERCEPTION_FAILED,
                "SAM3 returned an invalid point-segmentation response",
                details={"provider": "sam3", "error": str(exc)},
            )

        if not segmentations:
            return self._failure(
                ErrorCode.NO_SEGMENTATION,
                "SAM3 found no segmentation for the point prompt",
                details={"provider": "sam3", "point_px": (x, y)},
            )
        segmentations.sort(key=lambda item: item.score, reverse=True)
        segmentations = segmentations[:MAX_SEGMENTATIONS]
        return SegmentationSet(
            ok=True,
            segmentations=segmentations,
            diagnostics={"provider": "sam3", "point_px": (x, y)},
        )

    @staticmethod
    def _shape(value: object, *, dimensions: int) -> tuple[int, ...]:
        if not isinstance(value, list | tuple) or len(value) != dimensions:
            raise ValueError(f"shape must contain {dimensions} dimensions")
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
            raise ValueError("shape dimensions must be non-negative integers")
        return tuple(value)

    @staticmethod
    def _failure(
        code: ErrorCode,
        message: str,
        *,
        details: Mapping[str, object] | None = None,
    ) -> SegmentationSet:
        return SegmentationSet(
            ok=False,
            error=ApiError(code=code, message=message, details=details or {}),
        )


SAM3Provider = Sam3Provider
Sam3Client = Sam3Provider

__all__ = [
    "DEFAULT_SAM3_URL",
    "MAX_SEGMENTATIONS",
    "SUGGESTED_PRESENCE_THRESHOLD",
    "SAM3Provider",
    "Sam3Client",
    "Sam3Provider",
]
