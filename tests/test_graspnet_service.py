"""Contact-GraspNet service: the order in which the vendored model sees the cloud."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest

# The service imports torch and FastAPI, which only the GraspNet provider venv
# carries; elsewhere these tests skip rather than fail collection.
service = pytest.importorskip("cap_harness.providers.graspnet.service")


def _b64(array: np.ndarray) -> str:
    with io.BytesIO() as buffer:
        np.save(buffer, array, allow_pickle=False)
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def _raster_cloud(rows: int = 40, cols: int = 50) -> np.ndarray:
    """A depth image back-projected row by row, the order CapApi sends."""
    v, u = np.mgrid[0:rows, 0:cols]
    return np.column_stack(
        [(u.ravel() - cols / 2) * 0.0025, (v.ravel() - rows / 2) * 0.0025, np.full(u.size, 1.1)]
    ).astype(np.float32)


class _RecordingEstimator:
    def __init__(self) -> None:
        self.scene_clouds: list[np.ndarray] = []

    def predict_scene_grasps(self, pc_full, pc_segments, **_kwargs):
        self.scene_clouds.append(np.array(pc_full, copy=True))
        key = next(iter(pc_segments))
        return (
            {key: np.eye(4)[None]},
            {key: np.array([0.9])},
            {key: np.zeros((1, 3))},
            {key: np.array([0.04])},
        )


def _plan(monkeypatch: pytest.MonkeyPatch, full: np.ndarray) -> np.ndarray:
    estimator = _RecordingEstimator()
    monkeypatch.setattr(service, "_ESTIMATOR", estimator)
    service._plan(
        service.PlanPointCloudsRequest(
            pc_full_base64=_b64(full), pc_segment_base64=_b64(full[:200])
        )
    )
    (seen,) = estimator.scene_clouds
    return seen


def test_model_never_sees_the_callers_raster_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """The vendored PointNet++ ball query keeps the lowest indices in each ball.

    A raster-ordered crop below 20000 points reaches it unshuffled, every
    neighbourhood becomes a strip of image rows, and the width head collapses
    to its 4 mm bin. The service must hand over the same points, reordered.
    """
    full = _raster_cloud()
    seen = _plan(monkeypatch, full)
    assert seen.shape == full.shape
    assert not np.array_equal(seen, full)
    assert np.array_equal(np.unique(seen, axis=0), np.unique(full, axis=0))


def test_the_order_is_a_pure_function_of_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    full = _raster_cloud()
    assert np.array_equal(_plan(monkeypatch, full), _plan(monkeypatch, full))
