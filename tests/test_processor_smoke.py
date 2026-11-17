"""Smoke tests of the processor paths that import OpenCV lazily (MI-05).

The fields are synthetic: a raised rectangle (height 2 rad) on a flat
background with weak Gaussian noise, pixel size 1 µm.
"""

import logging

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from mira import StructureProcessor
from mira.processing.registration import ECCRefinement, PhaseCorrelationRefinement

cv2 = pytest.importorskip("cv2")


def _rectangle_field(shape=(120, 140), seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    field = np.zeros(shape, dtype=float)
    field[30:90, 40:110] = 2.0
    return field + rng.normal(0.0, 0.01, shape)


def _textured_field(seed=1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    field = gaussian_filter(rng.normal(size=(128, 128)), 4).astype(np.float32)
    field[40:90, 30:100] += 1.0
    return field


@pytest.fixture
def processor() -> StructureProcessor:
    proc = StructureProcessor(
        phase_data=_rectangle_field(), pixel_size=(1e-6, 1e-6), log_level=logging.ERROR
    )
    return proc.detect_structure()


def test_detection_finds_the_rectangle(processor):
    corners = processor.structure.corner_points
    np.testing.assert_array_equal(corners, [[30, 40], [30, 109], [89, 109], [89, 40]])
    # Perimeter uses cv2.findContours / arcLength (lazy import).
    assert processor.structure.perimeter == pytest.approx(256e-6, rel=0.02)


def test_apply_affine_transform_moves_structure_and_segments(processor):
    processor.segment_structure(num_segments=(2, 2))
    center_before = np.array(processor.structure.center)

    processor.apply_affine_transform(np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]]))

    # Translation by dx = +5 px, dy = -3 px; centers are (row, col).
    np.testing.assert_allclose(processor.structure.center, center_before + [-3, 5])
    assert len(processor.structure.segments) == 4


def test_phase_correlation_refinement_maps_moving_onto_fixed():
    fixed = _textured_field()
    moving = np.roll(fixed, (2, -3), axis=(0, 1))

    matrix = PhaseCorrelationRefinement().estimate(moving, fixed)
    warped = cv2.warpAffine(moving, matrix, fixed.shape[::-1])

    np.testing.assert_allclose(matrix[:, 2], [3.0, -2.0], atol=0.1)
    assert np.sqrt(np.mean((warped - fixed)[10:-10, 10:-10] ** 2)) < 1e-3


@pytest.mark.xfail(
    reason=(
        "ECCRefinement returns the inverse warp of cv2.findTransformECC "
        "(fixed -> moving), but RegistrationRefiner promises a matrix for "
        "cv2.warpAffine(moving, M); the residual doubles instead of vanishing. "
        "Fix waits for the maintainer (MI-79, may change published numbers)."
    ),
    strict=True,
)
def test_ecc_refinement_maps_moving_onto_fixed():
    fixed = _textured_field()
    moving = np.roll(fixed, (2, -3), axis=(0, 1))

    matrix = ECCRefinement(motion="translation").estimate(moving, fixed)
    warped = cv2.warpAffine(moving, matrix, fixed.shape[::-1])

    np.testing.assert_allclose(matrix[:, 2], [3.0, -2.0], atol=0.1)
    assert np.sqrt(np.mean((warped - fixed)[10:-10, 10:-10] ** 2)) < 0.05
