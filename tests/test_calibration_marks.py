"""Tests of the calibration-mark detection and fit (MI-41) on synthetic marks.

The marks are rendered analytically through a known stage-to-image model
(smooth edges, white noise); tolerances are stated per test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mira.calibration_marks import detect_cross, detect_square, fit_cross_raster
from mira.geometry import StageToImage
from mira.utils.exceptions import ValidationError

SHAPE = (256, 256)
META = {"objective": "Zeiss 20x", "device": "camera", "pose_tolerance_um": 2.0}
TRUTH = StageToImage.from_parameters(
    pixel_um=(0.276, 0.279),
    rotation_rad=math.radians(1.2),
    shear=0.01,
    offset_um=(2.0, -3.0),
    **META,
    calibration_id="truth",
)
PRIOR = StageToImage.nominal(
    (0.276, 0.276),
    **{k: META[k] for k in ("objective", "device")},
    pose_tolerance_um=5.0,
)


def _soft(d, edge):
    return 0.5 * (1 - np.tanh(d / edge))


def _render(
    kind,
    mark_um,
    stage_um,
    *,
    size_um=20.0,
    arm_um=3.0,
    noise=0.02,
    seed=0,
    polarity=1.0,
):
    rows, cols = np.indices(SHAPE)
    pts = TRUTH.to_sample(np.stack([rows, cols], -1).astype(float), stage_um, SHAPE)
    dy = pts[..., 0] - mark_um[0]
    dx = pts[..., 1] - mark_um[1]
    edge = 0.15
    if kind == "square":
        w = _soft(np.abs(dy) - size_um / 2, edge) * _soft(
            np.abs(dx) - size_um / 2, edge
        )
    else:
        bar_y = _soft(np.abs(dx) - arm_um / 2, edge) * _soft(
            np.abs(dy) - size_um / 2, edge
        )
        bar_x = _soft(np.abs(dy) - arm_um / 2, edge) * _soft(
            np.abs(dx) - size_um / 2, edge
        )
        w = np.clip(bar_y + bar_x, 0, 1)
    rng = np.random.default_rng(seed)
    return 0.3 + polarity * 1.2 * w + noise * rng.standard_normal(SHAPE)


@pytest.mark.parametrize("polarity", [1.0, -1.0])
def test_cross_centre(polarity):
    stage = (100.0, 200.0)
    mark = (104.3, 196.1)
    det = detect_cross(_render("cross", mark, stage, polarity=polarity), smooth_px=1.0)
    expected = TRUTH.to_pixel(mark, stage, SHAPE)
    np.testing.assert_allclose(det.centre_rc, expected, atol=0.05)  # px
    assert det.polarity == int(polarity)


def test_square_gives_rotation_scale_shear():
    stage = (0.0, 0.0)
    mark = np.array([1.0, -2.0])
    fit = detect_square(_render("square", mark, stage, size_um=30.0), 30.0, PRIOR)
    # 30 um ~ 108 px side; corner accuracy ~0.1 px -> J within ~0.3 %.
    np.testing.assert_allclose(
        fit.jacobian_um_per_px, TRUTH.jacobian_um_per_px, atol=1.5e-3
    )
    assert fit.residual_px < 0.2
    np.testing.assert_allclose(
        fit.centre_rc, TRUTH.to_pixel(mark, stage, SHAPE), atol=0.1
    )
    model = fit.model(PRIOR, calibrated=True, calibration_id="c")
    assert model.rotation_rad == pytest.approx(math.radians(1.2), abs=math.radians(0.1))


def test_cross_raster_affine_fit():
    """3 x 3 stage raster of one cross: J independent of the galvo scale."""
    cross = (500.0, 800.0)
    stages, images = [], []
    for i, dy in enumerate((-12.0, 0.0, 12.0)):
        for j, dx in enumerate((-12.0, 0.0, 12.0)):
            stage = (
                cross[0] - TRUTH.offset_um[0] + dy,
                cross[1] - TRUTH.offset_um[1] + dx,
            )
            stages.append(stage)
            images.append(_render("cross", cross, stage, seed=3 * i + j))
    fit, dets = fit_cross_raster(images, stages, cross_um=cross)
    assert len(dets) == 9
    np.testing.assert_allclose(
        fit.jacobian_um_per_px, TRUTH.jacobian_um_per_px, atol=1e-3
    )
    np.testing.assert_allclose(fit.offset_um, TRUTH.offset_um, atol=0.03)
    assert fit.rms_um < 0.02
    unknown, _ = fit_cross_raster(images, stages)
    assert unknown.offset_um is None
    np.testing.assert_allclose(
        unknown.jacobian_um_per_px, fit.jacobian_um_per_px, atol=1e-9
    )


def test_validation():
    with pytest.raises(ValidationError):
        fit_cross_raster([np.zeros(SHAPE)], [(0, 0), (1, 1)])
    with pytest.raises(ValidationError):
        detect_cross(np.zeros((4, 4, 2)))
