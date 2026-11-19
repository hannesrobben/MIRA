"""Tests of the stage-to-image geometry (MI-11) on synthetic mappings."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mira.geometry import (
    StageToImage,
    fit_stage_to_image,
    load_calibration,
    save_calibration,
)
from mira.products import Grid
from mira.utils.exceptions import ValidationError

SHAPE = (512, 512)
META = {"objective": "Zeiss 63x", "device": "dhm", "pose_tolerance_um": 1.0}


def _model(**kwargs):
    params = {
        "pixel_um": (0.0869, 0.0871),
        "rotation_rad": math.radians(0.8),
        "shear": 0.004,
        "offset_um": (3.2, -1.7),
    }
    params.update(kwargs)
    return StageToImage.from_parameters(**params, **META, calibration_id="cal-test")


def test_round_trip_pixel_sample():
    model = _model(flip_x=True)
    rng = np.random.default_rng(1)
    pixels = rng.uniform(0, 511, size=(50, 2))
    stage = {"y": 840.0, "x": 1200.0, "z": 30.0}
    points = model.to_sample(pixels, stage, SHAPE)
    back = model.to_pixel(points, stage, SHAPE)
    np.testing.assert_allclose(back, pixels, atol=1e-9)


def test_centre_pixel_maps_to_stage_plus_offset():
    model = _model()
    centre = model.to_sample([255.5, 255.5], (100.0, 200.0), SHAPE)
    np.testing.assert_allclose(centre, [103.2, 198.3], atol=1e-12)


def test_parameters_are_recovered():
    model = _model(rotation_rad=0.01, shear=0.0)
    assert model.rotation_rad == pytest.approx(0.01, abs=1e-12)
    assert model.pixel_um == pytest.approx((0.0869, 0.0871), rel=1e-12)


def test_nominal_fallback_is_axis_aligned_grid():
    model = StageToImage.nominal(
        (0.0869, 0.0869), objective="Zeiss 63x", device="dhm", pose_tolerance_um=5.0
    )
    assert not model.calibrated
    assert model.is_axis_aligned
    grid = model.grid({"y": 10.0, "x": 20.0}, (4, 6))
    assert grid == Grid.centred({"y": 10.0, "x": 20.0}, (0.0869, 0.0869), (4, 6))
    with pytest.raises(ValidationError):
        model.to_dict()


def test_rotated_model_cannot_be_a_grid():
    with pytest.raises(ValidationError, match="axis-aligned"):
        _model().grid((0.0, 0.0), SHAPE)


def test_xy_matrix_conversion_matches_nanofactorysystem_transform():
    # nanofactorysystem Transform: object_xy = stage_xy + P2D @ px_xy + offset_xy
    p2d = np.array([[0.276, 0.002], [-0.001, 0.275]])
    offset_xy = np.array([1.5, -2.0])
    model = StageToImage.from_xy_matrix(
        p2d, offset_xy, objective="Zeiss 20x", device="camera", pose_tolerance_um=2.0
    )
    px_xy = np.array([12.0, -30.0])  # relative to the image centre
    stage_xy = np.array([500.0, 700.0])
    expected_xy = stage_xy + p2d @ px_xy + offset_xy
    pixel_rc = np.array([px_xy[1], px_xy[0]]) + np.array([255.5, 255.5])
    got_yx = model.to_sample(pixel_rc, stage_xy[::-1], SHAPE)
    np.testing.assert_allclose(got_yx[::-1], expected_xy, atol=1e-12)


@pytest.mark.parametrize("known_points", [False, True])
def test_affine_fit_recovers_model(known_points):
    """3 x 3 raster of stage offsets, mark positions with 0.05 px noise."""
    truth = _model()
    rng = np.random.default_rng(7)
    mark = np.array([850.0, 1205.0])
    stages, pixels = [], []
    for dy in (-15.0, 0.0, 15.0):
        for dx in (-15.0, 0.0, 15.0):
            stage = mark + np.array([dy, dx]) - truth.offset_um
            stages.append(stage)
            pixels.append(truth.to_pixel(mark, stage, SHAPE) + rng.normal(0, 0.05, 2))
    points = np.repeat(mark[None, :], 9, axis=0) if known_points else None
    fit = fit_stage_to_image(stages, pixels, SHAPE, points_um=points)
    # 0.05 px noise over a 15 µm / 0.087 µm ≈ 170 px baseline: J to < 1e-3 rel.
    np.testing.assert_allclose(
        fit.jacobian_um_per_px, truth.jacobian_um_per_px, rtol=0, atol=2e-4
    )
    assert fit.rms_um < 0.02
    if known_points:
        np.testing.assert_allclose(fit.offset_um, truth.offset_um, atol=0.02)
        model = fit.model(
            objective="Zeiss 63x",
            device="dhm",
            calibration_id="c1",
            pose_tolerance_um=0.5,
        )
        assert model.residuals_um["n_points"] == 9
    else:
        assert fit.offset_um is None
        np.testing.assert_allclose(
            fit.point_minus_offset_um, mark - truth.offset_um, atol=0.02
        )
        with pytest.raises(ValidationError):
            fit.model(
                objective="Zeiss 63x",
                device="dhm",
                calibration_id="c1",
                pose_tolerance_um=0.5,
            )


def test_fit_rejects_degenerate_input():
    with pytest.raises(ValidationError, match="at least 3"):
        fit_stage_to_image([[0, 0], [1, 1]], [[0, 0], [1, 1]], SHAPE)
    with pytest.raises(ValidationError, match="collinear"):
        fit_stage_to_image([[0, 0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]], SHAPE)


def test_calibration_file_round_trip_and_fallback(tmp_path):
    model = _model()
    path = save_calibration(model, tmp_path / "cal.json")
    loaded = load_calibration(
        path,
        objective="Zeiss 63x",
        device="dhm",
        nominal_pixel_um=(0.0869, 0.0869),
        fallback_tolerance_um=5.0,
    )
    assert loaded.calibrated and loaded.calibration_id == "cal-test"
    np.testing.assert_allclose(loaded.jacobian_um_per_px, model.jacobian_um_per_px)
    np.testing.assert_allclose(loaded.offset_um, model.offset_um)
    with pytest.raises(ValidationError, match="not 'Zeiss 20x'"):
        load_calibration(
            path,
            objective="Zeiss 20x",
            device="dhm",
            nominal_pixel_um=(0.276, 0.276),
            fallback_tolerance_um=5.0,
        )
    fallback = load_calibration(
        tmp_path / "missing.json",
        objective="Zeiss 63x",
        device="dhm",
        nominal_pixel_um=(0.0869, 0.0869),
        fallback_tolerance_um=5.0,
    )
    assert not fallback.calibrated and fallback.pose_tolerance_um == 5.0


def test_validation():
    with pytest.raises(ValidationError, match="device"):
        StageToImage.nominal(
            (0.1, 0.1), objective="o", device="sem", pose_tolerance_um=1.0
        )
    with pytest.raises(ValidationError, match="singular"):
        StageToImage(jacobian_um_per_px=[[1, 1], [1, 1]], offset_um=(0, 0), **META)
    with pytest.raises(ValidationError, match="message_type"):
        StageToImage.from_dict({"message_type": "Other"})
