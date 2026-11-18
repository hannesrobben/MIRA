"""Tests of the comparison product ↔ target (MI-52) on synthetic cases."""

import numpy as np
import pytest

from mira.deviation import DeviationReport, compare
from mira.forward import Optics, height_to_phase_rad
from mira.products import Grid, LayerProduct
from mira.targets import LayerTarget
from mira.utils.exceptions import ValidationError

OPTICS = Optics(wavelength_um=0.674994, n_structure=1.52, n_surround=1.508, phase_sign=1)
PGRID = Grid((0.0, 0.0), (0.2, 0.2), (60, 60))


def _target(height=2.0):
    # 0.1 µm raster, square of 4 x 4 µm in the middle of a 12 x 12 µm field.
    values = np.zeros((120, 120), np.float32)
    values[40:80, 40:80] = height
    return LayerTarget(values=values, mask=values > 0, grid=Grid((-0.05, -0.05), (0.1, 0.1),
                       (120, 120)), domain="height", target_id="tgt", n_structure=1.52,
                       n_surround=1.508)


def _product(height_map_um, piston=0.0, noise=0.0, valid=None, seed=0):
    phase = height_to_phase_rad(height_map_um, **OPTICS.as_kwargs()) + piston
    phase = phase + np.random.default_rng(seed).normal(0, noise, phase.shape)
    return LayerProduct(phase_rad=phase, intensity_rel=np.ones(PGRID.shape), grid=PGRID,
                        valid=np.ones(PGRID.shape, bool) if valid is None else valid,
                        tiles=(), z_um=0.0, propagation_index=1.508,
                        wavelength_um=0.674994, product_id="lp")


def _truth(height=2.0):
    h = np.zeros(PGRID.shape)
    h[20:40, 20:40] = height
    return h


def test_perfect_match_after_levelling():
    report = compare(_product(_truth(), piston=1.3), _target(), optics=OPTICS, sigma=0.05)
    assert report.domain == "phase"
    assert report.stats["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert report.stats["levelling"]["offset"] == pytest.approx(1.3, abs=1e-6)
    assert report.stats["levelling"]["reference"] == "substrate"
    assert report.stats["valid_fraction"] == pytest.approx(1.0)
    assert report.stats["classification"]["outside_fraction"] == 0.0
    assert report.stats["validity"]["coverage_ok"]


def test_too_low_structure_is_classified():
    # Printed only 1.5 µm instead of 2 µm: deviation = phase of -0.5 µm.
    report = compare(_product(_truth(1.5), noise=0.01), _target(2.0), optics=OPTICS,
                     sigma=0.01, k=3)
    expected = height_to_phase_rad(-0.5, **OPTICS.as_kwargs())
    assert report.stats["segments"]["structure"]["mean"] == pytest.approx(expected, abs=0.005)
    assert report.stats["segments"]["substrate"]["mean"] == pytest.approx(0.0, abs=0.005)
    # Deviation -0.056 rad vs. 3 sigma = 0.03 rad: about 0.5 % of the pixels fall inside.
    assert report.stats["classification"]["too_low_fraction"] == pytest.approx(1.0, abs=0.01)
    assert report.stats["classification"]["too_high_fraction"] == 0.0


def test_height_domain_in_micrometres():
    report = compare(_product(_truth(2.2)), _target(2.0), optics=OPTICS, domain="height")
    assert report.domain == "height"
    assert report.stats["segments"]["structure"]["mean"] == pytest.approx(0.2, abs=1e-5)
    header, buffers = report.to_message()
    assert buffers[0][0] == "deviation_um"
    assert "no noise level" in " ".join(report.warnings)


def test_invalid_pixels_and_coverage_flag():
    valid = np.ones(PGRID.shape, bool)
    valid[20:40, 20:32] = False  # 60 % of the structure missing
    report = compare(_product(_truth(), valid=valid), _target(), optics=OPTICS)
    assert np.isnan(report.deviation[25, 25]) and not report.valid[25, 25]
    assert report.stats["validity"]["target_coverage"] == pytest.approx(0.4)
    assert not report.stats["validity"]["coverage_ok"]
    assert any("covered" in w for w in report.warnings)


def test_custom_segments_and_shift():
    labels = np.zeros(PGRID.shape, int)
    labels[20:40, 20:30] = 1
    labels[20:40, 30:40] = 2
    # Structure measured 0.4 µm further right: the pose shift corrects it.
    shifted = np.zeros(PGRID.shape)
    shifted[20:40, 22:42] = 2.0
    labels = np.roll(labels, 2, axis=1)
    report = compare(_product(shifted), _target(), optics=OPTICS, shift_um=(0.0, 0.4),
                     segments=labels, registration={"method": "phase_correlation",
                                                    "confidence": 0.9, "residual": 0.01})
    assert set(report.stats["segments"]) == {"1", "2"}
    assert report.stats["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert report.to_message()[0]["registration"]["matrix"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.4]]
    assert report.stats["validity"]["registration_confidence"] == 0.9


def test_missing_optics_for_height_target_raises():
    with pytest.raises(ValidationError, match="optics"):
        compare(_product(_truth()), _target(), optics=None)


def test_report_round_trip():
    report = compare(_product(_truth(), noise=0.02), _target(), optics=OPTICS, sigma=0.02)
    header, buffers = report.to_message()
    again = DeviationReport.from_message(header, buffers)
    assert again.to_message()[0] == header
    np.testing.assert_array_equal(again.deviation, report.deviation)
