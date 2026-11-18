"""Tests of the forward model and the domain choice (MI-51)."""

import math

import numpy as np
import pytest

from mira.forward import (
    Optics,
    choose_domain,
    height_to_phase_rad,
    phase_to_height_um,
    to_domain,
)
from mira.utils.exceptions import ValidationError

INSITU = dict(wavelength_um=0.674994, n_structure=1.52, n_surround=1.508, phase_sign=1)


def test_known_values_and_inverse():
    # 15 µm of SZ2080 in-situ: 2π · 0.012 · 15 / 0.674994 ≈ 1.6755 rad (< π, O5).
    phase = height_to_phase_rad(15.0, **INSITU)
    assert phase == pytest.approx(2 * math.pi * 0.012 * 15 / 0.674994, rel=1e-12)
    assert phase == pytest.approx(1.6755, abs=1e-4)
    h = np.linspace(-1, 15, 7)
    np.testing.assert_allclose(phase_to_height_um(height_to_phase_rad(h, **INSITU), **INSITU), h)
    flipped = dict(INSITU, phase_sign=-1)
    assert height_to_phase_rad(1.0, **flipped) == pytest.approx(-height_to_phase_rad(1.0, **INSITU))


def test_parity_with_offaxisholo():
    fieldops = pytest.importorskip("offaxisholo.reconstruction.fieldops")
    h = np.linspace(-2, 10, 11)
    for optics in (INSITU, dict(INSITU, n_surround=1.0, phase_sign=-1)):
        np.testing.assert_array_equal(height_to_phase_rad(h, **optics),
                                      fieldops.height_to_phase_rad(h, **optics))
        np.testing.assert_array_equal(phase_to_height_um(h, **optics),
                                      fieldops.phase_to_height_um(h, **optics))


@pytest.mark.parametrize("bad", [dict(phase_sign=0), dict(n_surround=1.52), dict(wavelength_um=0)])
def test_invalid_optics(bad):
    with pytest.raises(ValueError):
        height_to_phase_rad(1.0, **dict(INSITU, **bad))
    with pytest.raises(ValidationError):
        Optics(**dict(INSITU, **bad))


def test_optics_from_sources_names_missing_values():
    with pytest.raises(ValidationError, match=r"n_surround \(from product\).*phase_sign"):
        Optics.from_sources(wavelength_um=0.674994, n_structure=1.52, n_surround=None,
                            phase_sign=None, sources={"n_surround": "product",
                                                      "phase_sign": "parameter set"})
    optics = Optics.from_sources(**INSITU, sources={"n_structure": "target"})
    assert optics.delta_n == pytest.approx(0.012)
    calibrated = optics.with_delta_n(0.0135, "DeltaNResult run 3")
    assert calibrated.n_structure == pytest.approx(1.5215)
    assert "calibrated" in calibrated.sources["n_structure"]


def test_choose_domain():
    assert choose_domain("height", optics_known=True) == "phase"
    assert choose_domain("phase", optics_known=False) == "phase"
    assert choose_domain("height", optics_known=True, fixed="height") == "height"
    with pytest.raises(ValidationError, match="optics"):
        choose_domain("height", optics_known=False)
    with pytest.raises(ValidationError):
        choose_domain("phase", optics_known=False, fixed="height")
    with pytest.raises(ValidationError):
        choose_domain("intensity", optics_known=True)


def test_to_domain():
    optics = Optics(**INSITU)
    assert to_domain(2.0, "height", "height", None) == 2.0
    assert to_domain(1.0, "height", "phase", optics) == pytest.approx(
        height_to_phase_rad(1.0, **INSITU))
    with pytest.raises(ValidationError):
        to_domain(1.0, "phase", "height", None)
