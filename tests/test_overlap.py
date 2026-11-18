"""Tests of the piston / tilt overlap fit (MI-21) on MI-22 fields."""

import numpy as np
import pytest

from mira.overlap import fit_piston_tilt, overlap_slices, remove_piston_tilt
from mira.products import Grid
from mira.synthetic import Feature, SpeckleNoise, SyntheticScene, make_tile
from mira.utils.exceptions import ValidationError

SCENE = SyntheticScene((Feature("rect", (0.0, 0.0), (10.0, 8.0), phase_rad=2.5),))
PIXEL = (0.2, 0.2)


@pytest.mark.parametrize(
    "piston, tilt",
    [(0.0, (0.0, 0.0)), (2.9, (0.05, -0.08)), (-3.0, (-0.3, 0.2)), (1.0, (0.0, 0.6))],
)
def test_noise_free_fit_is_exact(piston, tilt):
    ref = make_tile(SCENE, (0, 0), (96, 96), PIXEL)
    new = make_tile(SCENE, (0, 0), (96, 96), PIXEL, piston_rad=piston, tilt_rad_per_um=tilt)
    fit = fit_piston_tilt(new.field, ref.field, ref.grid)
    assert fit.piston_rad == pytest.approx(np.angle(np.exp(1j * piston)), abs=1e-5)
    assert fit.tilt_rad_per_um == pytest.approx(tilt, abs=1e-6)
    assert fit.residual_rad < 1e-5 and fit.coherence == pytest.approx(1.0, abs=1e-9)
    cleaned = remove_piston_tilt(new.field, new.grid, fit)
    np.testing.assert_allclose(np.angle(cleaned * np.conj(ref.field)), 0, atol=2e-5)


@pytest.mark.parametrize(
    "sigma_coh, tol_piston, tol_tilt",
    # Tolerances (synthetic, until M7) are about 3 standard deviations measured
    # over 30 seeds on 128² px of 0.2 µm with 0.2 rad white noise per tile:
    # white only: piston std 0.0021 rad, tilt std 0.00028 rad/µm;
    # + 0.15 rad coherent pattern (corr 1 µm, independent per tile): piston std
    # 0.031 rad, tilt std 0.0034 rad/µm (the pattern's own low-order content).
    [(0.0, 0.01, 1.5e-3), (0.15, 0.1, 1.5e-2)],
)
def test_noisy_fit_within_stated_tolerance(sigma_coh, tol_piston, tol_tilt):
    noise = SpeckleNoise(sigma_t_rad=0.2, sigma_coh_rad=sigma_coh, corr_um=1.0)
    ref = make_tile(SCENE, (0, 0), (128, 128), PIXEL, noise=noise, coherent_seed=1, rng=1)
    new = make_tile(SCENE, (0, 0), (128, 128), PIXEL, piston_rad=0.7,
                    tilt_rad_per_um=(0.02, -0.03), noise=noise, coherent_seed=2, rng=2)
    fit = fit_piston_tilt(new.field, ref.field, ref.grid)
    assert fit.piston_rad == pytest.approx(0.7, abs=tol_piston)
    assert fit.tilt_rad_per_um == pytest.approx((0.02, -0.03), abs=tol_tilt)
    assert 0.2 < fit.residual_rad < 0.5
    assert 0.8 < fit.coherence < 1.0


def test_mask_and_weight_are_used():
    ref = make_tile(SCENE, (0, 0), (64, 64), PIXEL)
    new = make_tile(SCENE, (0, 0), (64, 64), PIXEL, piston_rad=0.5)
    # Corrupt the structure area in the new field; masking it out restores the fit.
    corrupted = new.field.copy()
    substrate = SCENE.substrate_mask(ref.grid)
    corrupted[~substrate] *= np.exp(1j * 1.5)
    biased = fit_piston_tilt(corrupted, ref.field, ref.grid, fit_tilt=False)
    masked = fit_piston_tilt(corrupted, ref.field, ref.grid, mask=substrate, fit_tilt=False)
    weighted = fit_piston_tilt(corrupted, ref.field, ref.grid,
                               weight=substrate.astype(float), fit_tilt=False)
    assert abs(biased.piston_rad - 0.5) > 0.1
    assert masked.piston_rad == pytest.approx(0.5, abs=1e-6)
    assert weighted.piston_rad == pytest.approx(0.5, abs=1e-6)
    assert masked.n_pixels == int(substrate.sum())


def test_piston_refers_to_the_given_centre():
    ref = make_tile(SCENE, (0, 0), (64, 64), PIXEL)
    new = make_tile(SCENE, (0, 0), (64, 64), PIXEL, piston_rad=0.2, tilt_rad_per_um=(0.0, 0.1))
    fit = fit_piston_tilt(new.field, ref.field, ref.grid, centre_um=(0.0, 2.0))
    assert fit.piston_rad == pytest.approx(0.2 + 0.1 * 2.0, abs=1e-5)
    np.testing.assert_allclose(fit.phase_at(ref.grid)[32, 32],
                               0.2 + 0.1 * (ref.grid.origin_um[1] + 32 * 0.2), atol=1e-6)


def test_errors():
    grid = Grid((0, 0), PIXEL, (8, 8))
    ones = np.ones((8, 8), np.complex64)
    with pytest.raises(ValidationError, match="do not match"):
        fit_piston_tilt(ones[:4], ones, grid)
    with pytest.raises(ValidationError, match="usable"):
        fit_piston_tilt(ones, ones, grid, mask=np.zeros((8, 8), bool))


def test_overlap_slices_of_shifted_tiles():
    a = make_tile(SCENE, (0, 0), (50, 60), PIXEL)
    b = make_tile(SCENE, (2.0, -3.0), (50, 60), PIXEL)  # 10 px down, 15 px left
    sa, sb, common = overlap_slices(a.grid, b.grid)
    assert common.shape == (40, 45)
    np.testing.assert_allclose(a.field[sa], b.field[sb], atol=1e-6)
    phase, _ = SCENE.sample(common)
    np.testing.assert_allclose(np.angle(a.field[sa]), phase, atol=1e-6)
    far = make_tile(SCENE, (100.0, 0), (50, 60), PIXEL)
    assert overlap_slices(a.grid, far.grid) is None
    off = Grid((0.1, 0.0), PIXEL, (5, 5))
    with pytest.raises(ValidationError, match="resample"):
        overlap_slices(a.grid, off)
