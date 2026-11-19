"""Tests of the shared noise metrics (MI-31)."""

from __future__ import annotations

import numpy as np
import pytest

from mira.noise import (
    NoiseModel,
    correlation_length_um,
    fit_noise_model,
    positions_needed,
    radial_psd,
    reference_noise,
)
from mira.synthetic import SpeckleNoise, SyntheticScene, make_tile
from mira.utils.exceptions import ValidationError

PIXEL = (0.2, 0.2)


def test_reference_noise_of_white_phase_with_plane():
    rng = np.random.default_rng(3)
    yy, xx = np.indices((200, 200))
    phase = 0.4 + 0.01 * yy - 0.02 * xx + 0.15 * rng.standard_normal((200, 200))
    stats = reference_noise(phase, detrend="plane")
    assert stats.sigma_rad == pytest.approx(0.15, rel=0.02)  # 40 000 px: rel. SE 0.35 %
    assert stats.mad_sigma_rad == pytest.approx(0.15, rel=0.03)
    # Without detrending the plane dominates.
    assert reference_noise(phase, detrend="none").sigma_rad > 1.0


def test_reference_noise_on_complex_field_with_mask_and_wrap():
    """Complex input is referred to its mean phase, so a piston near π is fine."""
    tile = make_tile(
        SyntheticScene(()),
        (0, 0),
        (128, 128),
        PIXEL,
        piston_rad=3.1,
        noise=SpeckleNoise(sigma_t_rad=0.1),
        rng=1,
    )
    mask = np.zeros((128, 128), bool)
    mask[10:-10, 10:-10] = True
    stats = reference_noise(tile.field, mask, detrend="piston")
    assert stats.n_pixels == 108 * 108
    assert stats.sigma_rad == pytest.approx(0.1, rel=0.05)


def test_radial_psd_integrates_to_variance():
    rng = np.random.default_rng(5)
    phase = 0.2 * rng.standard_normal((128, 128))
    freq, psd = radial_psd(phase, PIXEL, detrend="none", n_bins=16)
    assert freq.shape == psd.shape == (16,)
    # White noise: flat PSD = variance * pixel area.
    flat = np.nanmean(psd)
    assert flat == pytest.approx(0.04 * 0.04, rel=0.05)
    assert np.nanstd(psd) / flat < 0.1


def test_correlation_length_of_synthetic_coherent_pattern():
    """make_tile filters white noise with sigma = corr_um: 1/e radius = 2 corr_um."""
    lengths = []
    for seed in range(5):
        tile = make_tile(
            SyntheticScene(()),
            (0, 0),
            (256, 256),
            PIXEL,
            noise=SpeckleNoise(sigma_coh_rad=0.3, corr_um=0.8),
            coherent_seed=seed,
        )
        lengths.append(correlation_length_um(np.angle(tile.field), PIXEL))
    assert np.mean(lengths) == pytest.approx(1.6, rel=0.1)


def test_fit_noise_model_recovers_terms():
    truth = NoiseModel(sigma_t_rad=0.3, sigma_coh_rad=0.08, rms_residual_rad2=0.0)
    nf = np.array([1, 2, 5, 10, 20, 1, 5, 10, 10, 10])
    npos = np.array([1, 1, 1, 1, 1, 4, 4, 4, 9, 16])
    rng = np.random.default_rng(0)
    sigma = truth.sigma(nf, npos) * (1 + 0.01 * rng.standard_normal(nf.size))
    fit = fit_noise_model(nf, npos, sigma)
    assert fit.sigma_t_rad == pytest.approx(0.3, rel=0.03)
    assert fit.sigma_coh_rad == pytest.approx(0.08, rel=0.05)
    # 9 positions x 10 frames: sqrt(0.09/90 + 0.0064/9) = 0.0414 rad
    assert positions_needed(fit, 0.0415, 10) == 9


def test_fit_noise_model_clips_negative_terms_and_validates():
    fit = fit_noise_model([1, 4, 16, 4], [1, 1, 1, 4], [0.4, 0.2, 0.1, 0.1])  # white only
    assert fit.sigma_coh_rad == pytest.approx(0.0, abs=1e-6)
    assert fit.sigma_t_rad == pytest.approx(0.4, rel=1e-6)
    with pytest.raises(ValidationError, match="separate"):
        fit_noise_model([1, 1], [1, 1], [0.3, 0.3])
    with pytest.raises(ValidationError):
        reference_noise(np.zeros((3, 3)))
    with pytest.raises(ValidationError, match="detrend"):
        reference_noise(np.zeros((10, 10)), detrend="quadratic")
