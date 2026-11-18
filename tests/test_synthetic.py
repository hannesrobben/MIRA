"""Tests of the synthetic field generator (MI-22)."""

import numpy as np
import pytest

from mira.products import Grid
from mira.synthetic import Feature, SpeckleNoise, SyntheticScene, make_tile

SCENE = SyntheticScene(
    (
        Feature("rect", (0.0, 0.0), (8.0, 6.0), phase_rad=1.2, amplitude=0.8),
        Feature("disc", (5.0, -6.0), (3.0, 3.0), phase_rad=0.6),
    )
)
PIXEL = (0.2, 0.2)


def test_noise_free_tile_equals_scene_plus_piston_and_tilt():
    tile = make_tile(SCENE, (0.3, -0.7), (64, 80), PIXEL, piston_rad=0.4,
                     tilt_rad_per_um=(0.01, -0.02))
    yr = (np.arange(64) - 31.5) * 0.2
    xr = (np.arange(80) - 39.5) * 0.2
    expected = tile.truth_phase_rad + 0.4 + 0.01 * yr[:, None] - 0.02 * xr[None, :]
    np.testing.assert_allclose(np.angle(tile.field * np.exp(-1j * expected)), 0, atol=1e-6)
    assert tile.grid.centre_um == pytest.approx((0.3, -0.7))
    # Plateau and substrate values of the truth.
    assert tile.truth_phase_rad[32, 40] == pytest.approx(1.2, abs=1e-3)
    assert abs(tile.field[32, 40]) == pytest.approx(0.8, abs=1e-3)
    assert tile.truth_phase_rad[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_subpixel_positions_are_exact():
    # Two tiles shifted by 0.37 pixel sample the same analytic scene.
    a = make_tile(SCENE, (0.0, 0.0), (32, 32), PIXEL)
    b = make_tile(SCENE, (0.0, 0.074), (32, 32), PIXEL)
    grid = Grid(b.grid.origin_um, PIXEL, (32, 32))
    phase, _ = SCENE.sample(grid)
    np.testing.assert_allclose(b.truth_phase_rad, phase)
    assert b.grid.origin_um[1] - a.grid.origin_um[1] == pytest.approx(0.074)


def test_stage_error_moves_only_the_reported_grid():
    tile = make_tile(SCENE, (1.0, 2.0), (16, 16), PIXEL, stage_error_um=(0.5, -0.25))
    assert tile.stage_grid.centre_um == pytest.approx((1.5, 1.75))
    assert tile.grid.centre_um == pytest.approx((1.0, 2.0))
    product = tile.to_product(product_id="p")
    assert product.grid == tile.stage_grid
    assert product.position_actual_um["x"] == pytest.approx(1.75)


def test_white_noise_level_and_frame_averaging():
    empty = SyntheticScene()
    noise = SpeckleNoise(sigma_t_rad=0.2)
    one = make_tile(empty, (0, 0), (128, 128), PIXEL, noise=noise, rng=1)
    ten = make_tile(empty, (0, 0), (128, 128), PIXEL, noise=noise, n_frames=10, rng=1)
    # Tolerance 10 %: 16384 samples give a relative std error of about 0.6 %.
    assert np.angle(one.field).std() == pytest.approx(0.2, rel=0.1)
    assert np.angle(ten.field).std() == pytest.approx(0.2 / np.sqrt(10), rel=0.1)


def test_coherent_pattern_is_fixed_in_camera_coordinates():
    empty = SyntheticScene()
    noise = SpeckleNoise(sigma_coh_rad=0.3, corr_um=1.0)
    a = make_tile(empty, (0, 0), (128, 128), PIXEL, noise=noise, coherent_seed=5)
    b = make_tile(empty, (10, 10), (128, 128), PIXEL, noise=noise, coherent_seed=5)
    c = make_tile(empty, (0, 0), (128, 128), PIXEL, noise=noise, coherent_seed=6)
    np.testing.assert_allclose(np.angle(a.field), np.angle(b.field), atol=1e-6)
    assert np.angle(a.field).std() == pytest.approx(0.3, rel=0.05)
    # Independent patterns: the correlation of two seeds is small.
    r = np.corrcoef(np.angle(a.field).ravel(), np.angle(c.field).ravel())[0, 1]
    assert abs(r) < 0.15
    # Neighbouring pixels (0.2 µm < corr 1 µm) are strongly correlated.
    p = np.angle(a.field)
    r1 = np.corrcoef(p[:, :-1].ravel(), p[:, 1:].ravel())[0, 1]
    assert r1 > 0.9


def test_seeded_tiles_are_reproducible():
    noise = SpeckleNoise(sigma_t_rad=0.1, sigma_coh_rad=0.1, amplitude_rel=0.1)
    a = make_tile(SCENE, (0, 0), (32, 32), PIXEL, noise=noise, rng=3)
    b = make_tile(SCENE, (0, 0), (32, 32), PIXEL, noise=noise, rng=3)
    np.testing.assert_array_equal(a.field, b.field)
    assert a.field.dtype == np.complex64


def test_substrate_mask():
    tile = make_tile(SCENE, (0, 0), (100, 100), PIXEL)
    mask = SCENE.substrate_mask(tile.grid)
    assert mask[0, 0] and not mask[50, 50]
    assert np.all(np.abs(tile.truth_phase_rad[mask]) < 0.02)
