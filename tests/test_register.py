"""Tests of the registration by stage prior and image (MI-20) on MI-22 fields.

Tolerances are stated per test; acceptance tolerances wait for M7.
"""

from __future__ import annotations

import numpy as np
import pytest

from mira.register import corrected_grid, register_pair
from mira.synthetic import Feature, SpeckleNoise, SyntheticScene, make_tile
from mira.utils.exceptions import ValidationError

PIXEL = (0.2, 0.2)
SHAPE = (128, 128)
SCENE = SyntheticScene(
    (
        Feature("rect", (0.0, 4.0), (6.0, 9.0), phase_rad=1.6),
        Feature("disc", (-5.0, -6.0), (4.0, 4.0), phase_rad=1.1),
        Feature("rect", (7.0, -3.0), (2.0, 12.0), phase_rad=0.8),
        Feature("disc", (6.0, 9.0), (3.0, 3.0), phase_rad=1.3),
    )
)
CONF = 0.1  # minimum confidence used in these tests (M7 pending)


def _pair(
    stage_error, *, noise=None, offset=(3.0, 9.0), scene=SCENE, seed=0, n_frames=1
):
    ref = make_tile(
        scene,
        (0.0, 0.0),
        SHAPE,
        PIXEL,
        noise=noise,
        coherent_seed=11,
        rng=seed,
        n_frames=n_frames,
    )
    mov = make_tile(
        scene,
        offset,
        SHAPE,
        PIXEL,
        noise=noise,
        stage_error_um=stage_error,
        piston_rad=0.9,
        tilt_rad_per_um=(0.01, -0.02),
        coherent_seed=11,  # same optics-fixed pattern, moves with the tile
        rng=seed + 1,
        n_frames=n_frames,
    )
    return ref, mov


@pytest.mark.parametrize(
    "error", [(0.37, -0.52), (-0.81, 0.13), (0.0, 0.0), (0.05, 0.93)]
)
def test_recovers_stage_error_noise_free(error):
    ref, mov = _pair(error)
    reg = register_pair(
        mov.field,
        mov.stage_grid,
        ref.field,
        ref.stage_grid,
        pose_tolerance_um=1.5,
        min_confidence=CONF,
    )
    assert not reg.flagged and reg.method == "phase_correlation"
    # Truth: true minus reported position = -stage_error; 0.015 µm = 1.5 upsampling steps.
    np.testing.assert_allclose(reg.shift_um, -np.array(error), atol=0.015)
    fixed = corrected_grid(mov.stage_grid, reg)
    np.testing.assert_allclose(fixed.origin_um, mov.grid.origin_um, atol=0.02)


def test_recovers_stage_error_with_speckle_noise():
    """White + optics-fixed coherent noise, 10 frames per position (T85 start
    value): the coherent peak lies at the full tile offset and is excluded by
    the tolerance window. The error is unbiased and noise-limited (measured
    over 20 seeds: rms 0.017 µm with white noise only, 10 frames)."""
    noise = SpeckleNoise(sigma_t_rad=0.3, sigma_coh_rad=0.3, corr_um=0.6)
    errors = []
    for seed in range(10):
        ref, mov = _pair((0.4, -0.3), noise=noise, seed=2 * seed, n_frames=10)
        reg = register_pair(
            mov.field,
            mov.stage_grid,
            ref.field,
            ref.stage_grid,
            pose_tolerance_um=1.5,
            min_confidence=CONF,
        )
        assert not reg.flagged
        errors.append(np.array(reg.shift_um) + np.array([0.4, -0.3]))
    rms = np.sqrt(np.mean(np.square(errors)))
    assert rms < 0.04  # 0.2 px


@pytest.mark.parametrize("signal", ["amplitude", "phase"])
def test_other_signals(signal):
    scene = SyntheticScene(
        tuple(
            Feature(f.kind, f.centre_um, f.size_um, f.phase_rad, amplitude=0.7)
            for f in SCENE.features
        )
    )
    ref, mov = _pair((0.3, 0.2), scene=scene)
    reg = register_pair(
        mov.field,
        mov.stage_grid,
        ref.field,
        ref.stage_grid,
        pose_tolerance_um=1.5,
        min_confidence=CONF,
        signal=signal,
    )
    np.testing.assert_allclose(reg.shift_um, (-0.3, -0.2), atol=0.03)


def test_featureless_substrate_falls_back_to_stage():
    noise = SpeckleNoise(sigma_t_rad=0.3)
    ref, mov = _pair((0.4, -0.3), noise=noise, scene=SyntheticScene(()))
    reg = register_pair(
        mov.field,
        mov.stage_grid,
        ref.field,
        ref.stage_grid,
        pose_tolerance_um=1.5,
        min_confidence=0.5,
    )
    assert reg.flagged and reg.method == "stage_prior"
    assert reg.shift_um == (0.0, 0.0)
    assert "confidence" in reg.reason
    assert reg.to_record()["method"] == "stage_prior"


def test_small_or_missing_overlap_falls_back():
    ref, mov = _pair((0.0, 0.0), offset=(0.0, 24.0))  # 128 px * 0.2 = 25.6 µm wide
    reg = register_pair(
        mov.field,
        mov.stage_grid,
        ref.field,
        ref.stage_grid,
        pose_tolerance_um=1.0,
        min_confidence=CONF,
    )
    assert reg.flagged and "overlap" in reg.reason
    ref, mov = _pair((0.0, 0.0), offset=(0.0, 40.0))
    reg = register_pair(
        mov.field,
        mov.stage_grid,
        ref.field,
        ref.stage_grid,
        pose_tolerance_um=1.0,
        min_confidence=CONF,
    )
    assert reg.flagged and reg.reason == "no overlap"


def test_error_beyond_tolerance_is_not_found():
    ref, mov = _pair((1.6, 0.0))
    reg = register_pair(
        mov.field,
        mov.stage_grid,
        ref.field,
        ref.stage_grid,
        pose_tolerance_um=0.6,
        min_confidence=0.9,
    )
    assert reg.flagged


def test_validation():
    ref, mov = _pair((0.0, 0.0))
    with pytest.raises(ValidationError, match="signal"):
        register_pair(
            mov.field,
            mov.stage_grid,
            ref.field,
            ref.stage_grid,
            pose_tolerance_um=1.0,
            min_confidence=CONF,
            signal="x",
        )
    with pytest.raises(ValidationError, match="shapes"):
        register_pair(
            mov.field[:10],
            mov.stage_grid,
            ref.field,
            ref.stage_grid,
            pose_tolerance_um=1.0,
            min_confidence=CONF,
        )
