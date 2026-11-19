"""Tests of the shared fusion / stitching pipeline (MI-30, MI-40) on MI-22 fields.

Tolerances are stated per test; acceptance tolerances wait for M7.
"""

from __future__ import annotations

import numpy as np
import pytest

from mira.mosaic import (
    MosaicBuilder,
    MosaicConfig,
    border_band_estimate_um,
    build_mosaic,
    mosaic_grid,
    rasterise_polygons,
)
from mira.noise import reference_noise
from mira.products import Grid, LayerProduct
from mira.synthetic import Feature, SpeckleNoise, SyntheticScene, make_tile
from mira.utils.exceptions import ValidationError

PIXEL = (0.2, 0.2)
SHAPE = (128, 128)
COMMON = {
    "border_band_um": 1.0,
    "pose_tolerance_um": 1.0,
    "min_confidence": 0.1,
    "min_isotropy": 0.05,
}


def _fusion_tiles(seed=0, noise=None):
    rng = np.random.default_rng(seed)
    scene = SyntheticScene((Feature("rect", (0.0, 0.0), (8.0, 8.0), phase_rad=1.5),))
    noise = noise or SpeckleNoise(sigma_t_rad=0.2, sigma_coh_rad=0.2, corr_um=0.6)
    tiles = []
    for i, (dy, dx) in enumerate([(a, b) for a in (-4, 0, 4) for b in (-4, 0, 4)]):
        t = make_tile(
            scene,
            (dy, dx),
            SHAPE,
            PIXEL,
            noise=noise,
            n_frames=10,
            coherent_seed=5,
            rng=100 * seed + i,
            stage_error_um=tuple(rng.normal(0, 0.3, 2)),
            piston_rad=rng.uniform(-3, 3),
            tilt_rad_per_um=tuple(rng.normal(0, 0.01, 2)),
        )
        tiles.append(t.to_product(product_id=f"f{i}"))
    return scene, tiles, [[[-4, -4], [-4, 4], [4, 4], [4, -4]]]


def _raster_tiles(seed=0, step=15.0):
    rng = np.random.default_rng(seed)
    scene = SyntheticScene(
        (
            Feature("rect", (0.0, 0.0), (30.0, 30.0), phase_rad=9.0, edge_um=1.5),
            Feature("disc", (5.0, -6.0), (6.0, 6.0), phase_rad=3.0),
        )
    )
    order = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (0, 0),
    ]  # ring, centre
    tiles = []
    for i, (iy, ix) in enumerate(order):
        t = make_tile(
            scene,
            (iy * step, ix * step),
            SHAPE,
            PIXEL,
            noise=SpeckleNoise(sigma_t_rad=0.1),
            rng=100 * seed + i,
            stage_error_um=tuple(rng.normal(0, 0.3, 2)),
            piston_rad=rng.uniform(-3, 3),
            tilt_rad_per_um=tuple(rng.normal(0, 0.005, 2)),
        )
        tiles.append(t.to_product(product_id=f"s{i}"))
    return scene, tiles, [[[-15, -15], [-15, 15], [15, 15], [15, -15]]]


def test_border_band_estimate_matches_decision_g10():
    # 63x NA 1.4 oil, z = -2.6 um, n = 1.508 -> about 6.5 um (G10 SA6).
    assert border_band_estimate_um(-2.6, 1.4, 1.508) == pytest.approx(6.5, abs=0.1)


def test_rasterise_polygons_and_margin():
    grid = Grid((-5.0, -5.0), (0.5, 0.5), (21, 21))
    square = [[[-2.1, -2.1], [-2.1, 2.1], [2.1, 2.1], [2.1, -2.1]]]
    inside = rasterise_polygons(square, grid)
    assert inside.sum() == 9 * 9  # pixel centres -2.0 ... 2.0 in 0.5 steps
    wider = rasterise_polygons(square, grid, margin_um=1.0)
    assert wider.sum() > inside.sum() and wider[inside].all()
    with pytest.raises(ValidationError):
        rasterise_polygons([[[0, 0], [1, 1]]], grid)


def test_fusion_reduces_noise_and_keeps_the_scene():
    scene, tiles, occ = _fusion_tiles()
    product = build_mosaic(
        tiles, MosaicConfig.fusion(**COMMON, reference_margin_um=1.0), occupied_um=occ
    )
    assert isinstance(product, LayerProduct)
    assert product.quality["n_flagged"] == 0
    assert product.group_kind == "speckle"
    # Measured (Office PC): tile noise 0.20 rad -> fused 0.11 rad over the
    # whole reference region (coverage 1 ... 9).
    assert (
        product.quality["phase_noise_rad"]
        < 0.65 * product.quality["phase_noise_tiles_median_rad"]
    )
    truth, _ = scene.sample(product.grid)
    err = (product.phase_rad - truth)[product.valid]
    assert abs(err.mean()) < 0.03
    assert err.std() < 0.15
    # Fully covered centre: noise close to the M4 model
    # sqrt(0.2^2 / 90 + 0.2^2 / 9) = 0.070 rad (plus registration residuals).
    count9 = np.zeros(product.grid.shape, bool)
    count9[60:-60, 60:-60] = True
    centre = count9 & product.valid & rasterise_polygons(occ, product.grid, 1.0) == 0
    centre &= count9
    assert reference_noise(product.field, centre & product.valid).sigma_rad < 0.1


def test_fusion_mean_versus_median():
    """MI-30 comparison: complex mean vs per-pixel median of the tile phases."""
    scene, tiles, occ = _fusion_tiles(seed=1)
    base = dict(COMMON, reference_margin_um=1.0, phase_from="tiles", unwrap="none")
    mean = build_mosaic(
        tiles, MosaicConfig.fusion(**base, combine="mean"), occupied_um=occ
    )
    median = build_mosaic(
        tiles, MosaicConfig.fusion(**base, combine="median"), occupied_um=occ
    )
    ref = mean.valid & ~rasterise_polygons(occ, mean.grid, 2.0)
    ref[:50, :] = ref[-50:, :] = False  # well-covered region only
    ref[:, :50] = ref[:, -50:] = False
    noise_mean = reference_noise(mean.phase_rad, ref).sigma_rad
    noise_median = reference_noise(median.phase_rad, ref).sigma_rad
    assert noise_mean < noise_median  # Gaussian noise: the mean is more efficient

    def edge_width(product):
        row = product.phase_rad[product.grid.shape[0] // 2]
        x = product.grid.origin_um[1] + product.grid.pixel_um[1] * np.arange(row.size)
        sel = (x > -8) & (x < 0)
        prof = row[sel]
        lo, hi = np.percentile(prof, 5), np.percentile(prof, 95)
        rel = (prof - lo) / (hi - lo)
        return float(np.sum((rel > 0.1) & (rel < 0.9)) * product.grid.pixel_um[1])

    # Measured (seed 1): 10-90 % edge width 0.8 um (mean) vs 1.4 um (median);
    # the complex mean keeps the edge at least as sharp as the median.
    assert edge_width(mean) <= edge_width(median) + 0.2


def test_stitching_raster_with_unwrapping_per_tile():
    scene, tiles, occ = _raster_tiles()
    config = MosaicConfig.stitching(**COMMON, reference_margin_um=2.0)
    product = build_mosaic(tiles, config, occupied_um=occ)
    assert product.group_kind == "layer"
    assert product.quality["n_tiles"] == 9
    truth, _ = scene.sample(product.grid)
    err = (product.phase_rad - truth)[product.valid]
    # 9 rad plateau (> 2 pi) recovered without 2 pi errors; along-edge stage
    # errors (0.3 um) stay where an overlap holds only a straight edge.
    assert abs(err.mean()) < 0.05
    assert err.std() < 0.2
    assert np.mean(np.abs(err) > np.pi) < 1e-3
    ring = [t for t in product.tiles if t.tile_index < 8]
    assert all(t.anchor == "reference_region" for t in ring)
    centre = product.tiles[-1]
    assert centre.reference_fraction == 0.0 and centre.anchor == "overlap"
    assert all(t.reference_source == "mask_a_priori" for t in product.tiles)
    assert all(t.unwrap["method"] == "fast2d" for t in product.tiles)
    # Contract round trip.
    header, buffers = product.to_message()
    back = LayerProduct.from_message(
        header,
        (
            [b for _, b in buffers]
            if buffers and isinstance(buffers[0], tuple)
            else buffers
        ),
    )
    assert back.tiles[3].to_dict() == product.tiles[3].to_dict()


def test_inconsistent_tile_is_flagged():
    scene, tiles, occ = _raster_tiles(seed=2)
    bad = tiles[3]
    field = np.array(bad.field)
    field[5:40, 5:45] *= np.exp(
        1j * 2.5
    )  # local step, no multiple of 2 pi (in the overlap)
    tiles[3] = bad.__class__(**{**bad.__dict__, "field": field})
    product = build_mosaic(
        tiles,
        MosaicConfig.stitching(**COMMON, reference_margin_um=2.0),
        occupied_um=occ,
    )
    record = product.tiles[3]
    assert record.flagged
    assert record.unwrap["consistent"] is False
    assert product.quality["n_flagged"] >= 1


def test_incremental_builder_and_validation():
    scene, tiles, occ = _raster_tiles(seed=3)
    grid = mosaic_grid([t.grid.centre_um for t in tiles], SHAPE, PIXEL, 1.0)
    builder = MosaicBuilder(
        grid,
        MosaicConfig.stitching(**COMMON),
        occupied_um=occ,
        product_id="lp-1",
        metadata={"structure_id": "pillar", "layer_index": 4},
    )
    records = [
        builder.add_tile(t, tile_index=i, role="ring" if i < 8 else "center")
        for i, t in enumerate(tiles)
    ]
    assert records[0]["registration"]["method"] == "first_tile"
    assert all("time_s" in r for r in records)
    product = builder.close(include_field=False)
    assert (
        product.field is None
        and product.structure_id == "pillar"
        and product.layer_index == 4
    )
    assert product.timings_s["tiles"] > 0
    with pytest.raises(ValidationError, match="closed"):
        builder.add_tile(tiles[0])
    with pytest.raises(ValidationError, match="mode"):
        MosaicConfig(mode="other", **COMMON)
    with pytest.raises(ValidationError, match="border band"):
        MosaicBuilder(
            grid, MosaicConfig.stitching(**{**COMMON, "border_band_um": 20.0})
        ).add_tile(tiles[0])


def test_min_count_masks_low_coverage():
    scene, tiles, occ = _fusion_tiles(seed=4)
    product = build_mosaic(
        tiles, MosaicConfig.fusion(**COMMON, min_count=9), occupied_um=occ
    )
    covered = build_mosaic(
        tiles, MosaicConfig.fusion(**COMMON, min_count=1), occupied_um=occ
    )
    assert 0 < product.valid.sum() < covered.valid.sum()
    assert product.quality["coverage_min"] == 9
