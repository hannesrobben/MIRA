"""Tests of the DHMConnect group plug-in (MI-60) on MI-22 fields.

The plug-in is driven like the service would drive it: a ``GroupSpec``
(vendored example shape), tiles as products or ``ProductSet`` messages, and
the ``LayerProduct`` message as output. Tolerances as in ``test_mosaic.py``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from mira.plugins.dhmconnect import (
    GroupSession,
    PluginParameters,
    describe,
    open_group,
    run_group,
)
from mira.products import LayerProduct, TileProduct
from mira.synthetic import Feature, SpeckleNoise, SyntheticScene, make_tile
from mira.utils.exceptions import ValidationError

CONTRACTS = Path(__file__).parent / "contracts"
PIXEL = (0.2, 0.2)
SHAPE = (128, 128)
PARAMS = {
    "pose_tolerance_um": 1.0,
    "min_confidence": 0.1,
    "min_isotropy": 0.05,
    "optical_margin_um": 1.0,
}
SERVICE_IDS = {
    "experiment_uuid": "exp-1",
    "context_id": "ctx-1",
    "job_id": "job-1",
    "parameter_set_id": "ps-63x",
    "config_hash": "0" * 64,
}
RING_FIRST = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (0, 0),
]


def _spec(centres, occupied, *, unwrap="fast2d", min_positions=9):
    spec = json.loads((CONTRACTS / "GroupSpec.example.json").read_text("utf-8"))
    spec["structure"]["occupied_um"] = occupied
    spec["positions"] = [
        {
            "tile_index": i,
            "planned_um": {"x": cx, "y": cy, "z": 0.0},
            "role": "center" if (cy, cx) == (0.0, 0.0) else "ring",
            "n_frames": 10,
        }
        for i, (cy, cx) in enumerate(centres)
    ]
    spec["stitching"].update(
        {"border_band_um": 1.0, "unwrap": unwrap, "min_positions": min_positions}
    )
    return spec


def _tiles(scene, centres, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i, centre in enumerate(centres):
        tile = make_tile(
            scene,
            centre,
            SHAPE,
            PIXEL,
            noise=SpeckleNoise(sigma_t_rad=0.1),
            rng=100 * seed + i,
            stage_error_um=tuple(rng.normal(0, 0.3, 2)),
            piston_rad=rng.uniform(-3, 3),
            tilt_rad_per_um=tuple(rng.normal(0, 0.005, 2)),
        )
        out.append(
            tile.to_product(
                product_id=f"ps-{i}",
                tile_index=i,
                group_id="g-1",
                layer_index=4,
                n_surround=1.508,
            )
        )
    return out


def _raster(seed=0, step=15.0):
    scene = SyntheticScene(
        (
            Feature("rect", (0.0, 0.0), (30.0, 30.0), phase_rad=9.0, edge_um=1.5),
            Feature("disc", (5.0, -6.0), (6.0, 6.0), phase_rad=3.0),
        )
    )
    centres = [(iy * step, ix * step) for iy, ix in RING_FIRST]
    occupied = [[[-15, -15], [-15, 15], [15, 15], [15, -15]]]
    return scene, centres, occupied, _tiles(scene, centres, seed)


def _speckle_set(seed=0):
    scene = SyntheticScene((Feature("rect", (0.0, 0.0), (8.0, 8.0), phase_rad=1.5),))
    centres = [(dy, dx) for dy in (-2.0, 0.0, 2.0) for dx in (-2.0, 0.0, 2.0)]
    occupied = [[[-4, -4], [-4, 4], [4, 4], [4, -4]]]
    return scene, centres, occupied, _tiles(scene, centres, seed)


def _open(spec, **kwargs):
    return open_group(
        spec, PARAMS, tile_shape=SHAPE, pixel_um=PIXEL, group_id="g-1", **kwargs
    )


def test_describe_lists_the_capabilities():
    info = describe()
    assert info["name"] == "mira.mosaic"
    assert set(info["required_parameters"]) == {
        "pose_tolerance_um",
        "min_confidence",
        "min_isotropy",
    }
    assert "fast2d" in info["unwrap_methods"]
    json.dumps(info)


def test_stitching_group_incrementally_with_messages():
    scene, centres, occupied, tiles = _raster()
    session = _open(_spec(centres, occupied))
    assert session.mode == "stitching" and session.n_planned == 9
    for tile in tiles:
        header, buffers = tile.to_message(include_field=True)
        record = session.add_tile((json.loads(json.dumps(header)), buffers))
        json.dumps(record)  # records are JSON
        assert record["tile_index"] == tile.tile_index and not record["duplicate"]
    status = session.status()
    assert status["n_added"] == 9 and not status["closed"]
    header, buffers = session.close_message()
    product = LayerProduct.from_message(header, [b for _, b in buffers])
    assert session.closed
    assert product.field is None  # N1: field on request only
    assert product.structure_id == "pillar_03" and product.layer_index == 4
    assert product.group_id == "g-1" and product.group_kind == "layer"
    assert product.quality["plugin"] == "mira.mosaic"
    assert list(product.quality["missing_positions"]) == []
    per_tile = session.timings
    assert [t["tile_index"] for t in per_tile] == list(range(9))
    assert all(t["total_s"] >= t["mosaic_s"] > 0 for t in per_tile)
    assert all(t.extra["time_s"] > 0 for t in product.tiles)
    assert product.timings_s["group"] > 0
    assert product.tiles[0].planned_um == {"x": -15.0, "y": -15.0, "z": 0.0}
    truth, _ = scene.sample(product.grid)
    err = (product.phase_rad - truth)[product.valid]
    assert np.mean(np.abs(err) > np.pi) < 1e-3
    assert err.std() < 0.35


def test_small_structure_is_fused():
    scene, centres, occupied, tiles = _speckle_set()
    session = _open(_spec(centres, occupied))
    assert session.mode == "fusion"
    for tile in tiles:
        session.add_tile(tile)
    product = session.close()
    assert product.quality["mode"] == "fusion"
    assert (
        product.quality["phase_noise_rad"]
        < product.quality["phase_noise_tiles_median_rad"]
    )
    forced = open_group(
        _spec(centres, occupied),
        {**PARAMS, "mode": "stitching"},
        tile_shape=SHAPE,
        pixel_um=PIXEL,
    )
    assert forced.mode == "stitching"


def test_run_group_matches_the_session_and_validates_with_dhmconnect():
    scene, centres, occupied, tiles = _speckle_set(seed=1)
    params = PluginParameters(**PARAMS, include_field=True)
    header, buffers = run_group(
        _spec(centres, occupied),
        params,
        tiles,
        tile_shape=SHAPE,
        pixel_um=PIXEL,
        group_id="g-1",
        metadata=SERVICE_IDS,
    )
    assert any(name == "field" for name, _ in buffers)
    schemas = pytest.importorskip("dhmconnect.schemas")
    schemas.validate(json.loads(json.dumps(header)), "LayerProduct")


def test_duplicates_missing_positions_and_short_groups():
    scene, centres, occupied, tiles = _raster(seed=2)
    session = _open(_spec(centres, occupied))
    first = session.add_tile(tiles[0])
    again = session.add_tile(tiles[0])
    assert again["duplicate"] and again["tile_index"] == first["tile_index"]
    other = TileProduct(**{**tiles[0].__dict__, "product_id": "ps-x"})
    with pytest.raises(ValidationError, match="already added"):
        session.add_tile(other)
    for tile in tiles[1:6]:
        session.add_tile(tile)
    product = session.close()
    assert session.close() is product  # idempotent
    assert list(product.quality["missing_positions"]) == [6, 7, 8]
    assert any("missing planned positions" in w for w in product.warnings)
    assert any("below min_positions" in w for w in product.warnings)
    with pytest.raises(ValidationError, match="closed"):
        session.add_tile(tiles[6])


def test_validation_of_spec_tiles_and_parameters():
    scene, centres, occupied, tiles = _raster(seed=3)
    spec = _spec(centres, occupied)
    session = _open(spec)
    with pytest.raises(ValidationError, match="not a planned position"):
        session.add_tile(TileProduct(**{**tiles[0].__dict__, "tile_index": 42}))
    with pytest.raises(ValidationError, match="group"):
        session.add_tile(TileProduct(**{**tiles[0].__dict__, "group_id": "g-2"}))
    with pytest.raises(ValidationError, match="No tiles"):
        session.close()
    for kind, match in (("background", "not a MIRA job"),):
        bad = copy.deepcopy(spec)
        bad["group_kind"] = kind
        with pytest.raises(ValidationError, match=match):
            _open(bad)
    with pytest.raises(ValidationError, match="not implemented"):
        _open(_spec(centres, occupied, unwrap="temporal"))
    bad = copy.deepcopy(spec)
    bad["schema_version"] = "2.0"
    with pytest.raises(ValidationError, match="not supported"):
        _open(bad)
    bad = copy.deepcopy(spec)
    bad["positions"].append(dict(bad["positions"][0]))
    with pytest.raises(ValidationError, match="twice"):
        _open(bad)
    with pytest.raises(ValidationError, match="Unknown"):
        PluginParameters.from_dict({**PARAMS, "tolerance": 1})
    with pytest.raises(TypeError):
        PluginParameters.from_dict({"pose_tolerance_um": 1.0})  # M7: no defaults
    with pytest.raises(ValidationError, match="config must not set"):
        PluginParameters(**PARAMS, config={"unwrap": "none"})
    assert PluginParameters(**PARAMS).to_dict()["min_isotropy"] == 0.05


def test_session_state_holds_no_unpicklable_inputs():
    """Inputs and outputs are JSON + arrays (worker processes, D8)."""
    scene, centres, occupied, tiles = _speckle_set(seed=4)
    session = GroupSession(
        _spec(centres, occupied), PARAMS, tile_shape=SHAPE, pixel_um=PIXEL
    )
    json.dumps(session.params.to_dict())
    json.dumps(session.status())
    for tile in tiles:
        session.add_tile(tile.to_message(include_field=True))
    header, buffers = session.close_message()
    json.dumps(header)
    assert all(isinstance(b, np.ndarray) for _, b in buffers)
