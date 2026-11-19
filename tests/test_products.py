"""Tests of the product data model (MI-10) against the vendored contracts.

The examples in ``tests/contracts/`` are exported from DHMConnect
(``python -m dhmconnect.schemas.export``, commit in ``MANIFEST.json``).
"""

import copy
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from mira import StructureProcessor
from mira.products import Grid, LayerProduct, TileProduct, TileRecord
from mira.utils.exceptions import ValidationError

CONTRACTS = Path(__file__).parent / "contracts"
RNG = np.random.default_rng(1234)


def _example(name):
    return json.loads((CONTRACTS / f"{name}.example.json").read_text(encoding="utf-8"))


def _synthetic_buffers(descriptors, *, as_bytes=False):
    """One random array per descriptor (phase in (-pi, pi], intensity > 0)."""
    buffers = []
    for d in descriptors:
        shape = tuple(d["shape"])
        if d["name"] == "phase_rad":
            data = RNG.uniform(-3.1, 3.1, shape).astype("<f4")
        elif d["name"] == "intensity_rel":
            data = RNG.uniform(0.2, 1.5, shape).astype("<f4")
        elif d["name"] == "valid":
            data = (RNG.uniform(size=shape) > 0.1).astype("u1")
        else:
            data = (RNG.normal(size=shape) + 1j * RNG.normal(size=shape)).astype("<c8")
        buffers.append((d["name"], data.tobytes() if as_bytes else data))
    return buffers


def _wrap(a):
    return np.angle(np.exp(1j * a))


# ------------------------------------------------------------- TileProduct
def test_tile_from_product_set_example():
    header = _example("ProductSet")
    buffers = _synthetic_buffers(header["reconstruction"]["buffers"])
    tile = TileProduct.from_message(header, buffers)

    assert tile.field.dtype == np.complex64
    assert tile.grid.shape == (512, 512)
    assert tile.grid.pixel_um == (0.1738, 0.1738)
    actual = header["capture"]["position_actual_um"]
    # The image centre lies at the actual stage position (rotation 0, MI-11 later).
    assert tile.grid.centre_um == pytest.approx((actual["y"], actual["x"]), abs=1e-9)
    assert tile.z_um == -2.4 and tile.propagation_index == 1.508
    assert tile.n_structure == 1.52 and tile.n_surround == 1.508
    assert tile.tile_index == 3 and tile.structure_id == "pillar_03"
    assert tile.material["states"]["uncured"]["n"] == 1.508

    phase, intensity = (b[1] for b in buffers)
    np.testing.assert_allclose(_wrap(tile.phase_rad - phase), 0, atol=2e-6)
    np.testing.assert_allclose(tile.intensity_rel, intensity, rtol=2e-6)


def test_tile_message_round_trip_keeps_header():
    header = _example("ProductSet")
    buffers = _synthetic_buffers(header["reconstruction"]["buffers"], as_bytes=True)
    tile = TileProduct.from_message(header, buffers)
    out, out_buffers = tile.to_message()

    assert [name for name, _ in out_buffers] == ["phase_rad", "intensity_rel"]
    for key, value in header.items():
        if key != "reconstruction":
            assert out[key] == value, key
    for key, value in header["reconstruction"].items():
        assert out["reconstruction"][key] == value, key

    again = TileProduct.from_message(out, out_buffers)
    np.testing.assert_allclose(again.field, tile.field, rtol=1e-5, atol=1e-6)
    assert again.grid == tile.grid


def test_tile_with_field_buffer_uses_it():
    header = _example("ProductSet")
    descriptors = header["reconstruction"]["buffers"] + [
        {
            "name": "field",
            "dtype": "complex64",
            "shape": [512, 512],
            "byte_order": "little",
        }
    ]
    header["reconstruction"]["buffers"] = descriptors
    buffers = _synthetic_buffers(descriptors)
    tile = TileProduct.from_message(header, buffers)
    np.testing.assert_array_equal(tile.field, buffers[2][1])
    out, out_buffers = tile.to_message(include_field=True)
    assert [d["name"] for d in out["reconstruction"]["buffers"]][-1] == "field"
    np.testing.assert_array_equal(out_buffers[-1][1], buffers[2][1])


def test_tile_from_reconstruction_duck_typed():
    recon = _example("ReconstructionResult")
    buffers = _synthetic_buffers(recon["buffers"])

    class FakeResult:
        def to_message(self):
            return copy.deepcopy(recon), buffers

    capture = _example("CaptureContext")
    tile = TileProduct.from_reconstruction(FakeResult(), capture, product_id="ps-1")
    assert tile.product_id == "ps-1"
    assert tile.background_id == recon["quality"]["background_id"]
    assert tile.config_hash == recon["config_hash"]
    assert tile.capture["capture_id"] == capture["capture_id"]


def test_tile_requires_physical_parameters():
    header = _example("ProductSet")
    del header["reconstruction"]["propagation_index"]
    with pytest.raises(ValidationError, match="propagation_index"):
        TileProduct.from_message(
            header, _synthetic_buffers(header["reconstruction"]["buffers"])
        )


def test_tile_rejects_wrong_message_and_bad_buffers():
    header = _example("ProductSet")
    buffers = _synthetic_buffers(header["reconstruction"]["buffers"])
    bad = dict(header, message_type="LayerProduct")
    with pytest.raises(ValidationError, match="Not a ProductSet"):
        TileProduct.from_message(bad, buffers)
    bad = dict(header, schema_version="2.0")
    with pytest.raises(ValidationError, match="schema_version"):
        TileProduct.from_message(bad, buffers)
    with pytest.raises(ValidationError, match="buffers described"):
        TileProduct.from_message(header, buffers[:1])
    with pytest.raises(ValidationError, match="values for shape"):
        TileProduct.from_message(header, [buffers[0], ("intensity_rel", b"\0" * 16)])


def test_tile_from_arrays_and_immutability():
    grid = Grid.centred({"x": 10.0, "y": 20.0, "z": 0.0}, (0.2, 0.1), (4, 6))
    phase = np.full((4, 6), 0.5, np.float32)
    tile = TileProduct.from_arrays(
        phase_rad=phase,
        intensity_rel=np.full((4, 6), 0.25, np.float32),
        grid=grid,
        z_um=0.0,
        propagation_index=1.0,
        wavelength_um=0.674994,
        product_id="t",
        context_id=None,
        position_actual_um={"x": 10.0, "y": 20.0, "z": 0.0},
    )
    np.testing.assert_allclose(tile.field, 0.5 * np.exp(0.5j), rtol=1e-6)
    assert not tile.field.flags.writeable
    with pytest.raises(ValueError):
        tile.field[0, 0] = 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        tile.z_um = 1.0
    with pytest.raises(TypeError):
        tile.quality["x"] = 1
    phase[0, 0] = 9.0  # the caller's array is copied
    assert tile.phase_rad[0, 0] == pytest.approx(0.5)


def test_tile_from_arrays_survives_a_message_round_trip():
    """A product built from arrays sends its context fields (no capture yet)."""
    grid = Grid.centred({"x": 10.0, "y": 20.0, "z": 0.0}, (0.2, 0.1), (4, 6))
    tile = TileProduct.from_arrays(
        phase_rad=np.full((4, 6), 0.5, np.float32),
        intensity_rel=np.full((4, 6), 0.25, np.float32),
        grid=grid,
        z_um=0.0,
        propagation_index=1.0,
        wavelength_um=0.674994,
        product_id="t",
        context_id=None,
        position_actual_um={"x": 10.0, "y": 20.0, "z": 0.0},
        tile_index=3,
        group_id="g",
    )
    header, buffers = tile.to_message()
    back = TileProduct.from_message(header, buffers)
    assert back.grid == tile.grid
    assert back.tile_index == 3 and back.group_id == "g"
    np.testing.assert_allclose(back.field, tile.field, rtol=1e-6)


# ---------------------------------------------------------------- Grid
def test_grid_centred_and_validation():
    grid = Grid.centred({"x": 1200.0, "y": 850.0}, (0.1, 0.2), (11, 21))
    assert grid.origin_um == pytest.approx((850.0 - 0.5, 1200.0 - 2.0))
    assert grid.extent_um == pytest.approx((1.1, 4.2))
    assert Grid.from_dict(grid.to_dict()) == grid
    with pytest.raises(ValidationError):
        Grid((0, 0), (0.0, 0.1), (2, 2))
    with pytest.raises(ValidationError):
        Grid((0, 0), (0.1, 0.1), (0, 2))


# ------------------------------------------------------------ LayerProduct
def test_layer_round_trip_equals_example_header():
    header = _example("LayerProduct")
    buffers = _synthetic_buffers(header["buffers"])
    layer = LayerProduct.from_message(header, buffers)

    assert layer.valid.dtype == bool and layer.phase_rad.dtype == np.float32
    assert layer.grid.shape == (600, 600)
    assert isinstance(layer.tiles[0], TileRecord)
    assert layer.tiles[0].tilt_rad_per_um == (0.0001, -0.0002)
    assert layer.tiles[0].actual_um["x"] == 1200.02

    out, out_buffers = layer.to_message()
    assert out == header
    for (name, data), (_, original) in zip(out_buffers, buffers, strict=True):
        np.testing.assert_array_equal(data, original, err_msg=name)


def test_layer_keeps_unknown_keys_and_optional_field():
    header = _example("LayerProduct")
    header["future_key"] = {"a": 1}
    header["tiles"][0]["future_tile_key"] = 2
    header["buffers"].append(
        {
            "name": "field",
            "dtype": "complex64",
            "shape": [600, 600],
            "byte_order": "little",
        }
    )
    buffers = _synthetic_buffers(header["buffers"])
    layer = LayerProduct.from_message(header, buffers)
    assert layer.field is not None
    out, _ = layer.to_message()
    assert out["future_key"] == {"a": 1}
    assert out["tiles"][0]["future_tile_key"] == 2
    out, out_buffers = layer.to_message(include_field=False)
    assert [name for name, _ in out_buffers] == ["phase_rad", "intensity_rel", "valid"]


def test_layer_rejects_missing_buffer_and_shape_mismatch():
    header = _example("LayerProduct")
    buffers = _synthetic_buffers(header["buffers"])
    short = dict(header, buffers=header["buffers"][:2])
    with pytest.raises(ValidationError, match="Missing buffers"):
        LayerProduct.from_message(short, buffers[:2])
    wrong = copy.deepcopy(header)
    wrong["grid"]["shape"] = [600, 601]
    with pytest.raises(ValidationError, match="does not match grid"):
        LayerProduct.from_message(wrong, buffers)


def test_layer_h5_round_trip(tmp_path):
    h5py = pytest.importorskip("h5py")
    header = _example("LayerProduct")
    buffers = _synthetic_buffers(header["buffers"])
    layer = LayerProduct.from_message(header, buffers)
    path = tmp_path / "experiment.h5"
    with h5py.File(path, "w") as f:
        layer.to_h5(f.create_group("dhm_groups/grp-0007/layer_product"))
    with h5py.File(path, "r") as f:
        again = LayerProduct.from_h5(f["dhm_groups/grp-0007/layer_product"])
    assert again.to_message()[0] == layer.to_message()[0]
    np.testing.assert_array_equal(again.phase_rad, layer.phase_rad)
    np.testing.assert_array_equal(again.valid, layer.valid)


# ---------------------------------------------------------------- bridge
def test_from_product_converts_units_and_axis_order_once():
    header = _example("LayerProduct")
    header["grid"]["pixel_um"] = [0.2, 0.1]  # (y, x), deliberately different
    buffers = _synthetic_buffers(header["buffers"])
    layer = LayerProduct.from_message(header, buffers)

    proc = StructureProcessor.from_product(layer, n_resin=1.52)
    # StructureProcessor: pixel_size = (x, y) in metres.
    assert proc.phase_data.pixel_size == pytest.approx((0.1e-6, 0.2e-6))
    assert proc.wavelength == pytest.approx(0.674994e-6)
    assert proc.n_resin == 1.52
    assert proc.phase_data.data.shape == (600, 600)  # axis 0 stays y


def test_from_product_requires_index():
    header = _example("LayerProduct")
    layer = LayerProduct.from_message(header, _synthetic_buffers(header["buffers"]))
    with pytest.raises(TypeError):
        StructureProcessor.from_product(layer)  # n_resin is keyword-only, required
    with pytest.raises(ValidationError):
        StructureProcessor.from_product(layer, n_resin=float("nan"))
