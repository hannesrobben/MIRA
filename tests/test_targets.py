"""Tests of LayerTarget and its transfer onto the product grid (MI-50)."""

import json
from pathlib import Path

import numpy as np
import pytest

from mira.products import Grid
from mira.targets import LayerTarget, resample_target
from mira.utils.exceptions import ValidationError

CONTRACTS = Path(__file__).parent / "contracts"


def _example():
    return json.loads((CONTRACTS / "LayerTarget.example.json").read_text(encoding="utf-8"))


def _rect_target(pixel=0.1, shape=(40, 60), origin=(0.0, 0.0), box=(10, 30, 20, 40), h=2.0):
    values = np.zeros(shape, np.float32)
    y0, y1, x0, x1 = box
    values[y0:y1, x0:x1] = h
    return LayerTarget(values=values, mask=values > 0, grid=Grid(origin, (pixel, pixel), shape),
                       domain="height", target_id="t", n_structure=1.52, n_surround=1.508)


def test_message_round_trip_of_the_example():
    header = _example()
    rng = np.random.default_rng(0)
    buffers = [("height_um", rng.uniform(0, 3, (1000, 1000)).astype("<f4")),
               ("mask", (rng.uniform(size=(1000, 1000)) > 0.5).astype("u1"))]
    target = LayerTarget.from_message(header, buffers)
    assert target.domain == "height" and target.n_structure == 1.52
    out, out_buffers = target.to_message()
    assert out == header
    np.testing.assert_array_equal(out_buffers[0][1], buffers[0][1])
    np.testing.assert_array_equal(out_buffers[1][1], buffers[1][1])


def test_exact_two_by_two_averaging():
    target = _rect_target()
    # Product pixel 0.2 µm, aligned bins: origin = target origin + 0.05.
    grid = Grid((0.05, 0.05), (0.2, 0.2), (20, 30))
    on = resample_target(target, grid)
    expected = target.values.reshape(20, 2, 30, 2).mean(axis=(1, 3))
    np.testing.assert_allclose(on.values, expected, atol=1e-12)
    np.testing.assert_allclose(on.coverage, 1.0)
    assert on.mask.sum() == 10 * 10  # 20 x 20 target px -> 10 x 10 product px


def test_half_pixel_shift_gives_fractional_edges():
    target = _rect_target()
    grid = Grid((0.05, 0.05), (0.2, 0.2), (20, 30))
    on = resample_target(target, grid, shift_um=(0.0, 0.1))
    # The box moves by half a product pixel in x: the edge columns are half covered.
    row = on.mask_fraction[10]
    # Box x = target columns 20..40 = product columns 10..20, moved by +0.1 µm.
    assert row[10] == pytest.approx(0.5) and row[20] == pytest.approx(0.5)
    assert row[11:20] == pytest.approx(1.0)
    assert on.values[10, 10] == pytest.approx(1.0)
    assert on.matrix_um == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.1]]


def test_integral_is_conserved_and_outside_is_nan():
    target = _rect_target(pixel=0.087, shape=(50, 70), box=(5, 40, 8, 60), h=1.3)
    grid = Grid((-1.0, -1.0), (0.1738, 0.1738), (40, 50))
    on = resample_target(target, grid, shift_um=(0.031, -0.017))
    pixel_area = 0.1738**2
    integral_product = np.nansum(on.values * on.coverage) * pixel_area
    integral_target = target.values.sum(dtype=np.float64) * 0.087**2
    assert integral_product == pytest.approx(integral_target, rel=1e-9)
    assert np.isnan(on.values[0, 0]) and on.coverage[0, 0] == 0
    assert on.substrate.any() and not on.substrate[on.mask].any()


def test_never_upsamples():
    target = _rect_target(pixel=0.3)
    with pytest.raises(ValidationError, match="never upsamples"):
        resample_target(target, Grid((0, 0), (0.2, 0.2), (10, 10)))


def test_validation():
    with pytest.raises(ValidationError, match="domain"):
        LayerTarget(values=np.zeros((2, 2)), mask=np.zeros((2, 2)),
                    grid=Grid((0, 0), (0.1, 0.1), (2, 2)), domain="intensity", target_id="t")
    with pytest.raises(ValidationError, match="does not match grid"):
        LayerTarget(values=np.zeros((2, 3)), mask=np.zeros((2, 2)),
                    grid=Grid((0, 0), (0.1, 0.1), (2, 2)), domain="height", target_id="t")
