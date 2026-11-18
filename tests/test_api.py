"""Tests of mira.api.evaluate_layer (MI-53) against the shared examples."""

import json
from pathlib import Path

import numpy as np
import pytest

from mira.api import evaluate_layer
from mira.deviation import DeviationReport
from mira.forward import height_to_phase_rad
from mira.products import LayerProduct
from mira.targets import LayerTarget
from mira.utils.exceptions import ValidationError

CONTRACTS = Path(__file__).parent / "contracts"
OPTICS = dict(wavelength_um=0.674994, n_structure=1.52, n_surround=1.508, phase_sign=1)


def _example(name):
    return json.loads((CONTRACTS / f"{name}.example.json").read_text(encoding="utf-8"))


def _pair():
    """Product and target built from the vendored examples (same structure)."""
    t_header = _example("LayerTarget")
    p_header = _example("LayerProduct")
    # Target: 0.1 µm raster; square footprint of the example, 1 µm high.
    tg = t_header["grid"]
    y = tg["origin_um"][0] + 0.1 * np.arange(tg["shape"][0])
    x = tg["origin_um"][1] + 0.1 * np.arange(tg["shape"][1])
    inside = ((y[:, None] >= 840) & (y[:, None] < 860)) & ((x[None, :] >= 1190) & (x[None, :] < 1210))
    target = LayerTarget.from_message(
        t_header, [("height_um", np.where(inside, 1.0, 0.0).astype("<f4")),
                   ("mask", inside.astype("u1"))])
    # Product grid of the example (0.1738 µm); measured = expected + piston.
    pg = p_header["grid"]
    py = pg["origin_um"][0] + 0.1738 * np.arange(pg["shape"][0])
    px = pg["origin_um"][1] + 0.1738 * np.arange(pg["shape"][1])
    h = np.where(((py[:, None] >= 840) & (py[:, None] < 860))
                 & ((px[None, :] >= 1190) & (px[None, :] < 1210)), 1.0, 0.0)
    phase = height_to_phase_rad(h, **OPTICS) + 0.4
    product = LayerProduct.from_message(
        p_header, [("phase_rad", phase.astype("<f4")),
                   ("intensity_rel", np.ones(phase.shape, "<f4")),
                   ("valid", np.ones(phase.shape, "u1"))])
    return product, target


def test_report_matches_the_shared_example_structure():
    product, target = _pair()
    report = evaluate_layer(product, target,
                            context={"parameter_set": {"phase_sign": 1},
                                     "source": "nanofactorysystem.feedback",
                                     "report_id": "dev-000017"},
                            settings={"domain": "height", "sigma": 0.02})
    header, buffers = report.to_message()
    example = _example("DeviationReport")
    for key, value in example.items():
        assert key in header, key
        assert type(header[key]) is type(value), key
    for key in example["stats"]:
        assert key in header["stats"], key
    assert header["schema_version"] == example["schema_version"]
    assert header["domain"] == "height" and header["source"] == example["source"]
    assert header["target_id"] == example["target_id"]
    assert header["product_id"] == example["product_id"]
    assert [d["name"] for d in header["buffers"]] == [d["name"] for d in example["buffers"]]
    assert header["grid"] == example["grid"]
    assert header["timings_s"]["evaluate"] > 0
    assert "mira" in header["versions"]
    # Boundary pixels average the edges (0.1738 vs 0.1 µm grid); the plateau matches.
    structure = report.stats["segments"]["structure"]
    assert abs(structure["mean"]) < 0.05
    assert report.stats["levelling"]["offset"] == pytest.approx(
        0.4 * 0.674994 / (2 * np.pi * 0.012), rel=1e-4)
    again = DeviationReport.from_message(header, buffers)
    assert again.to_message()[0] == header


def test_validates_with_dhmconnect_when_available():
    schemas = pytest.importorskip("dhmconnect.schemas")
    product, target = _pair()
    report = evaluate_layer(product, target, context={"phase_sign": 1})
    header, _ = report.to_message()
    schemas.validate(header, "DeviationReport")


def test_phase_domain_by_default_and_optics_recorded():
    product, target = _pair()
    report = evaluate_layer(product, target, context={"phase_sign": 1})
    header, _ = report.to_message()
    assert header["domain"] == "phase"
    assert header["optics"]["n_structure"] == 1.52
    assert header["optics"]["n_surround"] == 1.508


def test_missing_phase_sign_names_the_source():
    product, target = _pair()
    with pytest.raises(ValidationError, match=r"phase_sign \(from context"):
        evaluate_layer(product, target, context={})


def test_unknown_setting_raises():
    product, target = _pair()
    with pytest.raises(ValidationError, match="Unknown settings"):
        evaluate_layer(product, target, context={"phase_sign": 1}, settings={"tolerance": 1})
