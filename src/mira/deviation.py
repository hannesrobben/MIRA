"""Comparison of a measured layer with its target → ``DeviationReport`` (MI-52).

``compare`` transfers the target onto the product grid (MI-50), converts both
into the comparison domain (MI-51), levels the measurement on the substrate
(piston between product and target, SA5: MIRA levels) and reports:

* the deviation map ``measured − expected`` (NaN outside the valid pixels);
* RMSE, MAE, standard deviation, mean, number and share of valid pixels;
* statistics per segment (structure / substrate, or caller-given labels);
* with a noise level ``sigma``: the share of the structure outside ±k·sigma,
  split into too low / too high;
* validity flags: registration confidence, coverage of the target, product
  quality.

Acceptance thresholds are not set here: they wait for M7 (deferred).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import numpy as np

from .forward import Optics, choose_domain, to_domain
from .products import (
    Grid,
    LayerProduct,
    _check_message,
    _decode_buffers,
    _descriptor,
    _freeze,
    _thaw,
    _to_jsonable,
    _utc_now,
    _wire,
)
from .targets import LayerTarget, TargetOnGrid, resample_target
from .utils.exceptions import ValidationError

__all__ = ["DEVIATION_REPORT_SCHEMA_VERSION", "DeviationReport", "compare"]

DEVIATION_REPORT_SCHEMA_VERSION = "1.0"
_DEVIATION_BUFFER = {"height": "deviation_um", "phase": "deviation_rad"}
_REPORT_KEYS = ("report_id", "target_id", "product_id", "domain", "stats", "registration",
                "source", "versions", "warnings", "created_utc")


@dataclass(frozen=True)
class DeviationReport:
    """Result of a comparison (contract ``DeviationReport`` 1.0).

    Parameters
    ----------
    deviation : numpy.ndarray
        ``measured − expected`` in the domain unit (µm or rad), float32, NaN
        outside ``valid``.
    valid : numpy.ndarray (bool)
        Pixels that entered the statistics.
    grid : Grid
        Product grid.
    domain : {"height", "phase"}
    report_id, target_id, product_id : str
    stats : mapping
        ``rmse``, ``mae``, ``sigma``, ``mean``, ``n_valid``,
        ``valid_fraction`` and the additional blocks described in
        :func:`compare`.
    registration : mapping
        ``matrix`` (2×3 target → product in µm), ``method``, ``residual``.
    source : str, optional
        Producer tag (nanofactorysystem stores ``"nanofactorysystem.feedback"``).
    versions : mapping, optional
    warnings : sequence of str, optional
    created_utc : str or None, optional
    extra : mapping, optional
        Unknown header keys (e.g. ``timings_s``).
    """

    deviation: np.ndarray
    valid: np.ndarray
    grid: Grid
    domain: str
    report_id: str
    target_id: str
    product_id: str
    stats: Mapping[str, Any]
    registration: Mapping[str, Any]
    source: str | None = None
    versions: Mapping[str, Any] = dc_field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    created_utc: str | None = None
    extra: Mapping[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.domain not in _DEVIATION_BUFFER:
            raise ValidationError(f"Unknown domain {self.domain!r}.")
        deviation = np.array(self.deviation, dtype=np.float32)
        valid = np.array(self.valid, dtype=bool)
        for name, array in (("deviation", deviation), ("valid", valid)):
            if array.shape != self.grid.shape:
                raise ValidationError(f"{name} shape {array.shape} != grid {self.grid.shape}.")
            array.flags.writeable = False
        object.__setattr__(self, "deviation", deviation)
        object.__setattr__(self, "valid", valid)
        for name in ("stats", "registration", "versions", "extra"):
            object.__setattr__(self, name, _freeze(dict(getattr(self, name) or {})))
        object.__setattr__(self, "warnings", tuple(str(w) for w in self.warnings or ()))

    def to_message(self) -> tuple[dict, list[tuple[str, np.ndarray]]]:
        """Serialise as a ``DeviationReport`` 1.0 message.

        Returns
        -------
        header : dict
        buffers : list of (str, numpy.ndarray)
            ``deviation_um`` or ``deviation_rad`` (float32) and ``valid`` (uint8).
        """
        buffers = [
            (_DEVIATION_BUFFER[self.domain], np.ascontiguousarray(self.deviation, dtype="<f4")),
            ("valid", _wire("valid", self.valid)),
        ]
        header = _thaw(self.extra)
        header.update({
            "message_type": "DeviationReport",
            "schema_version": DEVIATION_REPORT_SCHEMA_VERSION,
            "report_id": self.report_id,
            "target_id": self.target_id,
            "product_id": self.product_id,
            "domain": self.domain,
            "grid": self.grid.to_dict(),
            "stats": _thaw(self.stats),
            "registration": _thaw(self.registration),
            "source": self.source,
            "versions": _thaw(self.versions),
            "warnings": list(self.warnings),
            "buffers": [_descriptor(name, data) for name, data in buffers],
            "created_utc": self.created_utc or _utc_now(),
        })
        if header["source"] is None:
            del header["source"]
        return _to_jsonable(header), buffers

    @classmethod
    def from_message(cls, header: Mapping, buffers: Sequence) -> DeviationReport:
        """Build a report from a ``DeviationReport`` 1.0 message."""
        _check_message(header, "DeviationReport", DEVIATION_REPORT_SCHEMA_VERSION)
        domain = header.get("domain")
        if domain not in _DEVIATION_BUFFER:
            raise ValidationError(f"Unknown domain {domain!r}.")
        renamed = [
            dict(d, name="phase_rad") if d["name"] == _DEVIATION_BUFFER[domain] else d
            for d in header.get("buffers", [])
        ]
        payloads = [p[1] if isinstance(p, tuple) else p for p in buffers]
        arrays = _decode_buffers(renamed, payloads)
        if "phase_rad" not in arrays or "valid" not in arrays:
            raise ValidationError("DeviationReport needs the deviation and valid buffers.")
        known = {k: header[k] for k in _REPORT_KEYS if k in header}
        extra = {k: v for k, v in header.items()
                 if k not in _REPORT_KEYS and k not in {"message_type", "schema_version",
                                                        "grid", "buffers"}}
        return cls(deviation=arrays["phase_rad"], valid=arrays["valid"].astype(bool),
                   grid=Grid.from_dict(header["grid"]), extra=extra, **known)


def _stats(values: np.ndarray) -> dict:
    n = int(values.size)
    if n == 0:
        return {"rmse": None, "mae": None, "sigma": None, "mean": None, "n_valid": 0}
    return {
        "rmse": float(np.sqrt(np.mean(values**2))),
        "mae": float(np.mean(np.abs(values))),
        "sigma": float(np.std(values)),
        "mean": float(np.mean(values)),
        "n_valid": n,
    }


def compare(
    product: LayerProduct,
    target: LayerTarget | TargetOnGrid,
    *,
    optics: Optics | None = None,
    sigma: float | None = None,
    k: float = 3.0,
    domain: str | None = None,
    shift_um: tuple[float, float] = (0.0, 0.0),
    segments: np.ndarray | None = None,
    registration: Mapping[str, Any] | None = None,
    report_id: str = "deviation",
    min_coverage: float = 0.9,
) -> DeviationReport:
    """Compare a layer product with its target.

    Parameters
    ----------
    product : LayerProduct
        Measured mosaic; its unwrapped phase and validity mask are used.
    target : LayerTarget or TargetOnGrid
        Target raster; a :class:`LayerTarget` is first transferred onto the
        product grid with ``shift_um`` (MI-50).
    optics : Optics or None, optional
        Needed whenever a conversion between phase and height is required
        (MI-51); no defaults.
    sigma : float or None, optional
        Noise level in the comparison domain for the ±k·sigma classification;
        None leaves the classification empty with a warning (thresholds wait
        for M7).
    k : float, optional
        Multiple of ``sigma``.
    domain : {"height", "phase"} or None, optional
        Fixed comparison domain; None chooses automatically (M5 = OM4).
    shift_um : tuple of float, optional
        Pose shift ``(y, x)`` of the target (ignored for a ``TargetOnGrid``).
    segments : numpy.ndarray of int, optional
        Label image on the product grid (0 = ignore); default: structure
        (label ``"structure"``) and substrate (``"substrate"``).
    registration : mapping, optional
        Registration that placed the target: ``method``, ``residual`` and
        ``confidence`` (0 … 1). Default: stage prior, confidence None.
    report_id : str, optional
    min_coverage : float, optional
        Share of the target structure that must be valid in the product for
        ``stats["validity"]["coverage_ok"]``.

    Returns
    -------
    DeviationReport
        ``stats`` holds the contract keys plus ``levelling`` (offset removed,
        reference), ``segments``, ``classification`` and ``validity``.
    """
    warnings: list[str] = []
    if isinstance(target, LayerTarget):
        target_id = target.target_id
        target_on = resample_target(target, product.grid, shift_um=shift_um)
    elif isinstance(target, TargetOnGrid):
        target_id = "unknown"
        target_on = target
    else:
        raise ValidationError("target must be a LayerTarget or a TargetOnGrid.")
    if target_on.grid != product.grid:
        raise ValidationError("The target was transferred onto a different grid.")
    chosen = choose_domain(target_on.domain, optics_known=optics is not None, fixed=domain)

    measured = to_domain(np.asarray(product.phase_rad, dtype=np.float64), "phase", chosen, optics)
    expected = to_domain(target_on.values, target_on.domain, chosen, optics)
    valid = (
        np.asarray(product.valid, dtype=bool)
        & (target_on.coverage > 0.999)
        & np.isfinite(measured)
        & np.isfinite(expected)
    )
    raw = measured - expected
    substrate = valid & target_on.substrate
    if substrate.any():
        offset = float(np.median(raw[substrate]))
        level_ref = "substrate"
    elif valid.any():
        offset = float(np.median(raw[valid]))
        level_ref = "all_valid"
        warnings.append("no valid substrate pixels: levelled on all valid pixels")
    else:
        offset = 0.0
        level_ref = "none"
        warnings.append("no valid pixels")
    deviation = np.where(valid, raw - offset, np.nan)

    stats = _stats(deviation[valid])
    stats["valid_fraction"] = float(valid.mean())
    stats["levelling"] = {"offset": offset, "reference": level_ref}

    structure = valid & target_on.mask
    if segments is None:
        stats["segments"] = {
            "structure": _stats(deviation[structure]),
            "substrate": _stats(deviation[substrate]),
        }
    else:
        labels = np.asarray(segments)
        if labels.shape != product.grid.shape:
            raise ValidationError("segments must have the shape of the product grid.")
        stats["segments"] = {
            str(int(label)): _stats(deviation[valid & (labels == label)])
            for label in np.unique(labels) if label != 0
        }

    if sigma is None or not (sigma > 0 and math.isfinite(sigma)):
        stats["classification"] = None
        warnings.append("no noise level sigma: k*sigma classification skipped (M7)")
    else:
        values = deviation[structure]
        n = max(int(values.size), 1)
        low = int(np.count_nonzero(values < -k * sigma))
        high = int(np.count_nonzero(values > k * sigma))
        stats["classification"] = {
            "sigma": float(sigma), "k": float(k),
            "too_low_fraction": low / n, "too_high_fraction": high / n,
            "outside_fraction": (low + high) / n,
        }

    registration = dict(registration or {})
    target_pixels = int(np.count_nonzero(target_on.mask))
    coverage = int(np.count_nonzero(structure)) / target_pixels if target_pixels else 0.0
    quality = dict(product.quality)
    stats["validity"] = {
        "registration_confidence": registration.get("confidence"),
        "target_coverage": coverage,
        "coverage_ok": coverage >= min_coverage,
        "product_flagged_tiles": quality.get("n_flagged"),
        "product_warnings": len(product.warnings),
    }
    if coverage < min_coverage:
        warnings.append(f"target structure only {coverage:.0%} covered by valid pixels")
    return DeviationReport(
        deviation=deviation,
        valid=valid,
        grid=product.grid,
        domain=chosen,
        report_id=report_id,
        target_id=target_id,
        product_id=product.product_id,
        stats=stats,
        registration={
            "matrix": target_on.matrix_um,
            "method": registration.get("method", "stage_prior"),
            "residual": registration.get("residual"),
            **({"confidence": registration["confidence"]} if "confidence" in registration else {}),
        },
        warnings=tuple(warnings),
    )
