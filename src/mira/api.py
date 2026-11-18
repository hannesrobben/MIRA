"""Stable entry points of MIRA for the other packages (MI-53).

nanofactorysystem T90 calls :func:`evaluate_layer` on the Work-PC (G6) for
every layer capture in the operating modes ``monitor`` / ``shadow`` /
``correct``. The signature and the report format are kept stable; new
options go into ``settings``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from importlib import metadata
from typing import Any

from .deviation import DeviationReport, compare
from .forward import Optics
from .products import LayerProduct
from .targets import LayerTarget
from .utils.exceptions import ValidationError

__all__ = ["evaluate_layer", "SETTINGS_KEYS"]

#: Keys understood in ``settings`` of :func:`evaluate_layer`.
SETTINGS_KEYS = ("domain", "sigma", "k", "shift_um", "min_coverage", "segments")


def _version() -> str:
    try:
        return metadata.version("mira")
    except metadata.PackageNotFoundError:  # pragma: no cover - source checkout
        from . import __version__

        return __version__


def _resolve_optics(
    product: LayerProduct, target: LayerTarget, context: Mapping[str, Any]
) -> tuple[Optics | None, str | None]:
    parameter_set = context.get("parameter_set") or {}
    candidates = {
        "wavelength_um": (product.wavelength_um, "LayerProduct.wavelength_um"),
        "n_structure": (
            target.n_structure if target.n_structure is not None else context.get("n_structure"),
            "LayerTarget.n_structure or context['n_structure']",
        ),
        "n_surround": (
            product.n_surround if product.n_surround is not None else target.n_surround,
            "LayerProduct.n_surround or LayerTarget.n_surround",
        ),
        "phase_sign": (
            parameter_set.get("phase_sign", context.get("phase_sign")),
            "context['parameter_set']['phase_sign'] or context['phase_sign']",
        ),
    }
    try:
        optics = Optics.from_sources(
            **{k: v for k, (v, _) in candidates.items()},
            sources={k: s for k, (_, s) in candidates.items()},
        )
    except ValidationError as exc:
        return None, str(exc)
    return optics, None


def evaluate_layer(
    product: LayerProduct,
    target: LayerTarget,
    context: Mapping[str, Any] | None = None,
    settings: Mapping[str, Any] | None = None,
) -> DeviationReport:
    """Compare one measured layer with its target and return the report.

    Parameters
    ----------
    product : LayerProduct
        Stitched / fused product of the structure and layer (from the DHM
        service, ``LayerProduct.from_message`` or ``from_h5``).
    target : LayerTarget
        Expected geometry after this layer (nanofactorysystem T89).
    context : mapping, optional
        ``parameter_set`` (dict with at least ``phase_sign``) or
        ``phase_sign``; ``n_structure`` if the target has none;
        ``registration`` (``method``, ``residual``, ``confidence``) of the pose
        estimate; ``report_id``; ``source`` (stored in the report, e.g.
        ``"nanofactorysystem.feedback"``).
    settings : mapping, optional
        ``domain`` (``"phase"`` / ``"height"`` / None = automatic, M5 = OM4),
        ``sigma`` (noise level in the domain unit), ``k`` (default 3),
        ``shift_um`` (pose shift ``(y, x)``), ``min_coverage`` (default 0.9),
        ``segments`` (label image). Unknown keys raise.

    Returns
    -------
    DeviationReport
        Header with ``schema_version``, ``domain``, ``timings_s.evaluate``
        (computing time in s) and ``versions.mira``.

    Raises
    ------
    ValidationError
        Unknown settings, missing optics where a conversion is needed (the
        message names every missing value and its source), or invalid inputs.
    """
    start = time.perf_counter()
    context = dict(context or {})
    settings = dict(settings or {})
    unknown = set(settings) - set(SETTINGS_KEYS)
    if unknown:
        raise ValidationError(f"Unknown settings: {sorted(unknown)} (known: {SETTINGS_KEYS}).")
    if not isinstance(product, LayerProduct) or not isinstance(target, LayerTarget):
        raise ValidationError("evaluate_layer needs a LayerProduct and a LayerTarget.")

    optics, missing = _resolve_optics(product, target, context)
    domain = settings.get("domain")
    needs_optics = target.domain == "height" or domain == "height"
    if optics is None and needs_optics:
        raise ValidationError(missing)

    report = compare(
        product,
        target,
        optics=optics,
        sigma=settings.get("sigma"),
        k=float(settings.get("k", 3.0)),
        domain=domain,
        shift_um=tuple(settings.get("shift_um", (0.0, 0.0))),
        segments=settings.get("segments"),
        registration=context.get("registration"),
        report_id=str(context.get("report_id", f"dev-{product.product_id}")),
        min_coverage=float(settings.get("min_coverage", 0.9)),
    )
    elapsed = time.perf_counter() - start
    extra = dict(report.extra)
    extra["timings_s"] = {"evaluate": elapsed}
    if optics is not None:
        extra["optics"] = {
            "wavelength_um": optics.wavelength_um,
            "n_structure": optics.n_structure,
            "n_surround": optics.n_surround,
            "phase_sign": optics.phase_sign,
        }
    return DeviationReport(
        deviation=report.deviation,
        valid=report.valid,
        grid=report.grid,
        domain=report.domain,
        report_id=report.report_id,
        target_id=report.target_id,
        product_id=report.product_id,
        stats=report.stats,
        registration=report.registration,
        source=context.get("source"),
        versions={"mira": _version()},
        warnings=report.warnings,
        extra=extra,
    )
