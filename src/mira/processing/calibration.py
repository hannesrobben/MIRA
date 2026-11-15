"""
Refractive index (delta_n) calibration module.

Provides segment-based fitting of the refractive index difference delta_n
between resin and its surrounding medium, by comparing a phase measurement
(e.g. in-situ DHM) against an independent height measurement (e.g. LSM,
Sensofar) of the same, identically segmented structure.

The fit relies on the same linear relationship already used throughout
`phase_retrieval.py`:

    phase = 2 * pi * delta_n * height / wavelength

This is proportional with no additive offset, so delta_n can be recovered
from a set of paired (phase, height) samples via a least-squares fit
through the origin - the same closed-form solution historically used for
scalar scaling-factor fits (`c = <phase, height> / <height, height>`),
reinterpreted physically: `delta_n = c * wavelength / (2 * pi)`.

New fitting strategies register themselves via `register_delta_n_method`
instead of being added as branches inside `calculate_delta_n` - this is
the extension point for future methods (e.g. weighted by segment std,
robust/RANSAC fitting, or a pixel-wise instead of segment-wise fit).
"""

from __future__ import annotations
from dataclasses import dataclass

from collections.abc import Callable

import numpy as np

from ..core.structure import Structure
from ..utils.exceptions import ProcessingError, ValidationError
from ..analysis.segment_data import (
    BACKGROUND_SEGMENT_ID,
    SegmentStatistic,
    extract_segment_statistics,
    make_relative,
    paired_segment_values,
)

# Sentinel id used for the background "segment" in results below, chosen to
# never collide with a real segment id (segment ids start at 0).
BACKGROUND_SEGMENT_ID = -1


@dataclass
class DeltaNResult:
    """
    Result of a delta_n (refractive index difference) fit.

    Attributes:
        delta_n:         Fitted refractive index difference.
        method:          Name of the fitting method used.
        r_squared:       Coefficient of determination of the fit.
        residual_std:    Standard deviation of the fit residuals.
        n_segments:      Number of (segment + background, if included)
                         data points used in the fit.
        edge_offset_px:  Pixel margin excluded from every segment/
                         background edge before computing statistics.
        phase_stats:     Per-segment statistics of the phase channel, in
                         ascending segment-id order (background first, if
                         included).
        height_stats:    Per-segment statistics of the height channel, in
                         the same order as `phase_stats`.
    """

    delta_n: float
    method: str
    r_squared: float
    residual_std: float
    n_segments: int
    edge_offset_px: int
    phase_stats: list[SegmentStatistic]
    height_stats: list[SegmentStatistic]


_DELTA_N_METHODS: dict[str, Callable[..., DeltaNResult]] = {}


def register_delta_n_method(name: str):
    """
    Register a delta_n fitting method under `name`, making it available as
    `calculate_delta_n(..., method=name)`.

    This is the extension point for adding new fitting strategies without
    touching `calculate_delta_n` itself.
    """

    def decorator(func: Callable[..., DeltaNResult]) -> Callable[..., DeltaNResult]:
        _DELTA_N_METHODS[name] = func
        return func

    return decorator


def available_delta_n_methods() -> list[str]:
    """Return the names of all currently registered delta_n fitting methods."""
    return sorted(_DELTA_N_METHODS)


@register_delta_n_method("segment_mean")
def _fit_segment_mean(
    phase_structure: Structure,
    height_structure: Structure,
    wavelength: float,
    offset_pixels: int,
    include_background: bool,
) -> DeltaNResult:
    """
    Assumes each segment (and the background) has a near-constant phase/
    height value, up to measurement noise concentrated near segmentation
    edges. Fits delta_n from the background-relative segment means via a
    least-squares line through the origin.
    """
    phase_vals, height_vals, phase_stats, height_stats = paired_segment_values(
        phase_structure,
        height_structure,
        offset_pixels=offset_pixels,
        include_background=include_background,
        relative=True,
    )

    denom = float(np.dot(height_vals, height_vals))
    if denom == 0:
        raise ProcessingError(
            "All height values are identical after background subtraction "
            "- there is no height variation to fit delta_n against."
        )

    c = float(np.dot(phase_vals, height_vals) / denom)
    delta_n = c * wavelength / (2 * np.pi)

    predicted = c * height_vals
    residuals = phase_vals - predicted
    residual_std = float(np.std(residuals))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((phase_vals - np.mean(phase_vals)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return DeltaNResult(
        delta_n=float(delta_n),
        method="segment_mean",
        r_squared=r_squared,
        residual_std=residual_std,
        n_segments=len(phase_stats),
        edge_offset_px=offset_pixels,
        phase_stats=phase_stats,
        height_stats=height_stats,
    )


def calculate_delta_n(
    phase_structure: Structure,
    height_structure: Structure,
    wavelength: float,
    method: str = "segment_mean",
    offset_pixels: int = 10,
    include_background: bool = True,
    **kwargs,
) -> DeltaNResult:
    """
    Fit the refractive index difference (delta_n) between a phase
    measurement and an independent height measurement of the same,
    identically segmented structure.

    Args:
        phase_structure:     Segmented Structure holding phase data
                             (radians).
        height_structure:    Segmented Structure holding height data
                             (meters), segmented identically (same number
                             and ordering of segments) as phase_structure.
        wavelength:          Light wavelength in meters.
        method:              Registered fitting method name. Currently
                             available: "segment_mean" - see
                             `available_delta_n_methods()`.
        offset_pixels:       Pixels excluded from every segment/background
                             edge before computing statistics, to reduce
                             the influence of imprecise segmentation
                             boundaries.
        include_background:  If True, the background is included as an
                             additional (relative-zero) data point in the
                             fit.
        **kwargs:            Forwarded to the selected method.

    Returns:
        DeltaNResult with the fitted delta_n and fit-quality metrics.

    Raises:
        ValidationError: If `method` is unknown, either structure is
            unsegmented, or their segment counts do not match.
        ProcessingError: If the fit itself fails (e.g. no height variation).
    """
    if method not in _DELTA_N_METHODS:
        raise ValidationError(
            f"Unknown delta_n method: '{method}'. Available: "
            f"{available_delta_n_methods()}"
        )
    return _DELTA_N_METHODS[method](
        phase_structure,
        height_structure,
        wavelength,
        offset_pixels,
        include_background,
        **kwargs,
    )
