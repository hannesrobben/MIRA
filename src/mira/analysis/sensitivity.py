"""
Sensitivity analysis for the refractive index difference (delta_n) used in
phase-to-height conversion.

`convert_phase_to_height` / `convert_height_to_phase`
(processing/phase_retrieval.py) only depend on n_resin and n_air through
their difference, delta_n = n_resin - n_air - the two individual values
never enter the conversion formula separately. Sweeping n_resin and n_air
independently over a 2D grid would therefore be misleading: many
(n_resin, n_air) pairs with the same difference give an identical
predicted height, so a naive grid sweep would just re-run the same fit
many times under different labels and make the sensitivity look larger
(more grid points) than it physically is.

Instead, this module:
  1. Propagates the *individual* uncertainties you have for n_resin and
     n_air (e.g. from a datasheet, or a prior measurement series) into a
     single uncertainty on delta_n, via standard error propagation for a
     difference of two independent quantities:
         sigma_delta_n = sqrt(sigma_n_resin^2 + sigma_n_air^2)
  2. Sweeps delta_n over a range derived from that propagated uncertainty
     (nominal +/- n_sigma * sigma_delta_n),
  3. Re-evaluates the segment-based fit quality (RMSE, R^2) at each swept
     delta_n, using the same segment statistics
     `processing.calibration.calculate_delta_n` itself fits against - so
     the result traces out the fit-quality curve around the already-fitted
     optimum, translated back into physically meaningful n_resin/n_air
     uncertainty.

This gives a defensible basis for an uncertainty statement in the paper
("delta_n = 0.0123 +/- 0.0004, propagated from n_resin = 1.520 +/- 0.003
and n_air = 1.000 +/- 0.0002") instead of reporting the fitted delta_n
with no uncertainty at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.structure import Structure
from ..utils.exceptions import ValidationError
from .segment_data import paired_segment_values


@dataclass
class SensitivityResult:
    """
    Result of a delta_n sensitivity sweep.

    Attributes:
        delta_n_values:       Swept delta_n values.
        rmse_values:          RMSE (phase residual, same units as the
                              phase data) at each swept delta_n.
        r_squared_values:      R^2 of the implied fit at each swept
                              delta_n (how well a *fixed* delta_n explains
                              the segment data, vs. the best-fit delta_n).
        delta_n_nominal:       Nominal (center) delta_n the sweep was built
                              around - typically the already-fitted value
                              from `calculate_delta_n`.
        delta_n_uncertainty:   Propagated 1-sigma uncertainty on delta_n,
                              from n_resin_uncertainty and
                              n_air_uncertainty.
        n_resin_uncertainty:    The n_resin uncertainty that was propagated.
        n_air_uncertainty:      The n_air uncertainty that was propagated.
        best_fit_delta_n:      The delta_n within the swept range with the
                              lowest RMSE - compare against
                              `delta_n_nominal` as a sanity check (should
                              be close if the original fit converged
                              properly).
        rmse_at_nominal:       RMSE at `delta_n_nominal` itself.
        rmse_at_bounds:         (rmse_low, rmse_high) - RMSE at
                              `delta_n_nominal -/+ delta_n_uncertainty`,
                              i.e. how much the fit degrades at the edge
                              of the physically plausible range.
    """

    delta_n_values: np.ndarray
    rmse_values: np.ndarray
    r_squared_values: np.ndarray
    delta_n_nominal: float
    delta_n_uncertainty: float
    n_resin_uncertainty: float
    n_air_uncertainty: float
    best_fit_delta_n: float
    rmse_at_nominal: float
    rmse_at_bounds: tuple[float, float]


def propagate_delta_n_uncertainty(
    n_resin_uncertainty: float, n_air_uncertainty: float
) -> float:
    """
    Propagate independent 1-sigma uncertainties on n_resin and n_air into a
    1-sigma uncertainty on delta_n = n_resin - n_air, via standard
    quadrature error propagation for a difference of independent
    quantities.

    Args:
        n_resin_uncertainty: 1-sigma uncertainty on n_resin.
        n_air_uncertainty:   1-sigma uncertainty on n_air.

    Returns:
        Propagated 1-sigma uncertainty on delta_n.

    Raises:
        ValidationError: If either uncertainty is negative.
    """
    if n_resin_uncertainty < 0 or n_air_uncertainty < 0:
        raise ValidationError("Uncertainties must be non-negative.")
    return float(np.sqrt(n_resin_uncertainty**2 + n_air_uncertainty**2))


def _rmse_and_r_squared_at_delta_n(
    phase_vals: np.ndarray, height_vals: np.ndarray, wavelength: float, delta_n: float
) -> tuple[float, float]:
    """
    RMSE and R^2 of the phase-vs-height segment data at a FIXED delta_n
    (no fitting - delta_n is given, not solved for), using the same
    proportional-through-the-origin model as
    `processing.calibration._fit_segment_mean`.
    """
    c = 2 * np.pi * delta_n / wavelength
    predicted = c * height_vals
    residuals = phase_vals - predicted
    rmse = float(np.sqrt(np.mean(residuals**2)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((phase_vals - np.mean(phase_vals)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return rmse, r_squared


def sweep_delta_n_sensitivity(
    phase_structure: Structure,
    height_structure: Structure,
    wavelength: float,
    delta_n_nominal: float,
    n_resin_uncertainty: float,
    n_air_uncertainty: float,
    n_sigma: float = 3.0,
    n_points: int = 41,
    offset_pixels: int = 10,
    include_background: bool = True,
) -> SensitivityResult:
    """
    Sweep delta_n over a range derived from the propagated uncertainty of
    n_resin/n_air, and evaluate the segment-based fit quality (RMSE, R^2)
    at each value.

    Args:
        phase_structure:       Segmented Structure holding phase data, the
                               same one passed to
                               `processing.calibration.calculate_delta_n`.
        height_structure:      Segmented Structure holding height data,
                               segmented identically to phase_structure.
        wavelength:             Light wavelength in meters.
        delta_n_nominal:        Center of the sweep - typically the
                               already-fitted delta_n from
                               `calculate_delta_n`.
        n_resin_uncertainty:    1-sigma uncertainty on n_resin (e.g. from a
                               datasheet or a prior measurement series).
        n_air_uncertainty:      1-sigma uncertainty on n_air.
        n_sigma:                 How many propagated-sigma to sweep out to
                               on each side of `delta_n_nominal`.
        n_points:                Number of points in the sweep.
        offset_pixels:           Pixels excluded from every segment/
                               background edge (same meaning as in
                               `calculate_delta_n`).
        include_background:      If True, include the background as an
                               additional data point (same meaning as in
                               `calculate_delta_n`).

    Returns:
        SensitivityResult.

    Raises:
        ValidationError: If either structure is unsegmented, their segment
            counts don't match (propagated from `paired_segment_values`),
            or either uncertainty is negative.
    """
    delta_n_uncertainty = propagate_delta_n_uncertainty(
        n_resin_uncertainty, n_air_uncertainty
    )

    phase_vals, height_vals, _, _ = paired_segment_values(
        phase_structure,
        height_structure,
        offset_pixels=offset_pixels,
        include_background=include_background,
        relative=True,
    )

    half_range = n_sigma * delta_n_uncertainty
    delta_n_values = np.linspace(
        delta_n_nominal - half_range, delta_n_nominal + half_range, n_points
    )

    rmse_values = np.empty(n_points)
    r_squared_values = np.empty(n_points)
    for i, dn in enumerate(delta_n_values):
        rmse_values[i], r_squared_values[i] = _rmse_and_r_squared_at_delta_n(
            phase_vals, height_vals, wavelength, dn
        )

    best_idx = int(np.argmin(rmse_values))
    rmse_at_nominal, _ = _rmse_and_r_squared_at_delta_n(
        phase_vals, height_vals, wavelength, delta_n_nominal
    )
    rmse_low, _ = _rmse_and_r_squared_at_delta_n(
        phase_vals, height_vals, wavelength, delta_n_nominal - delta_n_uncertainty
    )
    rmse_high, _ = _rmse_and_r_squared_at_delta_n(
        phase_vals, height_vals, wavelength, delta_n_nominal + delta_n_uncertainty
    )

    return SensitivityResult(
        delta_n_values=delta_n_values,
        rmse_values=rmse_values,
        r_squared_values=r_squared_values,
        delta_n_nominal=float(delta_n_nominal),
        delta_n_uncertainty=delta_n_uncertainty,
        n_resin_uncertainty=float(n_resin_uncertainty),
        n_air_uncertainty=float(n_air_uncertainty),
        best_fit_delta_n=float(delta_n_values[best_idx]),
        rmse_at_nominal=rmse_at_nominal,
        rmse_at_bounds=(rmse_low, rmse_high),
    )
