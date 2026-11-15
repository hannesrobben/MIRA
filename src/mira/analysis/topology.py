"""
Error-vs-surface-topology analysis.

Answers: is a prediction error random (roughly uniform across the
surface, i.e. a noise floor) or systematic (concentrated where the
surface is steep or curved, e.g. at structure edges/flanks)? That
distinguishes a measurement noise floor from a genuine model or
reconstruction limitation that shows up specifically at high-gradient/
high-curvature regions.

The approach: compute the local gradient magnitude (slope) and curvature
(Laplacian) of a reference surface, then correlate the absolute error
against both - as a single Pearson correlation coefficient, and as a
binned mean-error-per-gradient/curvature-bin curve, which shows directly
whether error grows with slope/curvature rather than just whether the two
are loosely associated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.ma as ma

from ..utils.exceptions import ValidationError


@dataclass
class ErrorTopologyResult:
    """
    Result of correlating an error map against a reference surface's local
    gradient magnitude and curvature.

    Attributes:
        gradient_correlation:      Pearson r between |error| and gradient
                                   magnitude. NaN if the gradient has no
                                   variation.
        curvature_correlation:     Pearson r between |error| and
                                   |curvature|. NaN if the curvature has no
                                   variation.
        gradient_bin_centers:      Bin-center gradient-magnitude values
                                   used for the binned summary.
        gradient_bin_mean_error:    Mean |error| within each gradient bin
                                   (NaN for empty bins).
        gradient_bin_counts:       Number of pixels in each gradient bin.
        curvature_bin_centers:     Bin-center |curvature| values.
        curvature_bin_mean_error:  Mean |error| within each curvature bin
                                   (NaN for empty bins).
        curvature_bin_counts:      Number of pixels in each curvature bin.
        n_pixels:                  Number of valid (finite, unmasked)
                                   pixels used.
    """

    gradient_correlation: float
    curvature_correlation: float
    gradient_bin_centers: np.ndarray
    gradient_bin_mean_error: np.ndarray
    gradient_bin_counts: np.ndarray
    curvature_bin_centers: np.ndarray
    curvature_bin_mean_error: np.ndarray
    curvature_bin_counts: np.ndarray
    n_pixels: int


def _extract_array(value):
    """
    Accept an ImageData-like object (`.data`), a raw ndarray, or a masked
    array.

    ndarray/MaskedArray are returned as-is - MaskedArray also happens to
    expose a `.data` attribute (the underlying values *without* the mask),
    so checking `hasattr(value, "data")` first would silently strip the
    mask from any masked array passed in directly.
    """
    if isinstance(value, (np.ndarray, ma.MaskedArray)):
        return value
    return value.data if hasattr(value, "data") else value


def _raw(arr) -> np.ndarray:
    """Underlying plain ndarray of a possibly-masked array."""
    return arr.data if isinstance(arr, ma.MaskedArray) else np.asarray(arr)


def compute_gradient_magnitude(data, pixel_size: tuple[float, float]) -> np.ndarray:
    """
    Local surface gradient magnitude |grad(data)|, in data units per meter.

    Args:
        data:        2D height/phase array (ImageData-like, ndarray, or
                     masked array).
        pixel_size:  (dy, dx) physical pixel size in meters.

    Returns:
        2D array of gradient magnitudes, same shape as `data`.
    """
    raw = _raw(_extract_array(data))
    dy, dx = pixel_size
    grad_y, grad_x = np.gradient(raw, dy, dx)
    return np.sqrt(grad_x**2 + grad_y**2)


def compute_curvature(data, pixel_size: tuple[float, float]) -> np.ndarray:
    """
    Local surface curvature (Laplacian, d2/dx2 + d2/dy2) of `data`, in
    data units per meter^2.

    Args:
        data:        2D height/phase array (ImageData-like, ndarray, or
                     masked array).
        pixel_size:  (dy, dx) physical pixel size in meters.

    Returns:
        2D array of curvature values, same shape as `data`.
    """
    raw = _raw(_extract_array(data))
    dy, dx = pixel_size
    grad_y, grad_x = np.gradient(raw, dy, dx)
    _, d2x = np.gradient(grad_x, dy, dx)
    d2y, _ = np.gradient(grad_y, dy, dx)
    return d2x + d2y


def _valid_mask(*arrays) -> np.ndarray:
    """Combine finite + (if masked) unmasked pixels across all inputs."""
    mask = np.ones(np.asarray(arrays[0]).shape, dtype=bool)
    for arr in arrays:
        if isinstance(arr, ma.MaskedArray):
            mask &= ~ma.getmaskarray(arr)
            mask &= np.isfinite(arr.data)
        else:
            mask &= np.isfinite(arr)
    return mask


def _binned_mean(x: np.ndarray, y: np.ndarray, n_bins: int):
    """Mean of y within n_bins equal-width bins of x. Returns (centers, means, counts)."""
    edges = np.linspace(np.min(x), np.max(x), n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    bin_idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    means = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        sel = bin_idx == i
        counts[i] = int(np.sum(sel))
        if counts[i] > 0:
            means[i] = float(np.mean(y[sel]))
    return centers, means, counts


def analyze_error_vs_topology(
    error,
    reference,
    pixel_size: tuple[float, float],
    n_bins: int = 10,
) -> ErrorTopologyResult:
    """
    Correlate an absolute error map against the local gradient magnitude
    and curvature of a reference surface.

    Args:
        error:        Error map (e.g. |predicted - measured|), same shape
                      as `reference`. ImageData-like, ndarray, or masked
                      array.
        reference:     The surface the error's topology-dependence is
                      tested against (e.g. the measured/reference height)
                      - gradient and curvature are computed on this, not
                      on the error map itself.
        pixel_size:    (dy, dx) physical pixel size in meters.
        n_bins:        Number of bins for the mean-error-per-gradient/
                      curvature summary.

    Returns:
        ErrorTopologyResult.

    Raises:
        ValidationError: If `error`/`reference` shapes don't match, or
            fewer than 2 valid pixels remain.
    """
    error_arr = _extract_array(error)
    ref_arr = _extract_array(reference)

    if np.asarray(error_arr).shape != np.asarray(ref_arr).shape:
        raise ValidationError(
            f"error and reference must have the same shape, got "
            f"{np.asarray(error_arr).shape} and {np.asarray(ref_arr).shape}."
        )

    gradient = compute_gradient_magnitude(ref_arr, pixel_size)
    curvature = compute_curvature(ref_arr, pixel_size)

    valid = _valid_mask(error_arr, ref_arr, gradient, curvature)
    n_pixels = int(np.sum(valid))
    if n_pixels < 2:
        raise ValidationError(
            f"Only {n_pixels} valid pixels remain - need at least 2 to "
            f"compute a correlation."
        )

    abs_error = np.abs(_raw(error_arr))[valid]
    grad_valid = gradient[valid]
    curv_valid = np.abs(curvature)[valid]

    gradient_correlation = (
        float(np.corrcoef(abs_error, grad_valid)[0, 1])
        if np.std(grad_valid) > 0
        else float("nan")
    )
    curvature_correlation = (
        float(np.corrcoef(abs_error, curv_valid)[0, 1])
        if np.std(curv_valid) > 0
        else float("nan")
    )

    g_centers, g_means, g_counts = _binned_mean(grad_valid, abs_error, n_bins)
    c_centers, c_means, c_counts = _binned_mean(curv_valid, abs_error, n_bins)

    return ErrorTopologyResult(
        gradient_correlation=gradient_correlation,
        curvature_correlation=curvature_correlation,
        gradient_bin_centers=g_centers,
        gradient_bin_mean_error=g_means,
        gradient_bin_counts=g_counts,
        curvature_bin_centers=c_centers,
        curvature_bin_mean_error=c_means,
        curvature_bin_counts=c_counts,
        n_pixels=n_pixels,
    )
