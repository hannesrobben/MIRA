"""
Comparison metrics between two measurement channels.

Provides linear-regression-based comparison (slope, intercept, R^2, p-value)
between two data series - either raw arrays, or two identically segmented
Structures compared segment by segment - and the Bland-Altman agreement
statistics (bias, limits of agreement) of two paired measurements. The
matching plots are in `mira.visualization.plotter_stats`.

Replaces `calculate_linear_regression_metrics` /
`calculate_linear_regression_metrics_old` /
`calculate_linear_regression_metrics_dhm`, which existed as three
near-duplicate implementations across the legacy calibration scripts, each
tied to a specific nested-dict data format. This module works on plain
arrays (or Structures, via `fit_segment_regression`), so any new comparison
- a different segmentation, a different pair of quantities, a subset of
segments - is a call with different arguments, not a new function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.ma as ma
from scipy import stats

from ..core.structure import Structure
from ..utils.exceptions import ProcessingError, ValidationError
from .segment_data import SegmentStatistic, paired_segment_values


@dataclass
class RegressionResult:
    """
    Result of an ordinary least-squares linear regression y = slope*x + intercept.

    Attributes:
        slope:         Fitted slope.
        intercept:      Fitted intercept.
        r_value:       Pearson correlation coefficient.
        r_squared:      Coefficient of determination (r_value ** 2).
        p_value:       Two-sided p-value for a hypothesis test whose null
                       hypothesis is that the slope is zero.
        std_err:       Standard error of the slope estimate.
        residual_std:   Standard deviation of the fit residuals
                       (y - predicted).
        n_points:       Number of (x, y) pairs used in the fit.
        x_values:      The x data used for the fit.
        y_values:      The y data used for the fit.
    """

    slope: float
    intercept: float
    r_value: float
    r_squared: float
    p_value: float
    std_err: float
    residual_std: float
    n_points: int
    x_values: np.ndarray
    y_values: np.ndarray

    @property
    def predicted(self) -> np.ndarray:
        """Fitted y values (slope * x + intercept) for `x_values`."""
        return self.slope * self.x_values + self.intercept

    @property
    def residuals(self) -> np.ndarray:
        """Residuals (y_values - predicted)."""
        return self.y_values - self.predicted


def fit_linear_regression(x: np.ndarray, y: np.ndarray) -> RegressionResult:
    """
    Fit an ordinary least-squares line y = slope*x + intercept.

    Args:
        x: Independent variable values. A masked array is allowed; every
           pair in which `x` or `y` is masked is left out of the fit.
        y: Dependent variable values, same length as `x`.

    Returns:
        RegressionResult with fit parameters and quality metrics.
        `x_values` / `y_values` hold only the pairs used in the fit.

    Raises:
        ValidationError: If `x`/`y` have mismatched lengths or fewer than
            two (unmasked) points.
        ProcessingError: If the fit itself fails (e.g. no variation in x).
    """
    if isinstance(x, ma.MaskedArray) or isinstance(y, ma.MaskedArray):
        x_masked = ma.asarray(x, dtype=float)
        y_masked = ma.asarray(y, dtype=float)
        if x_masked.shape != y_masked.shape:
            raise ValidationError(
                f"x and y must have the same shape, got {x_masked.shape} and "
                f"{y_masked.shape}."
            )
        keep = ~(ma.getmaskarray(x_masked) | ma.getmaskarray(y_masked))
        x = x_masked.data[keep]
        y = y_masked.data[keep]
    else:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValidationError(
            f"x and y must have the same shape, got {x.shape} and {y.shape}."
        )
    if x.size < 2:
        raise ValidationError(
            f"At least 2 data points are required for a regression fit, "
            f"got {x.size}."
        )

    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    except Exception as e:
        raise ProcessingError(f"Linear regression fit failed: {e!s}")

    result = RegressionResult(
        slope=float(slope),
        intercept=float(intercept),
        r_value=float(r_value),
        r_squared=float(r_value**2),
        p_value=float(p_value),
        std_err=float(std_err),
        residual_std=0.0,  # filled in below
        n_points=int(x.size),
        x_values=x,
        y_values=y,
    )
    result.residual_std = float(np.std(result.residuals))
    return result


def fit_segment_regression(
    structure_x: Structure,
    structure_y: Structure,
    offset_pixels: int = 10,
    include_background: bool = True,
    relative: bool = True,
) -> RegressionResult:
    """
    Fit a linear regression between two identically segmented structures,
    using one representative (mean) value per segment (and optionally the
    background) as a single data point.

    This is the segment-wise counterpart to `fit_linear_regression`: it
    handles the segment extraction and pairing (via
    `analysis.segment_data.paired_segment_values`) and then delegates to
    `fit_linear_regression` for the actual fit.

    Args:
        structure_x:          Segmented Structure providing the x values
                              (independent variable), e.g. a reference
                              measurement.
        structure_y:          Segmented Structure providing the y values,
                              segmented identically to `structure_x`.
        offset_pixels:        Pixels excluded from every segment/background
                              edge before computing statistics.
        include_background:   If True, include the background as an
                              additional data point.
        relative:              If True, use background-relative segment
                              means instead of absolute values (recommended
                              when the two channels have different offsets).

    Returns:
        RegressionResult.

    Raises:
        ValidationError: If either structure is unsegmented, or their
            segment counts don't match.
        ProcessingError: If the fit itself fails.
    """
    x_values, y_values, _, _ = paired_segment_values(
        structure_x,
        structure_y,
        offset_pixels=offset_pixels,
        include_background=include_background,
        relative=relative,
    )
    return fit_linear_regression(x_values, y_values)


@dataclass
class BlandAltmanResult:
    """
    Agreement statistics of two paired measurements (Bland-Altman analysis).

    The first six fields keep the order of the former
    `mira.visualization.bland_altman.BlandAltmanResult`, so positional
    construction in old scripts still works.

    Attributes
    ----------
    bias : float
        Mean difference data_1 - data_2 (systematic offset).
    std_diff : float
        Standard deviation of the differences (ddof=1; 0.0 for one pair).
    loa_lower : float
        Lower limit of agreement, bias - ci_multiplier * std_diff.
    loa_upper : float
        Upper limit of agreement, bias + ci_multiplier * std_diff.
    n_points : int
        Number of finite, unmasked pairs used.
    mean_of_means : float
        Mean of the pair means.
    ci_multiplier : float
        Multiplier used for the limits of agreement.
    relative : bool
        True if the differences are percentages of the pair mean.
    means : numpy.ndarray or None
        Pair means of the pairs used (1D), for plotting.
    differences : numpy.ndarray or None
        Differences of the pairs used (1D; percent if `relative`).
    """

    bias: float
    std_diff: float
    loa_lower: float
    loa_upper: float
    n_points: int
    mean_of_means: float
    ci_multiplier: float = 1.96
    relative: bool = False
    means: np.ndarray | None = None
    differences: np.ndarray | None = None


def _extract_values(value) -> np.ndarray:
    """
    Return the array behind `value`, keeping the mask of masked arrays.

    ndarray and MaskedArray are returned unchanged. A MaskedArray also
    exposes `.data` (the values *without* the mask), so `.data` is only
    used for wrapper objects such as `ImageData`; anything else goes
    through `np.asarray`.
    """
    if isinstance(value, np.ndarray):  # includes numpy.ma.MaskedArray
        return value
    if hasattr(value, "data"):
        return value.data
    return np.asarray(value)


def bland_altman_compute(
    data_1,
    data_2,
    *,
    relative: bool = False,
    ci_multiplier: float = 1.96,
) -> BlandAltmanResult:
    """
    Compute bias and limits of agreement of two paired measurements.

    The Bland-Altman analysis separates a systematic offset (the bias, the
    mean difference) from the spread of the differences (the limits of
    agreement, bias +/- ci_multiplier * SD). Two data sets with the same
    RMSE can differ completely here: a pure offset can be calibrated away,
    random scatter cannot.

    Parameters
    ----------
    data_1, data_2 : array_like, numpy.ma.MaskedArray or ImageData-like
        The two measurements at the same locations (e.g. predicted and
        reference height), same shape; any shape, flattened internally.
        Objects that are not arrays are unwrapped through their `.data`
        attribute (e.g. `ImageData`). Masked entries and non-finite values
        in either input remove the pair.
    relative : bool, optional
        If True, the differences are expressed in percent of the pair mean,
        (d1 - d2) / mean * 100; pairs with mean 0 are left out.
    ci_multiplier : float, optional
        Multiplier of the SD for the limits of agreement; 1.96 gives the
        usual ~95 % interval under a normality assumption.

    Returns
    -------
    BlandAltmanResult
        Statistics plus the pair means and differences that were used.

    Raises
    ------
    ValidationError
        If the shapes differ or no finite, unmasked pair remains.

    Notes
    -----
    The arithmetic follows the former combined function
    `mira.visualization.bland_altman_snippet.plot_bland_altman` step by
    step (differences in the input dtype, statistics in float64), so its
    results are reproduced exactly.
    """
    arr_1 = _extract_values(data_1)
    arr_2 = _extract_values(data_2)
    if not isinstance(arr_1, ma.MaskedArray):
        arr_1 = np.asarray(arr_1)
    if not isinstance(arr_2, ma.MaskedArray):
        arr_2 = np.asarray(arr_2)

    if arr_1.shape != arr_2.shape:
        raise ValidationError(
            f"Shape mismatch between data_1 {arr_1.shape} and data_2 "
            f"{arr_2.shape} - Bland-Altman needs paired samples of the same "
            f"measurement locations."
        )

    means = (arr_1 + arr_2) / 2.0
    diffs = arr_1 - arr_2
    if relative:
        with np.errstate(divide="ignore", invalid="ignore"):
            diffs = np.where(means != 0, diffs / means * 100.0, np.nan)

    # Masked entries become NaN, so one finiteness check removes masked
    # pairs, NaNs of the input and pairs with mean 0 in relative mode.
    means_flat = ma.filled(means.ravel().astype(float), np.nan)
    diffs_flat = ma.filled(diffs.ravel().astype(float), np.nan)
    finite = np.isfinite(means_flat) & np.isfinite(diffs_flat)
    means_flat = means_flat[finite]
    diffs_flat = diffs_flat[finite]

    if means_flat.size == 0:
        raise ValidationError(
            "No finite, unmasked sample pairs remain - cannot compute "
            "Bland-Altman statistics."
        )

    bias = float(np.mean(diffs_flat))
    std_diff = float(np.std(diffs_flat, ddof=1)) if means_flat.size > 1 else 0.0
    return BlandAltmanResult(
        bias=bias,
        std_diff=std_diff,
        loa_lower=bias - ci_multiplier * std_diff,
        loa_upper=bias + ci_multiplier * std_diff,
        n_points=int(means_flat.size),
        mean_of_means=float(np.mean(means_flat)),
        ci_multiplier=float(ci_multiplier),
        relative=bool(relative),
        means=means_flat,
        differences=diffs_flat,
    )
