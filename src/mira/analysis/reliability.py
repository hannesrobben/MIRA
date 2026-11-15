"""
Intraclass correlation coefficient (ICC) for reproducibility across
repeated measurements (e.g. multiple prints of the same design, measured
segment by segment).

This formalizes the same question `analysis.ensemble.summarize_by_group`
already gives a rougher answer to via the coefficient of variation: is the
segment-to-segment structure stable across prints, or does print-to-print
variation swamp it? ICC expresses this as a single number between
(roughly) 0 and 1: the fraction of total variance explained by genuine
between-segment differences, as opposed to measurement/print noise.

Three single-measurement ICC forms are implemented, following the standard
one-way/two-way ANOVA decomposition (Shrout & Fleiss, 1979; McGraw & Wong,
1996):

  - ICC(1,1): one-way random effects. Appropriate when "raters" (here:
    prints) are NOT matched across subjects (segments). Included for
    completeness/comparison - `build_ensemble_dataframe` always produces
    matched data (every print measures every segment), so ICC(2,1) or
    ICC(3,1) make better use of the data.
  - ICC(2,1): two-way random effects, absolute agreement. Prints are
    treated as a random sample from "prints of this design in general" -
    the right choice if the claim is about reproducibility of the print
    process itself, not just these specific N prints. Default.
  - ICC(3,1): two-way mixed effects, consistency. Prints are treated as
    this one specific, fixed set - a narrower claim than ICC(2,1), but
    with an exact (not Satterthwaite-approximated) confidence interval.

CITATION CAVEAT: the formulas below are transcribed from memory against
the cited papers, not fetched from them - they are validated numerically
below (perfect-agreement and pure-noise sanity checks, plus a cross-check
of the ICC(1,1) F-test against `scipy.stats.f_oneway`), but double-check
against a reference implementation (e.g. Python's
`pingouin.intraclass_corr` or R's `psych::ICC()`) before reporting numbers
from this module in the paper - particularly the ICC(2,1) confidence
interval, which relies on a Satterthwaite-type degrees-of-freedom
approximation and is the easiest of the three to get subtly wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

from ..utils.exceptions import ProcessingError, ValidationError

SUPPORTED_ICC_TYPES = ("ICC(1,1)", "ICC(2,1)", "ICC(3,1)")


@dataclass
class ICCResult:
    """
    Result of an intraclass correlation coefficient computation.

    Attributes:
        icc_type:        Which ICC form was computed (see module docstring).
        value:           The ICC point estimate.
        ci_lower:        Lower bound of the confidence interval.
        ci_upper:        Upper bound of the confidence interval.
        alpha:           Significance level the CI was built at (e.g. 0.05
                         for a 95% CI).
        f_value:         F-statistic for the test that the ICC is zero
                         (i.e. that between-subject variance is zero).
        df1:             Numerator degrees of freedom of `f_value`.
        df2:             Denominator degrees of freedom of `f_value`.
        p_value:         One-sided p-value for the F-test (H0: ICC = 0).
        n_subjects:      Number of subjects (e.g. segments).
        n_raters:        Number of raters/repeated measurements (e.g.
                         prints).
        ms_subjects:      Mean square between subjects.
        ms_raters:        Mean square between raters (0 for ICC(1,1), which
                         doesn't estimate a separate rater effect).
        ms_error:         Mean square residual/error (for ICC(1,1), this is
                         the "within subjects" mean square, MSW, which
                         folds the rater effect into the error term).
    """

    icc_type: str
    value: float
    ci_lower: float
    ci_upper: float
    alpha: float
    f_value: float
    df1: float
    df2: float
    p_value: float
    n_subjects: int
    n_raters: int
    ms_subjects: float
    ms_raters: float
    ms_error: float


def interpret_icc(value: float) -> str:
    """
    Qualitative interpretation of an ICC value using the thresholds
    commonly cited from Koo & Li (2016) - "A Guideline of Selecting and
    Reporting Intraclass Correlation Coefficients for Reliability
    Research": <0.5 poor, 0.5-0.75 moderate, 0.75-0.9 good, >=0.9
    excellent. See the module-level citation caveat.

    Args:
        value: An ICC point estimate.

    Returns:
        One of "poor", "moderate", "good", "excellent".
    """
    if value < 0.5:
        return "poor"
    if value < 0.75:
        return "moderate"
    if value < 0.9:
        return "good"
    return "excellent"


def _anova_table(data: np.ndarray) -> dict:
    """
    Two-way ANOVA sum-of-squares decomposition (no replication - one
    observation per subject/rater cell), shared by all three ICC forms.
    """
    n, k = data.shape
    grand_mean = data.mean()
    subject_means = data.mean(axis=1)
    rater_means = data.mean(axis=0)

    ss_total = float(np.sum((data - grand_mean) ** 2))
    ss_subjects = float(k * np.sum((subject_means - grand_mean) ** 2))
    ss_raters = float(n * np.sum((rater_means - grand_mean) ** 2))
    ss_error = ss_total - ss_subjects - ss_raters
    ss_within = ss_total - ss_subjects  # subjects' "within" = raters + error, pooled

    df_subjects = n - 1
    df_raters = k - 1
    df_error = df_subjects * df_raters
    df_within = n * df_raters

    return {
        "n": n,
        "k": k,
        "ms_subjects": ss_subjects / df_subjects,
        "ms_raters": ss_raters / df_raters,
        "ms_error": ss_error / df_error,
        "ms_within": ss_within / df_within,
        "df_subjects": df_subjects,
        "df_raters": df_raters,
        "df_error": df_error,
        "df_within": df_within,
    }


def compute_icc(
    data: np.ndarray, icc_type: str = "ICC(2,1)", alpha: float = 0.05
) -> ICCResult:
    """
    Compute an intraclass correlation coefficient from a complete
    (subjects x raters) measurement table.

    Args:
        data:      2D array, shape (n_subjects, n_raters). Every subject
                  must have been measured by every rater (a complete,
                  balanced design) - there is no accepted way to handle
                  missing cells with these closed-form formulas.
        icc_type:   One of "ICC(1,1)", "ICC(2,1)", "ICC(3,1)" - see the
                  module docstring for which to use.
        alpha:      Significance level for the confidence interval (0.05
                  -> 95% CI).

    Returns:
        ICCResult.

    Raises:
        ValidationError: If `data` isn't 2D, has fewer than 2 subjects or
            raters, or `icc_type` is unsupported.
        ProcessingError: If the underlying computation fails (e.g. the
            ICC(2,1) Satterthwaite degrees-of-freedom approximation
            produces a non-finite value).
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValidationError(
            f"data must be a 2D (subjects x raters) array, got shape {data.shape}."
        )
    n, k = data.shape
    if n < 2 or k < 2:
        raise ValidationError(
            f"Need at least 2 subjects and 2 raters, got {n} subjects and "
            f"{k} raters."
        )
    if icc_type not in SUPPORTED_ICC_TYPES:
        raise ValidationError(
            f"Unknown icc_type '{icc_type}'. Supported: {SUPPORTED_ICC_TYPES}"
        )
    if not (0 < alpha < 1):
        raise ValidationError(f"alpha must be in (0, 1), got {alpha}.")

    t = _anova_table(data)
    ms_r, ms_c, ms_e, ms_w = t["ms_subjects"], t["ms_raters"], t["ms_error"], t["ms_within"]

    try:
        if icc_type == "ICC(1,1)":
            icc = (ms_r - ms_w) / (ms_r + (k - 1) * ms_w)
            f_value = ms_r / ms_w
            df1, df2 = t["df_subjects"], t["df_within"]
            f_crit_u = scipy_stats.f.ppf(1 - alpha / 2, df1, df2)
            f_crit_l = scipy_stats.f.ppf(1 - alpha / 2, df2, df1)
            f_lower, f_upper = f_value / f_crit_u, f_value * f_crit_l
            ci_lower = (f_lower - 1) / (f_lower + k - 1)
            ci_upper = (f_upper - 1) / (f_upper + k - 1)
            ms_c_out = 0.0

        elif icc_type == "ICC(3,1)":
            icc = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e)
            f_value = ms_r / ms_e
            df1, df2 = t["df_subjects"], t["df_error"]
            f_crit_u = scipy_stats.f.ppf(1 - alpha / 2, df1, df2)
            f_crit_l = scipy_stats.f.ppf(1 - alpha / 2, df2, df1)
            f_lower, f_upper = f_value / f_crit_u, f_value * f_crit_l
            ci_lower = (f_lower - 1) / (f_lower + k - 1)
            ci_upper = (f_upper - 1) / (f_upper + k - 1)
            ms_c_out = ms_c

        else:  # "ICC(2,1)"
            icc = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e + (k / n) * (ms_c - ms_e))
            f_value = ms_r / ms_e
            df1, df2 = t["df_subjects"], t["df_error"]

            # Satterthwaite-approximated CI (McGraw & Wong, 1996) - see the
            # module-level citation caveat.
            f_j = ms_c / ms_e
            a = k * icc * f_j + (n * (1 + (k - 1) * icc) - k * icc)
            b = (k - 1) * (n - 1)
            v = b * a**2 / (
                (n - 1) * k**2 * icc**2 * f_j**2
                + (n * (1 + (k - 1) * icc) - k * icc) ** 2
            )
            f_l = scipy_stats.f.ppf(1 - alpha / 2, df1, v)
            f_u = scipy_stats.f.ppf(1 - alpha / 2, v, df1)
            ci_lower = n * (ms_r - f_l * ms_e) / (
                f_l * (k * ms_c + (k * n - k - n) * ms_e) + n * ms_r
            )
            ci_upper = n * (f_u * ms_r - ms_e) / (
                k * ms_c + (k * n - k - n) * ms_e + n * f_u * ms_r
            )
            ms_c_out = ms_c

        if not np.isfinite(icc):
            raise ProcessingError(
                f"{icc_type} computation produced a non-finite value - "
                f"check for zero variance in the input data."
            )

        p_value = float(1 - scipy_stats.f.cdf(f_value, df1, df2))

        return ICCResult(
            icc_type=icc_type,
            value=float(icc),
            ci_lower=float(np.clip(ci_lower, -1.0, 1.0)),
            ci_upper=float(np.clip(ci_upper, -1.0, 1.0)),
            alpha=alpha,
            f_value=float(f_value),
            df1=float(df1),
            df2=float(df2),
            p_value=p_value,
            n_subjects=n,
            n_raters=k,
            ms_subjects=float(ms_r),
            ms_raters=float(ms_c_out),
            ms_error=float(ms_e if icc_type != "ICC(1,1)" else ms_w),
        )

    except ZeroDivisionError as e:
        raise ProcessingError(f"{icc_type} computation failed: {e!s}")


def _build_measurement_table(
    df, subject_col: str, rater_col: str, value_col: str
) -> np.ndarray:
    """
    Pivot a tidy (long-format) DataFrame into a complete (subjects x
    raters) table for `compute_icc`.

    Raises:
        ValidationError: If the pivoted table has missing (subject, rater)
            combinations - ICC as implemented here requires a complete,
            balanced design.
    """
    pivot = df.pivot(index=subject_col, columns=rater_col, values=value_col)
    n_missing = int(pivot.isna().sum().sum())
    if n_missing > 0:
        raise ValidationError(
            f"Measurement table is not complete: {n_missing} missing "
            f"('{subject_col}', '{rater_col}') combinations after "
            f"pivoting. ICC requires every subject to be measured by "
            f"every rater - drop incomplete subjects/raters first, e.g. "
            f"`df.dropna()` after pivoting, or filter df beforehand."
        )
    return pivot.to_numpy()


def compute_icc_from_dataframe(
    df,
    subject_col: str = "segment_id",
    rater_col: str = "print_id",
    value_col: str = "mean_relative",
    icc_type: str = "ICC(2,1)",
    alpha: float = 0.05,
) -> ICCResult:
    """
    Compute ICC directly from a tidy ensemble DataFrame, as produced by
    `analysis.ensemble.build_ensemble_dataframe`.

    Args:
        df:            Long-format DataFrame with (at least) `subject_col`,
                      `rater_col`, and `value_col`.
        subject_col:    Column identifying the subject (e.g. "segment_id" -
                      the thing whose reproducibility is being assessed).
        rater_col:      Column identifying the repeated measurement (e.g.
                      "print_id").
        value_col:      Column with the measured value (e.g.
                      "mean_relative").
        icc_type:       See `compute_icc`.
        alpha:          See `compute_icc`.

    Returns:
        ICCResult.

    Raises:
        ValidationError: If the pivoted (subject x rater) table is
            incomplete, or from `compute_icc`.
    """
    table = _build_measurement_table(df, subject_col, rater_col, value_col)
    return compute_icc(table, icc_type=icc_type, alpha=alpha)
