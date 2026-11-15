from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from ..utils.exceptions import VisualizationError


@dataclass
class BlandAltmanResult:
    """
    Summary statistics from a Bland-Altman comparison of two measurement
    methods.

    Attributes:
        bias:            Mean difference between the two methods
                         (systematic offset).
        std_diff:        Standard deviation of the differences.
        loa_lower:        Lower limit of agreement (bias - ci_multiplier * std_diff).
        loa_upper:        Upper limit of agreement (bias + ci_multiplier * std_diff).
        n_points:        Number of finite (non-NaN) sample pairs actually used.
        mean_of_means:   Mean of the per-pixel averages, for reference /
                         sanity-checking against the plotted x-range.
    """

    bias: float
    std_diff: float
    loa_lower: float
    loa_upper: float
    n_points: int
    mean_of_means: float


def plot_bland_altman(
    data_1,
    data_2,
    label_1: str = "Method 1",
    label_2: str = "Method 2",
    units: str = "µm",
    relative: bool = False,
    ci_multiplier: float = 1.96,
    title: str = "Bland-Altman Plot",
    fontsize: int = 18,
    point_size: float = 8.0,
    point_alpha: float = 0.35,
    ax: plt.Axes | None = None,
    dpi: int = 300,
    save_path: str | None = None,
) -> tuple[plt.Figure, BlandAltmanResult]:
    """
    Create a Bland-Altman (mean-difference) plot comparing two measurements
    of the same quantity, e.g. predicted height vs. reference height.

    The Bland-Altman plot separates systematic bias (the mean difference,
    plotted as a horizontal line) from the spread of that difference (the
    limits of agreement, bias +/- ci_multiplier * std). This is more
    informative than a single RMSE value, since two data sets can have the
    same RMSE while one has a pure offset (fixable via calibration) and the
    other has genuinely random scatter (a noise floor).

    Args:
        data_1:         First measurement (e.g. predicted height). Either an
                        ImageData-like object exposing `.data`, or a raw
                        np.ndarray. Any shape - will be flattened.
        data_2:         Second measurement (e.g. reference height). Must be
                        the same shape as data_1.
        label_1:        Display name for data_1, used in the axis/title text.
        label_2:        Display name for data_2, used in the axis/title text.
        units:          Physical unit string for axis labels (e.g. "µm").
        relative:       If True, plot the difference as a percentage of the
                        pair mean ((d1-d2) / mean * 100) instead of the
                        absolute difference. Useful when the measurement
                        range varies a lot across the data set.
        ci_multiplier:  Multiplier for the limits of agreement, applied to
                        the standard deviation of the differences.
                        1.96 corresponds to the classic ~95% agreement
                        interval under a normality assumption.
        title:          Plot title.
        fontsize:       Base font size; axis/tick labels are derived from it.
        point_size:     Marker size for the scatter cloud.
        point_alpha:    Marker transparency - keep this low for dense data,
                        so overlapping points are visible as darker regions.
        ax:             Optional existing axes to plot on. If None, a new
                        figure/axes pair is created.
        dpi:            Resolution used only when a new figure is created.
        save_path:      If given, the figure is saved to this path.

    Returns:
        A tuple of (Figure, BlandAltmanResult) - the figure for further
        styling/saving, and the summary statistics for programmatic reuse
        (e.g. dumping into the same info_dictionary.json you already build
        in investigate_difference()).

    Raises:
        VisualizationError: If data_1/data_2 have mismatched shapes, contain
            no finite sample pairs, or plotting otherwise fails.
    """
    try:
        # --- Data extraction: accept ImageData-like objects or raw arrays ---
        arr_1 = data_1.data if hasattr(data_1, "data") else data_1
        arr_2 = data_2.data if hasattr(data_2, "data") else data_2
        arr_1 = np.asarray(arr_1)
        arr_2 = np.asarray(arr_2)

        if arr_1.shape != arr_2.shape:
            raise VisualizationError(
                f"Shape mismatch between '{label_1}' {arr_1.shape} and "
                f"'{label_2}' {arr_2.shape} - Bland-Altman needs paired "
                f"samples of the same measurement locations."
            )

        means = (arr_1 + arr_2) / 2.0
        diffs = arr_1 - arr_2
        if relative:
            with np.errstate(divide="ignore", invalid="ignore"):
                diffs = np.where(means != 0, diffs / means * 100.0, np.nan)

        means_flat = means.ravel()
        diffs_flat = diffs.ravel()

        # --- Drop non-finite pairs (NaNs from masked regions, division above, etc.) ---
        finite_mask = np.isfinite(means_flat) & np.isfinite(diffs_flat)
        means_flat = means_flat[finite_mask]
        diffs_flat = diffs_flat[finite_mask]

        if means_flat.size == 0:
            raise VisualizationError(
                "No finite sample pairs remain after filtering NaN/Inf - "
                "cannot compute Bland-Altman statistics."
            )

        bias = float(np.mean(diffs_flat))
        std_diff = float(np.std(diffs_flat, ddof=1)) if means_flat.size > 1 else 0.0
        loa_lower = bias - ci_multiplier * std_diff
        loa_upper = bias + ci_multiplier * std_diff

        result = BlandAltmanResult(
            bias=bias,
            std_diff=std_diff,
            loa_lower=loa_lower,
            loa_upper=loa_upper,
            n_points=int(means_flat.size),
            mean_of_means=float(np.mean(means_flat)),
        )

        # --- Figure / axes setup ---
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 6), dpi=dpi)
        else:
            fig = ax.figure

        ax.scatter(
            means_flat,
            diffs_flat,
            s=point_size,
            alpha=point_alpha,
            color="steelblue",
            edgecolors="none",
            rasterized=True,  # keeps vector saves (e.g. PDF) manageable for large point clouds
        )

        ax.axhline(
            bias,
            color="black",
            linestyle="-",
            linewidth=1.5,
            label=f"Bias = {bias:.3g}",
        )
        ax.axhline(
            loa_upper,
            color="firebrick",
            linestyle="--",
            linewidth=1.5,
            label=f"+{ci_multiplier:g} SD = {loa_upper:.3g}",
        )
        ax.axhline(
            loa_lower,
            color="firebrick",
            linestyle="--",
            linewidth=1.5,
            label=f"-{ci_multiplier:g} SD = {loa_lower:.3g}",
        )
        ax.axhline(0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)

        diff_unit = "%" if relative else units
        ax.set_xlabel(
            f"Mean of {label_1} and {label_2} [{units}]", fontsize=fontsize - 2
        )
        ax.set_ylabel(f"{label_1} - {label_2} [{diff_unit}]", fontsize=fontsize - 2)
        ax.set_title(title, fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=fontsize - 4)
        ax.legend(loc="best", fontsize=fontsize - 6, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=":")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")

        return fig, result

    except Exception as e:
        if isinstance(e, VisualizationError):
            raise
        raise VisualizationError(f"Failed to plot Bland-Altman comparison: {e!s}")
