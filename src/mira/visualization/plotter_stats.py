"""
Statistical comparison plots.

Plot functions that consume the result dataclasses from `analysis/` (e.g.
`RegressionResult` from `analysis.comparison`) and render them - kept
separate from `plotter_2d.py` (which handles image/structure visualization)
so this file stays focused on statistics-over-scalars plots as more of them
get added (Bland-Altman, error-vs-curvature, PSD, Zernike bar charts, ...).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..analysis.comparison import RegressionResult
from ..analysis.sensitivity import SensitivityResult
from ..analysis.topology import ErrorTopologyResult
from ..analysis.reliability import ICCResult
from ..utils.exceptions import VisualizationError


def plot_regression(
    result: RegressionResult,
    x_label: str = "x",
    y_label: str = "y",
    title: str = "Linear Regression",
    fontsize: int = 18,
    point_size: float = 40.0,
    show_ci_band: bool = True,
    ax: plt.Axes | None = None,
    dpi: int = 300,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot a linear regression result: the data points, the fitted line, and
    a text box summarizing slope/intercept/R^2/p-value.

    Args:
        result:        RegressionResult from `analysis.comparison.
                       fit_linear_regression` or `fit_segment_regression`.
        x_label:       Label for the x-axis (should name the quantity/unit).
        y_label:       Label for the y-axis.
        title:         Plot title.
        fontsize:      Base font size; axis/tick labels are derived from it.
        point_size:    Marker size for the data points.
        show_ci_band:  If True, shade a +/- 1 residual_std band around the
                       fit line as a quick visual sense of scatter.
        ax:            Optional existing axes to plot on. If None, a new
                       figure/axes pair is created.
        dpi:           Resolution used only when a new figure is created.
        save_path:     If given, the figure is saved to this path.

    Returns:
        Figure object.

    Raises:
        VisualizationError: If plotting fails.
    """
    try:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
        else:
            fig = ax.figure

        x, y = result.x_values, result.y_values
        ax.scatter(
            x,
            y,
            s=point_size,
            color="steelblue",
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
            label="Data",
        )

        x_line = np.linspace(np.min(x), np.max(x), 100)
        y_line = result.slope * x_line + result.intercept
        ax.plot(
            x_line,
            y_line,
            color="firebrick",
            linewidth=2,
            zorder=2,
            label="Fit",
        )

        if show_ci_band and result.residual_std > 0:
            ax.fill_between(
                x_line,
                y_line - result.residual_std,
                y_line + result.residual_std,
                color="firebrick",
                alpha=0.15,
                zorder=1,
                label="±1 residual SD",
            )

        stats_text = (
            f"slope = {result.slope:.4g}\n"
            f"intercept = {result.intercept:.4g}\n"
            f"R² = {result.r_squared:.4f}\n"
            f"p = {result.p_value:.3g}\n"
            f"n = {result.n_points}"
        )
        ax.text(
            0.03,
            0.97,
            stats_text,
            transform=ax.transAxes,
            fontsize=fontsize - 6,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"
            ),
        )

        ax.set_xlabel(x_label, fontsize=fontsize - 2)
        ax.set_ylabel(y_label, fontsize=fontsize - 2)
        ax.set_title(title, fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=fontsize - 4)
        ax.legend(loc="lower right", fontsize=fontsize - 6, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=":")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")

        return fig

    except Exception as e:
        if isinstance(e, VisualizationError):
            raise
        raise VisualizationError(f"Failed to plot regression: {e!s}")


def plot_error_vs_topology(
    result: ErrorTopologyResult,
    error_unit: str = "a.u.",
    gradient_unit: str = "a.u./m",
    curvature_unit: str = "a.u./m²",
    title: str = "Error vs. Surface Topology",
    fontsize: int = 18,
    dpi: int = 300,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot mean |error| against binned gradient magnitude and curvature,
    side by side, with pixel counts per bin shown as a background bar
    chart on a secondary axis.

    Args:
        result:         ErrorTopologyResult from
                        `analysis.topology.analyze_error_vs_topology`.
        error_unit:     Unit string for the error axis.
        gradient_unit:  Unit string for the gradient-magnitude axis.
        curvature_unit: Unit string for the curvature axis.
        title:          Figure-level title.
        fontsize:       Base font size; axis/tick labels are derived from it.
        dpi:            Resolution.
        save_path:      If given, the figure is saved to this path.

    Returns:
        Figure with two panels: mean |error| vs. gradient magnitude, and
        mean |error| vs. |curvature|.

    Raises:
        VisualizationError: If plotting fails.
    """
    try:
        fig, (ax_grad, ax_curv) = plt.subplots(1, 2, figsize=(14, 6), dpi=dpi)

        panels = [
            (
                ax_grad,
                result.gradient_bin_centers,
                result.gradient_bin_mean_error,
                result.gradient_bin_counts,
                result.gradient_correlation,
                "Gradient magnitude",
                gradient_unit,
            ),
            (
                ax_curv,
                result.curvature_bin_centers,
                result.curvature_bin_mean_error,
                result.curvature_bin_counts,
                result.curvature_correlation,
                "|Curvature|",
                curvature_unit,
            ),
        ]

        for ax, centers, means, counts, corr, label, unit in panels:
            valid = ~np.isnan(means)
            width = (centers[1] - centers[0]) if len(centers) > 1 else 1.0

            count_ax = ax.twinx()
            count_ax.bar(
                centers,
                counts,
                width=width * 0.9,
                alpha=0.15,
                color="steelblue",
                zorder=1,
            )
            count_ax.set_ylabel("Pixel count", fontsize=fontsize - 6, color="steelblue")
            count_ax.tick_params(
                axis="y", labelsize=fontsize - 6, labelcolor="steelblue"
            )

            ax.plot(centers[valid], means[valid], "o-", color="firebrick", zorder=3)
            ax.set_xlabel(f"{label} [{unit}]", fontsize=fontsize - 2)
            ax.set_ylabel(f"Mean |error| [{error_unit}]", fontsize=fontsize - 2)
            ax.set_title(f"{label}: r = {corr:.3f}", fontsize=fontsize - 2)
            ax.tick_params(axis="both", labelsize=fontsize - 4)
            ax.grid(True, alpha=0.3, linestyle=":")
            ax.set_zorder(count_ax.get_zorder() + 1)
            ax.patch.set_visible(False)

        fig.suptitle(f"{title} (n = {result.n_pixels})", fontsize=fontsize)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")

        return fig

    except Exception as e:
        if isinstance(e, VisualizationError):
            raise
        raise VisualizationError(f"Failed to plot error-vs-topology: {e!s}")


def plot_delta_n_sensitivity(
    result: SensitivityResult,
    fontsize: int = 18,
    dpi: int = 300,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot RMSE vs. swept delta_n, with the nominal (fitted) delta_n, the
    best-fit-within-sweep delta_n, and the propagated 1-sigma uncertainty
    band marked - the figure to put next to an uncertainty statement in
    the paper.

    Args:
        result:     SensitivityResult from
                   `analysis.sensitivity.sweep_delta_n_sensitivity`.
        fontsize:   Base font size; axis/tick labels are derived from it.
        dpi:        Resolution.
        save_path:  If given, the figure is saved to this path.

    Returns:
        Figure object.

    Raises:
        VisualizationError: If plotting fails.
    """
    try:
        fig, ax = plt.subplots(figsize=(9, 6), dpi=dpi)

        ax.plot(
            result.delta_n_values,
            result.rmse_values,
            color="steelblue",
            linewidth=2,
            zorder=3,
            label="RMSE(Δn)",
        )
        ax.axvspan(
            result.delta_n_nominal - result.delta_n_uncertainty,
            result.delta_n_nominal + result.delta_n_uncertainty,
            color="black",
            alpha=0.08,
            zorder=1,
            label=f"±1σ (σ_Δn = {result.delta_n_uncertainty:.2g})",
        )
        ax.axvline(
            result.delta_n_nominal,
            color="black",
            linestyle="-",
            linewidth=1.5,
            zorder=2,
            label=f"Nominal Δn = {result.delta_n_nominal:.5g}",
        )
        ax.axvline(
            result.best_fit_delta_n,
            color="firebrick",
            linestyle="--",
            linewidth=1.5,
            zorder=2,
            label=f"Best fit in range = {result.best_fit_delta_n:.5g}",
        )

        ax.set_xlabel("Δn = n_resin - n_air", fontsize=fontsize - 2)
        ax.set_ylabel("RMSE (phase residual)", fontsize=fontsize - 2)
        ax.set_title("Δn Sensitivity to n_resin / n_air Uncertainty", fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=fontsize - 4)
        ax.legend(loc="best", fontsize=fontsize - 6, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=":")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")

        return fig

    except Exception as e:
        if isinstance(e, VisualizationError):
            raise
        raise VisualizationError(f"Failed to plot delta_n sensitivity: {e!s}")


def plot_icc_summary(
    results: dict[str, ICCResult],
    title: str = "Intraclass Correlation Coefficient",
    fontsize: int = 18,
    dpi: int = 300,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Forest plot of one or more ICCResult values with their confidence
    intervals - e.g. one ICC per DOE, or one per segment group, side by
    side for comparison.

    Args:
        results:     Dict mapping a label (e.g. a DOE id) to its
                     ICCResult. All entries should use the same
                     `icc_type` for a fair comparison - this isn't
                     enforced, but a mismatch is shown in the x-axis
                     label of any entry whose type differs from the
                     first one's, so it can't slip by unnoticed.
        title:       Plot title.
        fontsize:    Base font size; axis/tick labels are derived from it.
        dpi:         Resolution.
        save_path:   If given, the figure is saved to this path.

    Returns:
        Figure object.

    Raises:
        VisualizationError: If plotting fails, or `results` is empty.
    """
    try:
        if not results:
            raise VisualizationError("plot_icc_summary requires at least one result.")

        labels = list(results.keys())
        values = [results[label].value for label in labels]
        err_lower = [results[label].value - results[label].ci_lower for label in labels]
        err_upper = [results[label].ci_upper - results[label].value for label in labels]
        icc_type_ref = results[labels[0]].icc_type

        fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(labels) + 1.5)), dpi=dpi)
        y_pos = np.arange(len(labels))

        ax.errorbar(
            values,
            y_pos,
            xerr=[err_lower, err_upper],
            fmt="o",
            color="steelblue",
            ecolor="steelblue",
            elinewidth=2,
            capsize=4,
            markersize=8,
            zorder=3,
        )
        for band, color, label in [
            ((0.0, 0.5), "#f4cccc", "poor"),
            ((0.5, 0.75), "#fff2cc", "moderate"),
            ((0.75, 0.9), "#d9ead3", "good"),
            ((0.9, 1.0), "#b6d7a8", "excellent"),
        ]:
            ax.axvspan(band[0], band[1], color=color, alpha=0.4, zorder=0)

        ax.axvline(0, color="gray", linestyle=":", linewidth=1)
        ax.set_yticks(y_pos)
        y_labels = [
            f"{label}"
            + (
                f"\n[{results[label].icc_type}]"
                if results[label].icc_type != icc_type_ref
                else ""
            )
            for label in labels
        ]
        ax.set_yticklabels(y_labels, fontsize=fontsize - 4)
        ax.set_xlabel(f"{icc_type_ref} (point estimate ± CI)", fontsize=fontsize - 2)
        ax.set_xlim(min(-0.2, min(values) - 0.1), 1.05)
        ax.set_title(title, fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=fontsize - 4)
        ax.grid(True, axis="x", alpha=0.3, linestyle=":")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")

        return fig

    except Exception as e:
        if isinstance(e, VisualizationError):
            raise
        raise VisualizationError(f"Failed to plot ICC summary: {e!s}")
