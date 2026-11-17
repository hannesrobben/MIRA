"""
Combined Bland-Altman entry point, kept for existing scripts.

The analysis is split into `mira.analysis.comparison.bland_altman_compute`
(statistics, returns `BlandAltmanResult`) and
`mira.visualization.plotter_stats.plot_bland_altman` (figure). This module
keeps the former call `plot_bland_altman(data_1, data_2, ...) -> (figure,
result)` working on top of the two. New code should call the two functions
directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..analysis.comparison import BlandAltmanResult, bland_altman_compute
from ..utils.exceptions import ValidationError, VisualizationError
from .plotter_stats import plot_bland_altman as _plot_result

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

__all__ = ["BlandAltmanResult", "plot_bland_altman"]


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
    Compute and plot a Bland-Altman comparison in one call (former API).

    Equivalent to ``bland_altman_compute(data_1, data_2, relative=...,
    ci_multiplier=...)`` followed by
    ``mira.visualization.plotter_stats.plot_bland_altman(result, ...)``.
    Masked entries of masked arrays are left out (the former version of this
    module removed the mask).

    Parameters
    ----------
    data_1, data_2 : array_like, numpy.ma.MaskedArray or ImageData-like
        Paired measurements of the same shape.
    label_1, label_2, units, title, fontsize, point_size, point_alpha, ax, \
dpi, save_path
        See `mira.visualization.plotter_stats.plot_bland_altman`.
    relative, ci_multiplier
        See `mira.analysis.comparison.bland_altman_compute`.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, BlandAltmanResult)

    Raises
    ------
    VisualizationError
        If the inputs are invalid (shape mismatch, no finite pair) or
        plotting fails, as before.
    """
    try:
        result = bland_altman_compute(
            data_1, data_2, relative=relative, ci_multiplier=ci_multiplier
        )
    except ValidationError as e:
        raise VisualizationError(str(e)) from e
    fig = _plot_result(
        result,
        label_1=label_1,
        label_2=label_2,
        units=units,
        title=title,
        fontsize=fontsize,
        point_size=point_size,
        point_alpha=point_alpha,
        ax=ax,
        dpi=dpi,
        save_path=save_path,
    )
    return fig, result
