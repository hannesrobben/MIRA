"""
Former standalone Bland-Altman snippet, kept for existing imports.

Superseded by `mira.analysis.comparison.bland_altman_compute` and
`mira.visualization.plotter_stats.plot_bland_altman`; the combined call is
in `mira.visualization.bland_altman`. Proposed for the attic (MI-01).
"""

from .bland_altman import BlandAltmanResult, plot_bland_altman

__all__ = ["BlandAltmanResult", "plot_bland_altman"]
