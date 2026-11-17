"""
Visualization module for the image processing library.

This module provides functionality for visualizing image data,
structures, and analysis results in both 2D and 3D formats.
"""

from ..analysis.comparison import BlandAltmanResult
from .plotter_stats import plot_bland_altman
from .plotter_2d import (
    DualProfileResult,
    create_masked_overlay,
    flatten_profile_dict,
    plot_comparison,
    plot_dual_profile,
    plot_image,
    plot_profile_collection,
    plot_profiles,
    plot_segments,
    plot_structure,
    visualize_corners,
    visualize_edges,
)
from .plotter_3d import (
    create_interactive_plot,
    export_3d_visualization,
    plot_height_map,
    plot_structure_3d,
    plot_surface,
)

__all__ = [
    # 2D plotting
    "plot_image",
    "plot_structure",
    "plot_segments",
    "plot_comparison",
    "plot_profiles",
    "visualize_edges",
    "visualize_corners",
    "create_masked_overlay",
    "flatten_profile_dict",
    "plot_profile_collection",
    "plot_dual_profile",
    "DualProfileResult",
    "plot_bland_altman",
    "BlandAltmanResult",
    # 3D plotting
    "plot_surface",
    "plot_structure_3d",
    "plot_height_map",
    "create_interactive_plot",
    "export_3d_visualization",
]
