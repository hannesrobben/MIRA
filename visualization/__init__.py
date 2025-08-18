"""
Visualization module for the image processing library.

This module provides functionality for visualizing image data,
structures, and analysis results in both 2D and 3D formats.
"""

from .plotter_2d import (
    plot_image, plot_structure, plot_segments, plot_comparison,
    plot_profiles, visualize_edges, visualize_corners, create_masked_overlay
)

from .plotter_3d import (
    plot_surface, plot_structure_3d, plot_height_map,
    create_interactive_plot, export_3d_visualization
)

__all__ = [
    # 2D plotting
    'plot_image', 'plot_structure', 'plot_segments', 'plot_comparison',
    'plot_profiles', 'visualize_edges', 'visualize_corners', 'create_masked_overlay',

    # 3D plotting
    'plot_surface', 'plot_structure_3d', 'plot_height_map',
    'create_interactive_plot', 'export_3d_visualization'
]