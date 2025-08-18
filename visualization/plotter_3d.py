"""
3D visualization functions for image data and structures.

This module provides functions for creating 3D visualizations of image data,
detected structures, and height maps. It includes functions for surface plots,
interactive visualizations, and 3D exports.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from typing import Optional, Union, Tuple, List, Dict, Any, Callable
import os
from pathlib import Path

from ..core.image import ImageData
from ..core.structure import Structure, Segment
from ..utils.exceptions import VisualizationError


def plot_surface(image_data: Union[ImageData, np.ndarray],
                 title: str = "3D Surface Plot",
                 cmap: str = "viridis",
                 ax: Optional[plt.Axes] = None,
                 pixel_size: Optional[Tuple[float, float]] = None,
                 zlabel: str = "Value",
                 elev: float = 30,
                 azim: float = 30,
                 alpha: float = 1.0,
                 **kwargs) -> plt.Figure:
    """
    Create a 3D surface plot of image data.

    Args:
        image_data: Image data to plot
        title: Plot title
        cmap: Colormap name
        ax: Optional existing 3D axes to plot on
        pixel_size: Pixel size in physical units
        zlabel: Label for z-axis
        elev: Elevation angle in degrees
        azim: Azimuth angle in degrees
        alpha: Transparency of the surface
        **kwargs: Additional arguments passed to plot_surface

    Returns:
        Figure object with 3D surface plot

    Raises:
        VisualizationError: If visualization fails
    """
    try:
        # Extract data array if ImageData object
        data = image_data.data if isinstance(image_data, ImageData) else image_data

        # Get pixel size if available
        if pixel_size is None and isinstance(image_data, ImageData):
            pixel_size = image_data.pixel_size
        else:
            pixel_size = (1.0, 1.0) if pixel_size is None else pixel_size

        # Create figure if needed
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            if not isinstance(ax, Axes3D):
                raise VisualizationError("Provided axis must be a 3D axis")
            fig = ax.figure

        # Create coordinate grids
        height, width = data.shape
        x = np.arange(0, width) * pixel_size[0]
        y = np.arange(0, height) * pixel_size[1]
        X, Y = np.meshgrid(x, y)

        # Plot surface
        surf = ax.plot_surface(X, Y, data, cmap=cmap, alpha=alpha, **kwargs)

        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        # Set labels and title
        ax.set_xlabel("X Position" + (f" ({pixel_size[0]} units/pixel)" if pixel_size[0] != 1.0 else ""))
        ax.set_ylabel("Y Position" + (f" ({pixel_size[1]} units/pixel)" if pixel_size[1] != 1.0 else ""))
        ax.set_zlabel(zlabel)
        ax.set_title(title)

        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)

        # Tidy up
        plt.tight_layout()
        return fig

    except Exception as e:
        raise VisualizationError(f"Failed to create 3D surface plot: {str(e)}")


def plot_structure_3d(structure: Structure,
                      title: str = "3D Structure Visualization",
                      cmap: str = "viridis",
                      ax: Optional[plt.Axes] = None,
                      highlight_segments: bool = True,
                      elev: float = 30,
                      azim: float = 30,
                      alpha: float = 1.0,
                      **kwargs) -> plt.Figure:
    """
    Create a 3D visualization of a structure with optional segment highlighting.

    Args:
        structure: Structure object to visualize
        title: Plot title
        cmap: Colormap name or list of colormaps for segments
        ax: Optional existing 3D axes to plot on
        highlight_segments: Whether to color segments differently
        elev: Elevation angle in degrees
        azim: Azimuth angle in degrees
        alpha: Transparency of the surface
        **kwargs: Additional arguments passed to plot_surface

    Returns:
        Figure object with 3D structure plot

    Raises:
        VisualizationError: If visualization fails
    """
    try:
        # Create figure if needed
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            if not isinstance(ax, Axes3D):
                raise VisualizationError("Provided axis must be a 3D axis")
            fig = ax.figure

        # Get data and coordinates
        data = structure.data.filled(np.nan)  # Fill masked values with NaN
        height, width = data.shape

        x = np.arange(0, width) * structure.pixel_size[0]
        y = np.arange(0, height) * structure.pixel_size[1]
        X, Y = np.meshgrid(x, y)

        # If no segments or not highlighting segments, plot entire structure
        if not highlight_segments or not structure.segments:
            surf = ax.plot_surface(X, Y, data, cmap=cmap, alpha=alpha, **kwargs)
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        else:
            # Plot each segment with a different color
            segment_cmaps = plt.cm.get_cmap('tab10', len(structure.segments))

            for i, segment in enumerate(structure.segments):
                segment_data = segment.data.filled(np.nan)

                # Get segment color
                if isinstance(cmap, str):
                    segment_cmap = cmap
                elif isinstance(cmap, list) and i < len(cmap):
                    segment_cmap = cmap[i]
                else:
                    # Use color from tab10
                    rgba = segment_cmaps(i)
                    segment_cmap = cm.colors.ListedColormap([rgba])

                # Plot segment
                surf = ax.plot_surface(X, Y, segment_data, cmap=segment_cmap,
                                       alpha=alpha, label=f"Segment {segment.id}", **kwargs)

            # Create proxy artists for legend
            proxy_artists = []
            for i in range(len(structure.segments)):
                rgba = segment_cmaps(i)
                proxy = plt.Rectangle((0, 0), 1, 1, fc=rgba)
                proxy_artists.append(proxy)

            # Add legend
            ax.legend(proxy_artists, [f"Segment {i}" for i in range(len(structure.segments))],
                      loc='upper right', bbox_to_anchor=(1.2, 1.0))

        # Set labels and title
        ax.set_xlabel(f"X Position ({structure.pixel_size[0]} units/pixel)")
        ax.set_ylabel(f"Y Position ({structure.pixel_size[1]} units/pixel)")
        ax.set_zlabel("Value")
        ax.set_title(title)

        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)

        # Tidy up
        plt.tight_layout()
        return fig

    except Exception as e:
        raise VisualizationError(f"Failed to create 3D structure plot: {str(e)}")


def plot_height_map(height_data: Union[ImageData, np.ndarray, Structure],
                    title: str = "Height Map",
                    colorbar_label: str = "Height (µm)",
                    cmap: str = "viridis",
                    ax: Optional[plt.Axes] = None,
                    pixel_size: Optional[Tuple[float, float]] = None,
                    add_contours: bool = True,
                    n_contours: int = 10,
                    contour_color: str = 'black',
                    contour_alpha: float = 0.5,
                    elev: float = 30,
                    azim: float = 30,
                    **kwargs) -> plt.Figure:
    """
    Create a 3D height map visualization with optional contour lines.

    Args:
        height_data: Height data to visualize
        title: Plot title
        colorbar_label: Label for the colorbar
        cmap: Colormap name
        ax: Optional existing 3D axes to plot on
        pixel_size: Pixel size in physical units
        add_contours: Whether to add contour lines
        n_contours: Number of contour lines
        contour_color: Color for contour lines
        contour_alpha: Transparency for contour lines
        elev: Elevation angle in degrees
        azim: Azimuth angle in degrees
        **kwargs: Additional arguments passed to plot_surface

    Returns:
        Figure object with 3D height map

    Raises:
        VisualizationError: If visualization fails
    """
    try:
        # Extract data based on input type
        if isinstance(height_data, Structure):
            data = height_data.data.filled(np.nan)
            ps = height_data.pixel_size
        elif isinstance(height_data, ImageData):
            data = height_data.data
            ps = height_data.pixel_size
        else:
            data = height_data
            ps = (1.0, 1.0) if pixel_size is None else pixel_size

        # Create figure if needed
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            if not isinstance(ax, Axes3D):
                raise VisualizationError("Provided axis must be a 3D axis")
            fig = ax.figure

        # Create coordinate grids
        height, width = data.shape
        x = np.arange(0, width) * ps[0]
        y = np.arange(0, height) * ps[1]
        X, Y = np.meshgrid(x, y)

        # Plot surface
        surf = ax.plot_surface(X, Y, data, cmap=cmap, **kwargs)

        # Add contour lines if requested
        if add_contours:
            # Calculate contour levels
            z_min, z_max = np.nanmin(data), np.nanmax(data)
            levels = np.linspace(z_min, z_max, n_contours)

            # Add contour lines on the surface
            contour = ax.contour(X, Y, data, levels=levels, colors=contour_color,
                                 alpha=contour_alpha, linestyles='solid')

        # Add colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label(colorbar_label)

        # Set labels and title
        ax.set_xlabel(f"X Position ({ps[0]} units/pixel)")
        ax.set_ylabel(f"Y Position ({ps[1]} units/pixel)")
        ax.set_zlabel("Height")
        ax.set_title(title)

        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)

        # Tidy up
        plt.tight_layout()
        return fig

    except Exception as e:
        raise VisualizationError(f"Failed to create height map: {str(e)}")


def create_interactive_plot(image_data: Union[ImageData, np.ndarray, Structure],
                            title: str = "Interactive 3D Plot",
                            cmap: str = "viridis",
                            pixel_size: Optional[Tuple[float, float]] = None,
                            **kwargs) -> plt.Figure:
    """
    Create an interactive 3D plot that can be rotated and zoomed.

    This function simply sets up the plot with appropriate defaults
    for interactive use. The returned figure can be shown with plt.show().

    Args:
        image_data: Data to visualize
        title: Plot title
        cmap: Colormap name
        pixel_size: Pixel size in physical units
        **kwargs: Additional arguments passed to plot_surface

    Returns:
        Figure object with interactive 3D plot

    Raises:
        VisualizationError: If visualization fails
    """
    try:
        # Extract data based on input type
        if isinstance(image_data, Structure):
            data = image_data.data.filled(np.nan)
            ps = image_data.pixel_size
        elif isinstance(image_data, ImageData):
            data = image_data.data
            ps = image_data.pixel_size
        else:
            data = image_data
            ps = (1.0, 1.0) if pixel_size is None else pixel_size

        # Create figure with high DPI for better interactivity
        fig = plt.figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        # Create coordinate grids
        height, width = data.shape
        x = np.arange(0, width) * ps[0]
        y = np.arange(0, height) * ps[1]
        X, Y = np.meshgrid(x, y)

        # Use lower resolution for smoother interaction
        stride = max(1, min(height, width) // 100)  # Reduce points for smoother rotation

        # Plot surface
        surf = ax.plot_surface(X, Y, data, cmap=cmap, rstride=stride, cstride=stride, **kwargs)

        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        # Set labels and title
        ax.set_xlabel(f"X Position ({ps[0]} units/pixel)")
        ax.set_ylabel(f"Y Position ({ps[1]} units/pixel)")
        ax.set_zlabel("Value")
        ax.set_title(title)

        # Set initial view
        ax.view_init(elev=30, azim=30)

        # Add note about interactivity
        fig.text(0.5, 0.01, "Click and drag to rotate. Right-click and drag to zoom.",
                 ha='center', fontsize=10, style='italic')

        # Tidy up
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.05)  # Make room for the note

        return fig

    except Exception as e:
        raise VisualizationError(f"Failed to create interactive plot: {str(e)}")


def export_3d_visualization(image_data: Union[ImageData, np.ndarray, Structure],
                            filename: Union[str, Path],
                            title: str = "3D Visualization",
                            cmap: str = "viridis",
                            format: str = "png",
                            dpi: int = 300,
                            views: List[Tuple[float, float]] = [(30, 30), (0, 0), (0, 90), (90, 0)],
                            **kwargs) -> List[str]:
    """
    Export multiple views of a 3D visualization to image files.

    Args:
        image_data: Data to visualize
        filename: Base filename for exports (without extension)
        title: Plot title
        cmap: Colormap name
        format: Output image format ('png', 'jpg', 'pdf', etc.)
        dpi: Resolution in dots per inch
        views: List of (elevation, azimuth) angle pairs
        **kwargs: Additional arguments passed to plot_surface

    Returns:
        List of saved filenames

    Raises:
        VisualizationError: If export fails
    """
    try:
        # Extract data based on input type
        if isinstance(image_data, Structure):
            data = image_data.data.filled(np.nan)
            ps = image_data.pixel_size
        elif isinstance(image_data, ImageData):
            data = image_data.data
            ps = image_data.pixel_size
        else:
            data = image_data
            ps = (1.0, 1.0)

        # Ensure filename has no extension
        filename = Path(filename)
        base_path = filename.parent
        base_name = filename.stem

        # Create output directory if it doesn't exist
        os.makedirs(base_path, exist_ok=True)

        # List to store saved filenames
        saved_files = []

        # Export each view
        for i, (elev, azim) in enumerate(views):
            # Create figure
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Create coordinate grids
            height, width = data.shape
            x = np.arange(0, width) * ps[0]
            y = np.arange(0, height) * ps[1]
            X, Y = np.meshgrid(x, y)

            # Plot surface
            surf = ax.plot_surface(X, Y, data, cmap=cmap, **kwargs)

            # Add colorbar
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

            # Set labels and title
            view_name = f"Elevation: {elev}°, Azimuth: {azim}°"
            ax.set_xlabel(f"X Position ({ps[0]} units/pixel)")
            ax.set_ylabel(f"Y Position ({ps[1]} units/pixel)")
            ax.set_zlabel("Value")
            ax.set_title(f"{title} - {view_name}")

            # Set viewing angle
            ax.view_init(elev=elev, azim=azim)

            # Save figure
            output_file = base_path / f"{base_name}_view{i + 1}.{format}"
            plt.savefig(output_file, dpi=dpi, format=format, bbox_inches='tight')
            plt.close(fig)

            saved_files.append(str(output_file))

        return saved_files

    except Exception as e:
        raise VisualizationError(f"Failed to export 3D visualization: {str(e)}")