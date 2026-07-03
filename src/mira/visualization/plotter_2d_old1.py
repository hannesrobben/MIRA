"""
2D visualization functions for image data and structures.

This module provides functions for creating 2D visualizations of image data,
detected structures, structure segments, and comparative visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from typing import Optional, Union, Tuple, List, Dict, Any, Callable

from ..core.image import ImageData
from ..core.structure import Structure, Segment, Background
from ..utils.exceptions import VisualizationError


def old_plot_image(
    image_data: Union[ImageData, np.ndarray],
    title: str = "Image Data",
    cmap: str = "viridis",
    ax: Optional[plt.Axes] = None,
    colorbar: bool = True,
    scale_bar: bool = False,
    pixel_size: Optional[Tuple[float, float]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot a 2D image with optional colorbar and scale bar.

    Args:
        image_data: Image data to plot
        title: Plot title
        cmap: Colormap name
        ax: Optional existing axes to plot on
        colorbar: Whether to add a colorbar
        scale_bar: Whether to add a scale bar
        pixel_size: Pixel size in physical units (needed for scale bar)
        vmin: Minimum value for color scaling
        vmax: Maximum value for color scaling
        **kwargs: Additional arguments passed to imshow

    Returns:
        Figure object

    Raises:
        VisualizationError: If visualization fails
    """
    try:
        # Extract data array if ImageData object
        data = image_data.data if isinstance(image_data, ImageData) else image_data

        # Get pixel size if available
        if pixel_size is None and isinstance(image_data, ImageData):
            pixel_size = image_data.pixel_size

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # Plot image
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        ax.set_title(title)

        # Add colorbar if requested
        if colorbar:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add scale bar if requested
        if scale_bar and pixel_size is not None:
            # Determine a reasonable scale bar size (about 1/5 of the image width)
            img_width_pixels = data.shape[1]
            img_width_units = img_width_pixels * pixel_size[0]

            # Calculate a "nice" round scale bar size
            scale_magnitude = 10 ** np.floor(np.log10(img_width_units / 5))
            scale_length = (
                np.round(img_width_units / 5 / scale_magnitude) * scale_magnitude
            )
            scale_pixels = scale_length / pixel_size[0]

            # Create scale bar
            x_pos = img_width_pixels * 0.05
            y_pos = data.shape[0] * 0.95

            ax.add_patch(
                patches.Rectangle(
                    (x_pos, y_pos),
                    scale_pixels,
                    img_width_pixels * 0.01,
                    linewidth=0,
                    edgecolor="none",
                    facecolor="white",
                )
            )

            # Add scale text
            if scale_length >= 1:
                scale_text = f"{int(scale_length)} units"
            else:
                scale_text = f"{scale_length} units"

            ax.text(
                x_pos + scale_pixels / 2,
                y_pos + img_width_pixels * 0.03,
                scale_text,
                color="white",
                horizontalalignment="center",
                verticalalignment="center",
            )

        # Tidy up
        plt.tight_layout()
        return fig

    except Exception as e:
        raise VisualizationError(f"Failed to plot image: {str(e)}")


def plot_image(
    image_data: Union[ImageData, np.ndarray],
    title: str = "Image Data",
    cmap: str = "viridis",
    ax: Optional[plt.Axes] = None,
    colorbar: bool = True,
    scale_bar: bool = False,
    pixel_size: Optional[Tuple[float, float]] = None,
    pixel_size_unit: str = "m",
    physical_axes: bool = False,
    clip_percentile: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    colorbar_label: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot a 2D image with optional colorbar and scale bar.

    Args:
        image_data:        Image data to plot.
        title:             Plot title.
        cmap:              Colormap name.
        ax:                Optional existing axes to plot on.
        colorbar:          Whether to add a colorbar.
        scale_bar:         Whether to add a scale bar (requires pixel_size).
        pixel_size:        Pixel size as (dy, dx) in physical units (row, col).
                           Follows NumPy shape convention.
        pixel_size_unit:   Physical unit string, e.g. "m", "µm", "nm".
                           Used for axis labels and scale bar text.
                           Unit scaling (e.g. m → µm) is NOT applied automatically;
                           pass values already in the desired unit.
        physical_axes:     If True, axis ticks show physical coordinates
                           (requires pixel_size).
        clip_percentile:   If True and vmin/vmax are not given, clip colormap
                           to the 1st and 99th percentile of the data.
        vmin:              Explicit minimum value for colormap. Overrides
                           clip_percentile.
        vmax:              Explicit maximum value for colormap. Overrides
                           clip_percentile.
        xlabel:            Label for the x-axis.
        ylabel:            Label for the y-axis.
        colorbar_label:    Label for the colorbar.
        **kwargs:          Additional arguments passed to imshow.

    Returns:
        Figure object.

    Raises:
        VisualizationError: If visualization fails.
    """
    try:
        # --- Data extraction ---
        data = image_data.data if isinstance(image_data, ImageData) else image_data

        if pixel_size is None and isinstance(image_data, ImageData):
            pixel_size = image_data.pixel_size

        ny, nx = data.shape[:2]

        # --- Auto vmin/vmax from percentiles ---
        if clip_percentile:
            if vmin is None:
                vmin = float(np.nanpercentile(data, 1))
            if vmax is None:
                vmax = float(np.nanpercentile(data, 99))

        # --- Physical extent for imshow ---
        # extent = [left, right, bottom, top] in physical units.
        # origin='upper' (imshow default) means row 0 is at the top,
        # so bottom > top in pixel rows; we map accordingly.
        extent = None
        if physical_axes and pixel_size is not None:
            dy, dx = pixel_size
            extent = [0, nx * dx, ny * dy, 0]  # [left, right, bottom, top]

        # --- Figure / axes setup ---
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # --- Plot ---
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, **kwargs)
        ax.set_title(title)

        # --- Axis labels ---
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif physical_axes and pixel_size is not None:
            ax.set_xlabel(f"x [{pixel_size_unit}]")

        if ylabel is not None:
            ax.set_ylabel(ylabel)
        elif physical_axes and pixel_size is not None:
            ax.set_ylabel(f"y [{pixel_size_unit}]")

        # --- Colorbar ---
        if colorbar:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if colorbar_label is not None:
                cbar.set_label(colorbar_label)

        # --- Scale bar ---
        if scale_bar and pixel_size is not None:
            dy, dx = pixel_size

            # Image dimensions in physical units
            img_width_phys = nx * dx

            # Round scale bar length to a "nice" value (~1/5 of image width)
            raw_length = img_width_phys / 5
            magnitude = 10 ** np.floor(np.log10(raw_length))
            scale_length_phys = np.round(raw_length / magnitude) * magnitude

            # Scale bar size in axes coordinates (fraction of image)
            scale_frac_x = scale_length_phys / img_width_phys  # width as fraction
            bar_height_frac = 0.012  # thin bar
            text_gap_frac = 0.025  # gap bar → text

            # Position: bottom-left corner of bar (in axes fraction)
            bar_x0 = 0.05
            bar_y0 = 0.92  # near bottom; y-axis in axes coords: 0=bottom, 1=top

            # Use axes.transAxes for coordinate-system independence
            bar = patches.Rectangle(
                (bar_x0, bar_y0),
                scale_frac_x,
                bar_height_frac,
                transform=ax.transAxes,
                linewidth=0,
                edgecolor="none",
                facecolor="white",
                zorder=5,
            )
            ax.add_patch(bar)

            # Scale bar label
            if scale_length_phys >= 1:
                scale_text = f"{int(scale_length_phys)} {pixel_size_unit}"
            else:
                scale_text = f"{scale_length_phys:.2g} {pixel_size_unit}"

            ax.text(
                bar_x0 + scale_frac_x / 2,
                bar_y0 - text_gap_frac,  # above the bar (axes coords, y up)
                scale_text,
                color="white",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=8,
                zorder=6,
            )

        plt.tight_layout()
        return fig

    except Exception as e:
        raise VisualizationError(f"Failed to plot image: {str(e)}")


def plot_structure(
    structure: Structure,
    title: str = "Detected Structure",
    cmap: str = "viridis",
    ax: Optional[plt.Axes] = None,
    highlight_bounds: bool = True,
    highlight_corners: bool = True,
    mark_center: bool = True,
    colorbar: bool = True,
    **kwargs,
) -> plt.Figure:
    """
    Plot a structure with optional highlighting of bounds and corners.

    Args:
        structure: Structure object to plot
        title: Plot title
        cmap: Colormap name
        ax: Optional existing axes to plot on
        highlight_bounds: Whether to highlight structure bounds
        highlight_corners: Whether to highlight structure corners
        mark_center: Whether to mark the structure center
        colorbar: Whether to add a colorbar
        **kwargs: Additional arguments passed to imshow

    Returns:
        Figure object

    Raises:
        VisualizationError: If visualization fails
    """
    try:
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # Plot masked data
        # data = structure.data.filled(np.nan)  # Fill masked values with NaN
        data = structure.data.data  # Fill masked values with NaN
        im = ax.imshow(data, cmap=cmap, **kwargs)
        ax.set_title(title)

        # Add colorbar if requested
        if colorbar:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Highlight bounds if requested
        if highlight_bounds:
            y_slice, x_slice = structure.bounds
            width = x_slice.stop - x_slice.start
            height = y_slice.stop - y_slice.start

            rect = patches.Rectangle(
                (x_slice.start, y_slice.start),
                width,
                height,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)

        # Highlight corners if requested
        if highlight_corners:
            try:
                corners = structure.corner_points
                ax.plot(corners[:, 1], corners[:, 0], "ro", markersize=8)

                # Connect corners with lines
                for i in range(len(corners)):
                    next_idx = (i + 1) % len(corners)
                    ax.plot(
                        [corners[i, 1], corners[next_idx, 1]],
                        [corners[i, 0], corners[next_idx, 0]],
                        "r-",
                        linewidth=1,
                    )
            except Exception:
                # Silently ignore if corner detection fails
                pass

        # Mark center if requested
        if mark_center:
            center_y, center_x = structure.center
            ax.plot(center_x, center_y, "g+", markersize=12, markeredgewidth=2)

        # Tidy up
        plt.tight_layout()
        return fig

    except Exception as e:
        raise VisualizationError(f"Failed to plot structure: {str(e)}")


def plot_segments(
    structure: Structure,
    title: str = "Structure Segments",
    cmap: str = "tab20",
    ax: Optional[plt.Axes] = None,
    label_segments: bool = True,
    colorbar: bool = True,
    **kwargs,
) -> plt.Figure:
    """
    Plot structure segments with different colors.

    Args:
        structure: Structure object with segments
        title: Plot title
        cmap: Colormap name for segment colors
        ax: Optional existing axes to plot on
        label_segments: Whether to add segment labels
        colorbar: Whether to add a colorbar
        **kwargs: Additional arguments passed to imshow

    Returns:
        Figure object

    Raises:
        VisualizationError: If visualization fails
    """
    try:
        # Ensure structure has segments
        if not structure.segments:
            raise VisualizationError("Structure has no segments to plot")

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # Create labeled array for segments
        height, width = structure.data.shape
        segment_array = np.full((height, width), -1, dtype=int)

        # Fill in segment IDs
        for segment in structure.segments:
            segment_array[~segment.data.mask] = segment.id

        # Create colormap with transparent background
        n_segments = len(structure.segments)
        colors = plt.cm.get_cmap(cmap, n_segments + 1)(range(n_segments + 1))
        colors[0, 3] = 0  # Make background transparent
        custom_cmap = ListedColormap(colors)

        # Plot segments
        im = ax.imshow(segment_array, cmap=custom_cmap, **kwargs)
        ax.set_title(title)

        # Add colorbar if requested
        if colorbar:
            cbar = fig.colorbar(
                im, ax=ax, fraction=0.046, pad=0.04, ticks=range(-1, n_segments)
            )
            cbar.set_ticklabels(
                ["Background"] + [f"Segment {i}" for i in range(n_segments)]
            )

        # Add segment labels if requested
        if label_segments:
            for segment in structure.segments:
                # Get center of segment
                y_indices, x_indices = np.where(~segment.data.mask)
                if len(y_indices) > 0:
                    center_y = y_indices.mean()
                    center_x = x_indices.mean()

                    # Add label
                    ax.text(
                        center_x,
                        center_y,
                        str(segment.id),
                        color="white",
                        fontsize=10,
                        ha="center",
                        va="center",
                        bbox=dict(
                            facecolor="black", alpha=0.5, boxstyle="round,pad=0.3"
                        ),
                    )

        # Tidy up
        plt.tight_layout()
        return fig

    except Exception as e:
        raise VisualizationError(f"Failed to plot segments: {str(e)}")


def plot_comparison(
    image1: Union[ImageData, np.ndarray],
    image2: Union[ImageData, np.ndarray],
    title1: str = "Image 1",
    title2: str = "Image 2",
    diff_title: str = "Difference",
    cmap: str = "viridis",
    diff_cmap: str = "coolwarm",
    show_difference: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    fig: Optional[plt.Figure] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot two images side by side with optional difference image.

    Args:
        image1: First image data
        image2: Second image data
        title1: Title for first image
        title2: Title for second image
        diff_title: Title for difference image
        cmap: Colormap for images
        diff_cmap: Colormap for difference image
        show_difference: Whether to show difference image
        vmin: Minimum value for color scaling (original images)
        vmax: Maximum value for color scaling (original images)
        fig: Optional existing figure to plot on
        **kwargs: Additional arguments passed to imshow

    Returns:
        Figure object

    Raises:
        VisualizationError: If visualization fails
    """
    try:
        # Extract data arrays if ImageData objects
        data1 = image1.data if isinstance(image1, ImageData) else image1
        data2 = image2.data if isinstance(image2, ImageData) else image2

        # Check shapes match
        if data1.shape != data2.shape:
            raise VisualizationError(
                f"Image shapes don't match: {data1.shape} vs {data2.shape}"
            )

        # Create figure
        if fig is None:
            if show_difference:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            else:
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        else:
            axes = fig.axes
            if show_difference and len(axes) < 3:
                raise VisualizationError("Not enough axes in provided figure")
            if not show_difference and len(axes) < 2:
                raise VisualizationError("Not enough axes in provided figure")

        # Plot first image
        im1 = axes[0].imshow(data1, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        axes[0].set_title(title1)
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # Plot second image
        im2 = axes[1].imshow(data2, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        axes[1].set_title(title2)
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # Plot difference if requested
        if show_difference:
            # Calculate difference (handle both regular arrays and masked arrays)
            if hasattr(data1, "mask") and hasattr(data2, "mask"):
                # For masked arrays, only compare valid data points
                combined_mask = data1.mask | data2.mask
                diff = np.ma.array(data1 - data2, mask=combined_mask)
            else:
                diff = data1 - data2

            # Calculate symmetric limits for difference colormap
            if vmin is None and vmax is None:
                abs_max = np.nanmax(np.abs(diff))
                diff_vmin, diff_vmax = -abs_max, abs_max
            else:
                diff_vmin, diff_vmax = None, None

            # Plot difference
            im3 = axes[2].imshow(
                diff, cmap=diff_cmap, vmin=diff_vmin, vmax=diff_vmax, **kwargs
            )
            axes[2].set_title(diff_title)
            fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

        # Tidy up
        plt.tight_layout()
        return fig

    except Exception as e:
        raise VisualizationError(f"Failed to plot comparison: {str(e)}")


def plot_profiles(
    image_data: Union[ImageData, np.ndarray],
    x_position: Optional[int] = None,
    y_position: Optional[int] = None,
    title: str = "Image Profiles",
    x_profile_label: str = "Horizontal Profile",
    y_profile_label: str = "Vertical Profile",
    fig: Optional[plt.Figure] = None,
    ax1: Optional[plt.Axes] = None,
    ax2: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot horizontal and vertical profiles through an image.

    Args:
        image_data: Image data to plot profiles from
        x_position: X position for vertical profile (default: middle)
        y_position: Y position for horizontal profile (default: middle)
        title: Overall plot title
        x_profile_label: Label for horizontal profile
        y_profile_label: Label for vertical profile
        fig: Optional existing figure to plot on
        ax1: Optional axis for horizontal profile
        ax2: Optional axis for vertical profile
        **kwargs: Additional arguments passed to plot

    Returns:
        Figure object

    Raises:
        VisualizationError: If visualization fails
    """
    try:
        # Extract data array if ImageData object
        data = image_data.data if isinstance(image_data, ImageData) else image_data

        # Get image dimensions
        height, width = data.shape

        # Use middle positions if not specified
        if x_position is None:
            x_position = width // 2
        if y_position is None:
            y_position = height // 2

        # Validate positions
        if not 0 <= x_position < width:
            raise VisualizationError(
                f"X position {x_position} is outside image width {width}"
            )
        if not 0 <= y_position < height:
            raise VisualizationError(
                f"Y position {y_position} is outside image height {height}"
            )

        # Extract profiles
        x_profile = data[y_position, :]
        y_profile = data[:, x_position]

        # Create figure if needed
        if fig is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        elif ax1 is None or ax2 is None:
            if len(fig.axes) >= 2:
                ax1, ax2 = fig.axes[:2]
            else:
                raise VisualizationError("Not enough axes in provided figure")

        # Plot horizontal profile
        ax1.plot(x_profile, **kwargs)
        ax1.set_title(x_profile_label)
        ax1.set_xlabel("X Position (pixels)")
        ax1.set_ylabel("Value")
        ax1.grid(True, alpha=0.3)

        # Plot vertical profile
        ax2.plot(y_profile, **kwargs)
        ax2.set_title(y_profile_label)
        ax2.set_xlabel("Y Position (pixels)")
        ax2.set_ylabel("Value")
        ax2.grid(True, alpha=0.3)

        # Set overall title
        fig.suptitle(title, fontsize=16)

        # Tidy up
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)  # Make room for suptitle

        return fig

    except Exception as e:
        raise VisualizationError(f"Failed to plot profiles: {str(e)}")


def visualize_edges(
    image_data: Union[ImageData, np.ndarray],
    threshold: Optional[float] = None,
    title: str = "Edge Detection",
    cmap: str = "gray",
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Figure:
    """
    Visualize edges in an image using Sobel edge detection.

    Args:
        image_data: Image data to detect edges in
        threshold: Threshold for edge detection (None for automatic)
        title: Plot title
        cmap: Colormap for edge image
        ax: Optional existing axes to plot on
        **kwargs: Additional arguments passed to imshow

    Returns:
        Figure object

    Raises:
        VisualizationError: If visualization fails
    """
    try:
        from scipy import ndimage

        # Extract data array if ImageData object
        data = image_data.data if isinstance(image_data, ImageData) else image_data

        # Compute gradients
        gx = ndimage.sobel(data, axis=0)
        gy = ndimage.sobel(data, axis=1)

        # Compute gradient magnitude
        magnitude = np.sqrt(gx**2 + gy**2)

        # Normalize magnitude
        magnitude = magnitude / magnitude.max()

        # Apply threshold if specified
        if threshold is not None:
            edges = magnitude > threshold
        else:
            # Automatic threshold using Otsu's method
            try:
                from skimage.filters import threshold_otsu

                threshold = threshold_otsu(magnitude)
                edges = magnitude > threshold
            except ImportError:
                # Fallback to simple thresholding
                edges = magnitude > 0.2

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # Plot edges
        im = ax.imshow(edges, cmap=cmap, **kwargs)
        ax.set_title(title)

        # Tidy up
        plt.tight_layout()
        return fig

    except Exception as e:
        raise VisualizationError(f"Failed to visualize edges: {str(e)}")


def visualize_corners(
    image_data: Union[ImageData, np.ndarray],
    threshold: float = 0.01,
    min_distance: int = 5,
    title: str = "Corner Detection",
    cmap: str = "gray",
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Figure:
    """
    Visualize corners in an image using Harris corner detection.

    Args:
        image_data: Image data to detect corners in
        threshold: Threshold for corner detection (relative to maximum)
        min_distance: Minimum distance between corners
        title: Plot title
        cmap: Colormap for corner image
        ax: Optional existing axes to plot on
        **kwargs: Additional arguments passed to imshow

    Returns:
        Figure object with corners highlighted

    Raises:
        VisualizationError: If visualization fails
    """
    try:
        # Import corner detection functions
        try:
            from skimage.feature import corner_harris, corner_peaks
        except ImportError:
            raise VisualizationError("Corner detection requires scikit-image")

        # Extract data array if ImageData object
        data = image_data.data if isinstance(image_data, ImageData) else image_data

        # Compute Harris response
        harris = corner_harris(data)

        # Find corner peaks
        corners = corner_peaks(
            harris, min_distance=min_distance, threshold_rel=threshold
        )

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # Plot image
        ax.imshow(data, cmap=cmap, **kwargs)

        # Plot corners
        if len(corners) > 0:
            ax.plot(corners[:, 1], corners[:, 0], "r.", markersize=8)

        ax.set_title(f"{title} ({len(corners)} corners)")

        # Tidy up
        plt.tight_layout()
        return fig

    except Exception as e:
        raise VisualizationError(f"Failed to visualize corners: {str(e)}")


def _extract_profile(
    data: np.ndarray,
    axis: str,
    location: Union[str, int],
) -> Tuple[np.ndarray, int]:
    """
    Extract a 1D profile line from a 2D array.

    Args:
        data:       2D array to extract the profile from.
        axis:       "x" extracts a horizontal line (varies along columns,
                    fixed row); "y" extracts a vertical line (varies along
                    rows, fixed column).
        location:   Row/column index to extract at, or "mid" for the
                    centre of the array.

    Returns:
        Tuple of (1D profile array, index used to extract it). The index
        is also needed to draw a matching reference line on the 2D heatmap.

    Raises:
        VisualizationError: If axis or location are invalid.
    """
    ny, nx = data.shape[:2]

    if axis not in ("x", "y"):
        raise VisualizationError(f"Invalid profile axis: {axis!r}. Use 'x' or 'y'.")

    if axis == "x":
        index = ny // 2 if location == "mid" else int(location)
        if not 0 <= index < ny:
            raise VisualizationError(f"Profile row index {index} out of bounds (0-{ny - 1}).")
        return data[index, :], index
    else:
        index = nx // 2 if location == "mid" else int(location)
        if not 0 <= index < nx:
            raise VisualizationError(f"Profile column index {index} out of bounds (0-{nx - 1}).")
        return data[:, index], index


def _align_zero_multi(axes: List[plt.Axes]) -> None:
    """
    Align the zero line of several y-axes so they sit at the same vertical
    position in the figure, while preserving each axis' own data range.

    This generalizes the common "dual-zero" trick (used for two twinned
    y-axes) to an arbitrary number of axes: the axis with the most extreme
    zero-position ratio sets the target ratio, and every axis is then
    rescaled (range preserved, only the min/max split point shifts) to
    match that ratio.

    Args:
        axes: List of matplotlib Axes objects sharing the same x-axis
              (typically created via repeated calls to `ax.twinx()`).
    """
    ratios = []
    for ax in axes:
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ratio = abs(y_min) / y_range if y_range != 0 else 0.0
        ratios.append(ratio)

    target_ratio = max(ratios)

    for ax in axes:
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        new_min = -target_ratio * y_range
        new_max = (1 - target_ratio) * y_range
        ax.set_ylim(new_min, new_max)


def plot_structure_comparison(
    data_list: List[np.ndarray],
    pixel_size_list: List[Tuple[float, float]],
    titles: Optional[List[str]] = None,
    cbar_labels: Optional[List[str]] = None,
    x_label_heatmap: Optional[List[Optional[str]]] = None,
    y_label_heatmap: Optional[List[Optional[str]]] = None,
    profile_labels: Optional[List[str]] = None,
    profile_colors: Optional[List[str]] = None,
    x_label_profile: Optional[str] = "X Position",
    y_label_profile: Optional[List[Optional[str]]] = None,
    pixel_size_unit: str = "m",
    profile_axis: str = "x",
    profile_location: Union[str, int] = "mid",
    cmap: str = "jet",
    clip_percentile: Union[bool, Tuple[float, float]] = True,
    vmin: Optional[List[Optional[float]]] = None,
    vmax: Optional[List[Optional[float]]] = None,
    same_y_lim: bool = False,
    align_zero: bool = True,
    show_center_line: bool = True,
    suptitle: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    fontsize: int = 14,
    dpi: int = 300,
    save_path: Optional[str] = None,
    show_plot: bool = False,
) -> Tuple[plt.Figure, Dict[str, List[plt.Axes]]]:
    """
    Compare two or three aligned 2D structures: one heatmap per structure
    plus a shared profile plot below.

    This function assumes the input arrays are already aligned to a common
    canvas (same size, orientation and scale) - e.g. the output of this
    library's alignment pipeline. It does NOT perform any alignment itself.

    Layout (matplotlib GridSpec):
        Row 1: one heatmap per structure, side by side.
        Row 2: a single profile plot spanning the full width, with one
               y-axis per structure (colour-matched to its profile line),
               so that differing physical units (e.g. phase vs. height)
               remain individually readable.

    Robust color scaling for outliers:
        Some measurement techniques produce single-pixel outliers that are
        far outside the main contrast range, which can wash out the
        colormap for the rest of the image. Two complementary mechanisms
        are available, mirroring `matplotlib.pyplot.imshow`'s vmin/vmax:
          - `clip_percentile`: automatically clips the colormap range to a
            percentile interval of each structure's own data (default:
            1st-99th percentile), independently per subplot.
          - `vmin` / `vmax`: explicit per-structure override lists. Any
            entry that is not None takes precedence over the percentile
            clipping for that structure.

    Args:
        data_list:          List of 2 or 3 raw 2D arrays (e.g. phase or
                             height data), one per structure to compare.
        pixel_size_list:    List of (dy, dx) pixel sizes (one per array,
                             same physical unit as `pixel_size_unit`),
                             matching the convention used by `plot_image`.
        titles:             Heatmap titles, one per structure. Defaults to
                             "Structure 1", "Structure 2", ...
        cbar_labels:        Colorbar labels, one per structure.
        x_label_heatmap:    X-axis labels for the heatmaps, one per
                             structure. Defaults to "x [<unit>]".
        y_label_heatmap:    Y-axis labels for the heatmaps, one per
                             structure. Defaults to "y [<unit>]".
        profile_labels:     Legend labels for the profile lines, one per
                             structure. Defaults to the heatmap titles.
        profile_colors:     Line colors for the profiles, one per
                             structure. Defaults to
                             ["tab:blue", "tab:red", "tab:green"].
        x_label_profile:    Shared x-axis label of the profile plot.
        y_label_profile:    Y-axis labels for the profile plot, one per
                             structure. Defaults to `cbar_labels`.
        pixel_size_unit:    Physical unit string used in default axis
                             labels and titles (e.g. "m", "um").
        profile_axis:       "x" for a horizontal profile line (fixed row),
                             "y" for a vertical profile line (fixed
                             column). Same convention for all structures.
        profile_location:   Row/column index for the profile line, or
                             "mid" (default) for the array center. Applied
                             identically to all structures, since they are
                             assumed to already share the same canvas.
        cmap:                Colormap used for all heatmaps.
        clip_percentile:    If True, clip each heatmap's colormap range to
                             the 1st-99th percentile of its own data. Pass
                             a custom (low, high) percentile tuple to
                             override the default interval. Set to False
                             to disable (full data range is used unless
                             vmin/vmax are given).
        vmin:                Optional per-structure list of explicit
                             colormap minimums. None falls back to
                             percentile clipping (or full range) for that
                             entry.
        vmax:                Optional per-structure list of explicit
                             colormap maximums. Same fallback behaviour as
                             `vmin`.
        same_y_lim:          If True, force all profile y-axes to share
                             identical limits (taken from the first
                             structure). Mutually exclusive in effect with
                             `align_zero` (same_y_lim takes precedence).
        align_zero:          If True (and `same_y_lim` is False), align
                             the zero-lines of all profile y-axes so
                             structures can be visually compared even
                             though their value ranges differ.
        show_center_line:   If True, draw a horizontal/vertical reference
                             line on each heatmap marking the location of
                             the extracted profile.
        suptitle:            Optional overall figure title.
        figsize:             Optional explicit figure size. Defaults to a
                             size scaled by the number of structures.
        fontsize:            Base font size for titles and labels.
        dpi:                 Resolution used when saving the figure.
        save_path:           Optional path to save the figure to.
        show_plot:           If True, call `plt.show()` before returning.

    Returns:
        Tuple of (Figure, axes_dict) where axes_dict has keys
        "heatmap_axes" (list of Axes, one per structure) and
        "profile_axes" (list of Axes, one per structure, the first being
        the primary/shared profile axis).

    Raises:
        VisualizationError: If the number of structures is not 2 or 3, if
            input lists have mismatched lengths, or if plotting fails.
    """
    try:
        n = len(data_list)
        if n not in (2, 3):
            raise VisualizationError(
                f"plot_structure_comparison supports 2 or 3 structures, got {n}."
            )
        if len(pixel_size_list) != n:
            raise VisualizationError(
                "data_list and pixel_size_list must have the same length."
            )

        # --- Defaults for all per-structure parameter lists ---
        titles = titles or [f"Structure {i + 1}" for i in range(n)]
        cbar_labels = cbar_labels or [None] * n
        x_label_heatmap = x_label_heatmap or [f"x [{pixel_size_unit}]"] * n
        y_label_heatmap = y_label_heatmap or [f"y [{pixel_size_unit}]"] * n
        profile_labels = profile_labels or list(titles)
        profile_colors = profile_colors or ["tab:blue", "tab:red", "tab:green"][:n]
        y_label_profile = y_label_profile or list(cbar_labels)
        vmin = vmin or [None] * n
        vmax = vmax or [None] * n

        for name, lst in (
            ("titles", titles), ("cbar_labels", cbar_labels),
            ("x_label_heatmap", x_label_heatmap), ("y_label_heatmap", y_label_heatmap),
            ("profile_labels", profile_labels), ("profile_colors", profile_colors),
            ("y_label_profile", y_label_profile), ("vmin", vmin), ("vmax", vmax),
        ):
            if len(lst) != n:
                raise VisualizationError(f"'{name}' must have exactly {n} entries, got {len(lst)}.")

        # --- Resolve percentile clipping settings ---
        if isinstance(clip_percentile, tuple):
            percentile_low, percentile_high = clip_percentile
            use_percentile_clip = True
        else:
            percentile_low, percentile_high = 1, 99
            use_percentile_clip = bool(clip_percentile)

        # --- Figure and layout ---
        if figsize is None:
            figsize = (5.5 * n, 11)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, n, height_ratios=[3, 2], hspace=0.35, wspace=0.35)

        heatmap_axes: List[plt.Axes] = [fig.add_subplot(gs[0, i]) for i in range(n)]
        profile_axes: List[plt.Axes] = []

        # --- Row 1: heatmaps ---
        profile_index = None  # row/column index shared by all heatmaps + profiles
        for i in range(n):
            data = data_list[i]
            dy, dx = pixel_size_list[i]

            # Resolve effective vmin/vmax: explicit override > percentile clip > full range
            eff_vmin, eff_vmax = vmin[i], vmax[i]
            if use_percentile_clip:
                if eff_vmin is None:
                    eff_vmin = float(np.nanpercentile(data, percentile_low))
                if eff_vmax is None:
                    eff_vmax = float(np.nanpercentile(data, percentile_high))

            plot_image(
                data,
                ax=heatmap_axes[i],
                cmap=cmap,
                pixel_size=pixel_size_list[i],
                pixel_size_unit=pixel_size_unit,
                physical_axes=True,
                vmin=eff_vmin,
                vmax=eff_vmax,
                title=titles[i],
                xlabel=x_label_heatmap[i],
                ylabel=y_label_heatmap[i],
                colorbar_label=cbar_labels[i],
                fontsize=fontsize,
            )

            # Extract the profile here so we know the exact row/column used
            profile_data, idx = _extract_profile(data, profile_axis, profile_location)
            if profile_index is None:
                profile_index = idx

            if show_center_line:
                if profile_axis == "x":
                    heatmap_axes[i].axhline(
                        y=idx * dy, color="red", linestyle="-", linewidth=2, alpha=0.8
                    )
                else:
                    heatmap_axes[i].axvline(
                        x=idx * dx, color="red", linestyle="-", linewidth=2, alpha=0.8
                    )

        # --- Row 2: shared profile plot with one y-axis per structure ---
        ax_primary = fig.add_subplot(gs[1, :])

        for i in range(n):
            data = data_list[i]
            dy, dx = pixel_size_list[i]
            profile_data, _ = _extract_profile(data, profile_axis, profile_location)

            # Physical x-coordinates along the profile, using this structure's
            # own pixel size (relevant if pixel sizes differ slightly).
            step = dx if profile_axis == "x" else dy
            x_coords = np.arange(len(profile_data)) * step

            if i == 0:
                ax = ax_primary
            else:
                ax = ax_primary.twinx()
                if i == 2:
                    # Offset the third y-axis outward so it doesn't overlap the second.
                    ax.spines["right"].set_position(("outward", 60))
            profile_axes.append(ax)

            color = profile_colors[i]
            ax.plot(x_coords, profile_data, color=color, linewidth=2.5,
                     label=profile_labels[i], alpha=0.85)
            ax.set_ylabel(y_label_profile[i], color=color, fontsize=fontsize - 2)
            ax.tick_params(axis="y", labelcolor=color, labelsize=fontsize - 4)

        ax_primary.set_xlabel(x_label_profile, fontsize=fontsize - 2)
        ax_primary.tick_params(axis="x", labelsize=fontsize - 4)
        ax_primary.grid(True, alpha=0.3, linestyle=":")
        ax_primary.set_title("Profile Comparison", fontsize=fontsize, pad=20)

        # --- Y-axis scaling across profile axes ---
        if same_y_lim:
            reference_ylim = profile_axes[0].get_ylim()
            for ax in profile_axes:
                ax.set_ylim(reference_ylim)
        elif align_zero:
            _align_zero_multi(profile_axes)

        # --- Combined legend from all profile axes ---
        lines, labels = [], []
        for ax in profile_axes:
            l, lab = ax.get_legend_handles_labels()
            lines += l
            labels += lab
        ax_primary.legend(lines, labels, loc="upper left", fontsize=fontsize - 4)

        if suptitle:
            fig.suptitle(suptitle, fontsize=fontsize + 2, y=0.99)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")

        if show_plot:
            plt.show()

        return fig, {"heatmap_axes": heatmap_axes, "profile_axes": profile_axes}

    except Exception as e:
        if isinstance(e, VisualizationError):
            raise
        raise VisualizationError(f"Failed to plot structure comparison: {str(e)}")


def create_masked_overlay(
    base_image: Union[ImageData, np.ndarray],
    mask: np.ndarray,
    color: str = "red",
    alpha: float = 0.5,
    title: str = "Masked Overlay",
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Figure:
    """
    Create an overlay visualization with a colored mask on a base image.

    Args:
        base_image: Base image data
        mask: Boolean mask (True for overlay areas)
        color: Color for the overlay
        alpha: Transparency of the overlay (0-1)
        title: Plot title
        ax: Optional existing axes to plot on
        **kwargs: Additional arguments passed to imshow

    Returns:
        Figure object with overlay

    Raises:
        VisualizationError: If visualization fails
    """
    try:
        # Extract data array if ImageData object
        data = base_image.data if isinstance(base_image, ImageData) else base_image

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # Plot base image
        ax.imshow(data, cmap="gray", **kwargs)

        # Create mask overlay
        colored_mask = np.zeros((*mask.shape, 4))

        # Convert color string to RGB
        from matplotlib.colors import to_rgba

        rgba = to_rgba(color)

        # Fill in colored_mask where mask is True
        for i in range(3):
            colored_mask[mask, i] = rgba[i]
        colored_mask[mask, 3] = alpha

        # Plot overlay
        ax.imshow(colored_mask)
        ax.set_title(title)

        # Tidy up
        plt.tight_layout()
        return fig

    except Exception as e:
        raise VisualizationError(f"Failed to create masked overlay: {str(e)}")
