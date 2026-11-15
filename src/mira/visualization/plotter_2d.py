"""
2D visualization functions for image data and structures.

This module provides functions for creating 2D visualizations of image data,
detected structures, structure segments, and comparative visualizations.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.colors import ListedColormap

from ..core.image import ImageData
from ..core.structure import Structure
from ..utils.exceptions import VisualizationError


def old_plot_image(
    image_data: ImageData | np.ndarray,
    title: str = "Image Data",
    cmap: str = "viridis",
    ax: plt.Axes | None = None,
    colorbar: bool = True,
    scale_bar: bool = False,
    pixel_size: tuple[float, float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
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
        raise VisualizationError(f"Failed to plot image: {e!s}")


def plot_image(
    image_data: ImageData | np.ndarray,
    title: str = "Image Data",
    cmap: str = "viridis",
    ax: plt.Axes | None = None,
    colorbar: bool = True,
    scale_bar: bool = False,
    pixel_size: tuple[float, float] | None = None,
    pixel_size_unit: str = "m",
    physical_axes: bool = False,
    clip_percentile: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    colorbar_label: str | None = None,
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

        if "dpi" in kwargs:
            dpi_value = kwargs.pop("dpi")
            fig.set_dpi(dpi_value)
        font_sz = 12
        if "font_size" in kwargs:
            font_sz = kwargs.pop("font_size")
            plt.tick_params(axis="both", which="major", labelsize=font_sz)

        # --- Plot ---
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, **kwargs)
        ax.set_title(title)

        # --- Axis labels ---
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=font_sz)
        elif physical_axes and pixel_size is not None:
            ax.set_xlabel(f"x [{pixel_size_unit}]", fontsize=font_sz)

        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=font_sz)
        elif physical_axes and pixel_size is not None:
            ax.set_ylabel(f"y [{pixel_size_unit}]", fontsize=font_sz)

        # --- Colorbar ---
        if colorbar:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if colorbar_label is not None:
                cbar.set_label(colorbar_label, fontsize=font_sz)
                cbar.ax.tick_params(labelsize=font_sz)

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
                fontsize=font_sz,
                zorder=6,
            )

        plt.tight_layout()
        return fig

    except Exception as e:
        raise VisualizationError(f"Failed to plot image: {e!s}")


@dataclass
class HeatmapProfileResult:
    """
    Container for the outputs of `plot_image_with_profile`.

    Bundling the two independent figures/axes plus the extracted profile
    in one object (instead of a plain tuple) keeps the call site readable
    and leaves room to add fields later - e.g. a fitted peak position -
    without breaking existing code that only accesses the attributes it
    needs. This mirrors how `Structure`/`Segment`/`ImageData` are already
    modeled as dataclasses elsewhere in this library.

    Attributes:
        heatmap_fig:      Figure containing the 2D heatmap.
        heatmap_ax:       Axes of the heatmap.
        profile_fig:      Figure containing the 1D profile plot.
        profile_ax:       Axes of the profile plot.
        profile_data:     The extracted 1D profile values.
        profile_index:    Row/column index the profile was extracted at
                          (resolved from "mid" if that was passed in).
        profile_axis:     "x" or "y", as passed to the function.
    """

    heatmap_fig: plt.Figure
    heatmap_ax: plt.Axes
    profile_fig: plt.Figure
    profile_ax: plt.Axes
    profile_data: np.ndarray
    profile_index: int
    profile_axis: str


def _style_axes(ax: plt.Axes, fontsize: int) -> None:
    """
    Re-apply a consistent font-size hierarchy to an axes' *already-set*
    title, axis labels, and tick labels: title at `fontsize`, axis labels
    two points smaller, tick labels four points smaller.

    Call this last, after titles/labels have already been set by the
    plotting calls above - it only touches font size, not the text
    itself. Centralising the sizing rule here (instead of repeating it at
    every `set_title`/`set_xlabel`/`set_ylabel` call) means every plot in
    this module can be kept visually consistent by tuning one function,
    and any future plotting function can reuse it too.

    Args:
        ax:       Axes to restyle.
        fontsize: Base font size (used for the title).
    """
    ax.set_title(ax.get_title(), fontsize=fontsize)
    ax.set_xlabel(ax.get_xlabel(), fontsize=max(fontsize - 2, 1))
    ax.set_ylabel(ax.get_ylabel(), fontsize=max(fontsize - 2, 1))
    ax.tick_params(axis="both", labelsize=max(fontsize - 4, 1))


def plot_image_with_profile(
    image_data: ImageData | np.ndarray,
    profile_axis: str = "x",
    profile_location: str | int = "mid",
    pixel_size: tuple[float, float] | None = None,
    pixel_size_unit: str = "m",
    cmap: str = "viridis",
    physical_axes: bool = True,
    clip_percentile: bool | tuple[float, float] = True,
    vmin: float | None = None,
    vmax: float | None = None,
    heatmap_title: str = "",
    profile_title: str = "",
    xlabel: str | None = None,
    ylabel: str | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    profile_xlabel: str | None = None,
    profile_ylabel: str | None = None,
    profile_color: str = "tab:blue",
    show_profile_line: bool = True,
    line_color: str = "red",
    line_style: str = "-",
    line_width: float = 2.0,
    line_alpha: float = 0.8,
    fontsize: int = 18,
    heatmap_figsize: tuple[float, float] = (8, 6),
    profile_figsize: tuple[float, float] = (10, 6),
    dpi: int = 300,
    save_path_heatmap: str | None = None,
    save_path_profile: str | None = None,
    show_plot: bool = False,
) -> HeatmapProfileResult:
    """
    Plot a 2D heatmap and an independent 1D profile plot for the same data.

    The profile is extracted along a single row ("x") or column ("y") at
    `profile_location`, using the same extraction logic as
    `plot_structure_comparison` (`_extract_profile`). The exact row/column
    used is marked on the heatmap with a reference line, so the two plots
    stay visually linked even though they are separate figures that can
    be saved and used independently.

    Args:
        image_data:          Image data to plot. Either an `ImageData`
                             instance (pixel_size is taken from it unless
                             overridden) or a raw 2D array.
        profile_axis:        "x" extracts a horizontal profile (fixed
                             row, varies along columns); "y" extracts a
                             vertical profile (fixed column, varies along
                             rows). Same convention as `get_profile` on
                             `ImageData` and as `plot_structure_comparison`.
        profile_location:    Row/column index to extract the profile at,
                             or "mid" for the array center.
        pixel_size:          (dy, dx) pixel size in physical units.
                             Falls back to `image_data.pixel_size` if an
                             `ImageData` was passed, else (1.0, 1.0).
        pixel_size_unit:     Physical unit string, e.g. "m", "µm", "nm".
        cmap:                 Colormap for the heatmap.
        physical_axes:       If True, heatmap axes and the profile x-axis
                             use physical units; if False, pixel indices.
        clip_percentile:     If True, clip the heatmap colormap range to
                             the 1st-99th percentile of the data. Pass a
                             custom (low, high) percentile tuple to
                             override the interval, or False to disable
                             (full data range unless vmin/vmax are given).
        vmin:                 Explicit colormap minimum. Overrides
                             `clip_percentile` if given.
        vmax:                 Explicit colormap maximum. Overrides
                             `clip_percentile` if given.
        heatmap_title:       Title of the heatmap.
        profile_title:       Title of the profile plot. Defaults to
                             "Profile at row/column <index>".
        xlabel:               Heatmap x-axis label override.
        ylabel:               Heatmap y-axis label override.
        colorbar:             Whether to add a colorbar to the heatmap.
        colorbar_label:      Colorbar label. Also used as the default
                             profile y-axis label (same physical
                             quantity), unless `profile_ylabel` is given.
        profile_xlabel:      Profile x-axis label override.
        profile_ylabel:      Profile y-axis label override. Defaults to
                             `colorbar_label`.
        profile_color:        Line color of the profile curve.
        show_profile_line:   If True, draw a reference line on the
                             heatmap at the row/column the profile was
                             taken from.
        line_color:           Color of the reference line on the heatmap.
        line_style:           Linestyle of the reference line.
        line_width:           Width of the reference line.
        line_alpha:           Opacity of the reference line.
        fontsize:             Base font size for both plots. Axis labels
                             are 2pt smaller, tick labels 4pt smaller -
                             see `_style_axes`.
        heatmap_figsize:      Figure size of the heatmap.
        profile_figsize:      Figure size of the profile plot.
        dpi:                   Resolution used when saving either plot.
        save_path_heatmap:    Optional path to save the heatmap to.
        save_path_profile:    Optional path to save the profile plot to.
        show_plot:             If True, call `plt.show()` before returning.

    Returns:
        `HeatmapProfileResult` with both figures/axes and the extracted
        profile data.

    Raises:
        VisualizationError: If plotting or profile extraction fails.
    """
    try:
        # --- Data extraction ---
        data = image_data.data if isinstance(image_data, ImageData) else image_data

        if pixel_size is None and isinstance(image_data, ImageData):
            pixel_size = image_data.pixel_size
        if pixel_size is None:
            pixel_size = (1.0, 1.0)
        dy, dx = pixel_size

        # --- Resolve percentile clipping (same convention as
        # plot_structure_comparison, kept consistent on purpose) ---
        if isinstance(clip_percentile, tuple):
            percentile_low, percentile_high = clip_percentile
            use_percentile_clip = True
        else:
            percentile_low, percentile_high = 1, 99
            use_percentile_clip = bool(clip_percentile)

        eff_vmin, eff_vmax = vmin, vmax
        if use_percentile_clip:
            if eff_vmin is None:
                eff_vmin = float(np.nanpercentile(data, percentile_low))
            if eff_vmax is None:
                eff_vmax = float(np.nanpercentile(data, percentile_high))

        # --- Extract the profile once; reuse its index for the
        # reference line, so heatmap and profile can never disagree
        # about which row/column was actually used. ---
        profile_data, idx = _extract_profile(data, profile_axis, profile_location)

        # --- Heatmap (independent figure) ---
        fig_heatmap, ax_heatmap = plt.subplots(figsize=heatmap_figsize)
        plot_image(
            data,
            ax=ax_heatmap,
            cmap=cmap,
            pixel_size=pixel_size,
            pixel_size_unit=pixel_size_unit,
            physical_axes=physical_axes,
            vmin=eff_vmin,
            vmax=eff_vmax,
            title=heatmap_title,
            xlabel=xlabel,
            ylabel=ylabel,
            colorbar=colorbar,
            colorbar_label=colorbar_label,
        )

        if show_profile_line:
            if profile_axis == "x":
                line_pos = idx * dy if physical_axes else idx
                ax_heatmap.axhline(
                    line_pos,
                    color=line_color,
                    linestyle=line_style,
                    linewidth=line_width,
                    alpha=line_alpha,
                )
            else:
                line_pos = idx * dx if physical_axes else idx
                ax_heatmap.axvline(
                    line_pos,
                    color=line_color,
                    linestyle=line_style,
                    linewidth=line_width,
                    alpha=line_alpha,
                )

        # Restyle fonts last, after plot_image/axhline have set their
        # own title/labels, so `fontsize` always wins.
        _style_axes(ax_heatmap, fontsize)
        if colorbar and ax_heatmap.images:
            im = ax_heatmap.images[0]
            if im.colorbar is not None:
                im.colorbar.ax.tick_params(labelsize=max(fontsize - 4, 1))
                im.colorbar.ax.yaxis.label.set_fontsize(max(fontsize - 2, 1))

        plt.tight_layout()
        if save_path_heatmap:
            fig_heatmap.savefig(
                save_path_heatmap, dpi=dpi, bbox_inches="tight", facecolor="white"
            )

        # --- Profile plot (independent figure) ---
        fig_profile, ax_profile = plt.subplots(figsize=profile_figsize)

        step = dx if profile_axis == "x" else dy
        x_coords = (
            np.arange(len(profile_data)) * step
            if physical_axes
            else np.arange(len(profile_data))
        )
        ax_profile.plot(
            x_coords, profile_data, color=profile_color, linewidth=2.5, alpha=0.9
        )
        ax_profile.grid(True, alpha=0.3, linestyle=":")

        ax_profile.set_title(profile_title)
        # location_kind = "row" if profile_axis == "x" else "column"
        # ax_profile.set_title(profile_title or f"Profile at {location_kind} {idx}")

        if profile_xlabel is not None:
            ax_profile.set_xlabel(profile_xlabel)
        elif physical_axes:
            axis_letter = "x" if profile_axis == "x" else "y"
            ax_profile.set_xlabel(f"{axis_letter} [{pixel_size_unit}]")
        else:
            ax_profile.set_xlabel(
                "column index" if profile_axis == "x" else "row index"
            )

        if profile_ylabel is not None:
            ax_profile.set_ylabel(profile_ylabel)
        elif colorbar_label is not None:
            ax_profile.set_ylabel(colorbar_label)

        _style_axes(ax_profile, fontsize)

        plt.tight_layout()
        if save_path_profile:
            fig_profile.savefig(
                save_path_profile, dpi=dpi, bbox_inches="tight", facecolor="white"
            )

        if show_plot:
            plt.show()

        return HeatmapProfileResult(
            heatmap_fig=fig_heatmap,
            heatmap_ax=ax_heatmap,
            profile_fig=fig_profile,
            profile_ax=ax_profile,
            profile_data=profile_data,
            profile_index=idx,
            profile_axis=profile_axis,
        )

    except Exception as e:
        if isinstance(e, VisualizationError):
            raise
        raise VisualizationError(f"Failed to plot image with profile: {e!s}")


def plot_structure(
    structure: Structure,
    title: str = "Detected Structure",
    cmap: str = "viridis",
    ax: plt.Axes | None = None,
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
        raise VisualizationError(f"Failed to plot structure: {e!s}")


def plot_segments(
    structure: Structure,
    title: str = "Structure Segments",
    cmap: str = "tab20",
    ax: plt.Axes | None = None,
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
        raise VisualizationError(f"Failed to plot segments: {e!s}")


def plot_comparison(
    image1: ImageData | np.ndarray,
    image2: ImageData | np.ndarray,
    title1: str = "Image 1",
    title2: str = "Image 2",
    diff_title: str = "Difference",
    cmap: str = "viridis",
    diff_cmap: str = "coolwarm",
    show_difference: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    fig: plt.Figure | None = None,
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
        raise VisualizationError(f"Failed to plot comparison: {e!s}")


def plot_profiles(
    image_data: ImageData | np.ndarray,
    x_position: int | None = None,
    y_position: int | None = None,
    title: str = "Image Profiles",
    x_profile_label: str = "Horizontal Profile",
    y_profile_label: str = "Vertical Profile",
    fig: plt.Figure | None = None,
    ax1: plt.Axes | None = None,
    ax2: plt.Axes | None = None,
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
        raise VisualizationError(f"Failed to plot profiles: {e!s}")


def plot_profiles_physical(
    *profiles: np.ndarray,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    pixel_pitch: float = 0.0869,
    pitch_unit: str = "µm",
    title: str = "Profile Comparison",
    ylabel: str = "Height [µm]",
    font_size: int = 14,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot one or more already-extracted 1D profiles against a shared
    physical x-axis, using a single pixel pitch to convert sample
    index to physical distance.

    This is a lightweight counterpart to `plot_dual_profile` for the
    common case of N profiles on one shared y-axis (no per-profile
    y-scale alignment). For profiles that live on genuinely different
    scales, use `plot_dual_profile` or `plot_structure_comparison`
    instead.

    Args:
        *profiles: One or more 1D arrays of equal or differing length.
        labels: Legend label per profile, in the same order. Defaults
            to "Profile 1", "Profile 2", ... if omitted.
        pixel_pitch: Physical size of one pixel/sample along the
            profile, in `pitch_unit` (e.g. 8e-2 for 0.08 µm/px).
            NOTE: pass the value already in the target unit - do not
            pass a metre-scale pitch (e.g. 8e-8) here.
        pitch_unit: Unit string used for the x-axis label.
        title: Plot title.
        ylabel: Y-axis label.
        font_size: Base font size for the whole figure. Title uses
            `font_size`, axis labels `font_size - 2`, tick labels and
            legend `font_size - 4` - the same hierarchy `_style_axes`
            applies elsewhere in this module, so figures stay visually
            consistent across the library.
        fig: Optional existing figure to plot on.
        ax: Optional existing axes to plot on.
        **kwargs: Additional arguments passed to each `ax.plot` call.

    Returns:
        Figure object.

    Raises:
        VisualizationError: If no profiles are given or plotting fails.
    """
    try:
        if not profiles:
            raise VisualizationError("At least one profile must be provided")

        if fig is None or ax is None:
            # fig, ax = plt.subplots(dpi=300)
            fig, ax = plt.subplots(figsize=(9, 6))

        if labels is None:
            labels = [f"Profile {i + 1}" for i in range(len(profiles))]
        elif len(labels) != len(profiles):
            raise VisualizationError(
                f"Got {len(profiles)} profiles but {len(labels)} labels"
            )

        if colors is not None and len(colors) != len(profiles):
            raise VisualizationError(
                f"Got {len(profiles)} profiles but {len(colors)} colors"
            )

        for i, (profile, label) in enumerate(zip(profiles, labels)):
            profile = np.asarray(profile)
            x = np.arange(len(profile)) * pixel_pitch
            color = colors[i] if colors is not None else None
            ax.plot(x, profile, label=label, color=color, **kwargs)

        ax.set_xlabel(f"Position [{pitch_unit}]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=max(font_size - 4, 1))

        _style_axes(ax, font_size)

        plt.tight_layout()
        return fig

    except Exception as e:
        if isinstance(e, VisualizationError):
            raise
        raise VisualizationError(f"Failed to plot profiles: {e!s}")


def visualize_edges(
    image_data: ImageData | np.ndarray,
    threshold: float | None = None,
    title: str = "Edge Detection",
    cmap: str = "gray",
    ax: plt.Axes | None = None,
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
        raise VisualizationError(f"Failed to visualize edges: {e!s}")


def visualize_corners(
    image_data: ImageData | np.ndarray,
    threshold: float = 0.01,
    min_distance: int = 5,
    title: str = "Corner Detection",
    cmap: str = "gray",
    ax: plt.Axes | None = None,
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
        raise VisualizationError(f"Failed to visualize corners: {e!s}")


def _extract_profile(
    data: np.ndarray,
    axis: str,
    location: str | int,
) -> tuple[np.ndarray, int]:
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
            raise VisualizationError(
                f"Profile row index {index} out of bounds (0-{ny - 1})."
            )
        return data[index, :], index
    else:
        index = nx // 2 if location == "mid" else int(location)
        if not 0 <= index < nx:
            raise VisualizationError(
                f"Profile column index {index} out of bounds (0-{nx - 1})."
            )
        return data[:, index], index


def _ensure_range_contains_data(
    y_min: float,
    y_max: float,
    d_min: float,
    d_max: float,
    padding: float,
) -> tuple[float, float]:
    """
    Expand a y-axis range just enough to fully contain a data interval,
    plus a small breathing-room margin, without needlessly shrinking or
    shifting a range that already fits.

    This is the shared safety net behind every profile-alignment mode
    (`same_y_lim`, `align_levels`, `align_zero`) in
    `plot_structure_comparison`. Each mode first decides *where* zero, or
    the background/foreground levels, should sit on the axis - that
    computation alone says nothing about whether the actual curve fits
    inside the result. Think of it like a camera that has already been
    aimed and focused (the alignment decision) but still needs a final
    check that the whole subject is inside the frame before the shutter
    fires; this function is that check, applied independently to the top
    and bottom edge.

    Args:
        y_min, y_max: Proposed axis limits, e.g. from a zero- or
                      level-alignment computation.
        d_min, d_max: Actual min/max of the data that will be plotted on
                      this axis.
        padding:      Extra headroom to add, as a fraction of the axis
                      span, on whichever side(s) needed expanding. Left
                      untouched (no padding added) on a side that already
                      comfortably contains the data.

    Returns:
        Tuple (new_min, new_max) that satisfies new_min <= d_min and
        new_max >= d_max (with `padding` extra margin on any expanded
        side).
    """
    span = y_max - y_min
    # Fallback scale for the padding term if the proposed range is
    # degenerate (e.g. a flat structure with zero contrast).
    pad = padding * span if span > 0 else padding * max(abs(d_max), abs(d_min), 1.0)

    if d_min < y_min:
        y_min = d_min - pad
    if d_max > y_max:
        y_max = d_max + pad
    return y_min, y_max


def _align_zero_multi(
    axes: list[plt.Axes],
    profile_data_list: list[np.ndarray],
    padding: float = 0.08,
) -> None:
    """
    Align the zero line of several y-axes so they sit at the same vertical
    position in the figure, while guaranteeing every axis fully contains
    its own profile data.

    This generalizes the common "dual-zero" trick (used for two twinned
    y-axes) to an arbitrary number of axes: the axis with the most extreme
    zero-position ratio sets the target ratio, and every axis is then
    rescaled (range preserved, only the min/max split point shifts) to
    match that ratio. `_ensure_range_contains_data` is then applied per
    axis so that shifting the split point can never clip the curve it
    carries - the previous version skipped this check, which is why
    peaks near an axis edge could disappear behind the frame.

    Args:
        axes:              List of matplotlib Axes objects sharing the
                            same x-axis (typically created via repeated
                            calls to `ax.twinx()`).
        profile_data_list: The 1D profile array actually plotted on each
                            axis, in the same order as `axes`. Used only
                            to verify/expand the final range, not to
                            change the target zero ratio.
        padding:            Extra headroom fraction passed through to
                            `_ensure_range_contains_data`.
    """
    ratios = []
    for ax in axes:
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ratio = abs(y_min) / y_range if y_range != 0 else 0.0
        ratios.append(ratio)

    target_ratio = max(ratios)

    for ax, profile in zip(axes, profile_data_list):
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        new_min = -target_ratio * y_range
        new_max = (1 - target_ratio) * y_range

        d_min = float(np.nanmin(profile))
        d_max = float(np.nanmax(profile))
        new_min, new_max = _ensure_range_contains_data(
            new_min, new_max, d_min, d_max, padding
        )

        ax.set_ylim(new_min, new_max)


def _compute_robust_levels(data: np.ndarray) -> tuple[float, float]:
    """
    Estimate robust background/foreground reference levels of a structure.

    Rather than using the raw min/max of the data (which single noisy
    pixels or edge artifacts can easily distort), this bipartitions the
    pixel population with Otsu's method and takes the median of each
    resulting class. The median is insensitive to a small fraction of
    outlier pixels, making it a much more stable "reference level" than
    an extreme value.

    Args:
        data: Array (any shape) containing a roughly bimodal distribution
              of values, e.g. background vs. structure pixels.

    Returns:
        Tuple (background_level, foreground_level) as floats, with
        background_level < foreground_level.

    Raises:
        VisualizationError: If Otsu thresholding fails, e.g. because the
            data has (near-)zero variance and is not bimodal.
    """
    from skimage.filters import threshold_otsu

    try:
        threshold = threshold_otsu(data)
    except ValueError as e:
        raise VisualizationError(f"Could not determine robust reference levels: {e}")

    background = float(np.median(data[data <= threshold]))
    foreground = float(np.median(data[data > threshold]))
    return background, foreground


def _align_levels_multi(
    axes: list[plt.Axes],
    data_list: list[np.ndarray],
    profile_data_list: list[np.ndarray],
    level_margin: float = 0.1,
    padding: float = 0.08,
) -> list[tuple[float, float]]:
    """
    Align profile y-axes using robust background/foreground levels.

    This generalizes `_align_zero_multi` from a single anchor point
    (zero) to two robust anchor points per structure - a background and
    a foreground level, found via `_compute_robust_levels` - and places
    both at the same fixed fractional height on every axis. The effect
    is comparable to a two-point instrument calibration: because both
    reference points are themselves robust (median of an Otsu class,
    not an extreme value), noise or edge artifacts elsewhere in the
    signal cannot pull the visual alignment off.

    A robust *level* is deliberately not the same as the data's actual
    extremum - e.g. noisy pixels can easily poke out above the
    foreground median. Placing the levels at `level_margin` alone can
    therefore still clip such peaks; `_ensure_range_contains_data` is
    applied per axis afterwards to widen the range just enough (plus a
    little `padding`) to keep the whole curve on screen, only where
    needed.

    Args:
        axes:              List of Axes sharing the same x-axis (e.g.
                            created via repeated `ax.twinx()` calls), one
                            per structure.
        data_list:         Full 2D array for each structure, in the same
                            order as `axes`. The full array is used (not
                            just the 1D profile) since it gives far more
                            pixels per class and thus a much more stable
                            median estimate.
        profile_data_list: The 1D profile array actually plotted on each
                            axis, in the same order as `axes`. Used only
                            to verify/expand the final range.
        level_margin:      Fractional distance from the axis edges at
                            which the background/foreground levels are
                            placed (e.g. 0.1 places them at 10% / 90% of
                            the axis height).
        padding:           Extra headroom fraction passed through to
                            `_ensure_range_contains_data`.

    Returns:
        The per-structure (background, foreground) levels, so the caller
        can optionally draw them as reference lines on the plot.

    Raises:
        VisualizationError: If robust level detection fails for any
            structure.
    """
    levels = [_compute_robust_levels(data) for data in data_list]
    frac_lo, frac_hi = level_margin, 1.0 - level_margin

    for ax, (lo, hi), profile in zip(axes, levels, profile_data_list):
        span = hi - lo
        if span <= 0:
            continue  # Degenerate structure (no contrast) - leave autoscaled.
        scale = span / (frac_hi - frac_lo)
        new_min = lo - frac_lo * scale
        new_max = new_min + scale

        d_min = float(np.nanmin(profile))
        d_max = float(np.nanmax(profile))
        new_min, new_max = _ensure_range_contains_data(
            new_min, new_max, d_min, d_max, padding
        )

        ax.set_ylim(new_min, new_max)

    return levels


def _resolve_style_mapping(
    value: dict[str, str] | list[str] | None,
    labels: list[str],
    default_cycle: list[str],
) -> dict[str, str]:
    """
    Resolve a per-curve styling argument (color, linestyle, ...) into a
    label -> value mapping, accepting three equally convenient input
    shapes so callers never have to build the mapping by hand.

    Args:
        value:         Either a dict keyed by label (labels missing from
                       the dict fall back to `default_cycle`), a list
                       applied to `labels` in order (cycled if shorter
                       than `labels`), or None to use `default_cycle`
                       for every label.
        labels:        Ordered list of curve labels to resolve a value
                       for.
        default_cycle: Fallback sequence, cycled as needed. Must be
                       non-empty.

    Returns:
        Dictionary mapping every entry in `labels` to a resolved value.
    """
    if isinstance(value, dict):
        return {
            label: value.get(label, default_cycle[i % len(default_cycle)])
            for i, label in enumerate(labels)
        }
    if isinstance(value, list) and value:
        return {label: value[i % len(value)] for i, label in enumerate(labels)}
    return {
        label: default_cycle[i % len(default_cycle)] for i, label in enumerate(labels)
    }


def flatten_profile_dict(
    nested_profiles: dict[str, dict[str, np.ndarray]],
    separator: str = " - ",
) -> dict[str, np.ndarray]:
    """
    Flatten a two-level {group: {sub_label: profile}} dictionary into a
    single {combined_label: profile} dictionary, ready for
    `plot_profile_collection`.

    This is the shape naturally produced when collecting profiles across
    several structures/processors in a loop, each contributing several
    profiles of its own - e.g. one outer entry per processor, with one
    inner entry per pixel location the profile was cut at.

    Args:
        nested_profiles: Two-level dictionary, e.g.
                          `{"sensofar": {"Height profile 650": arr, ...},
                          "developed": {...}}`.
        separator:        String used to join the outer and inner key
                          into the flat label, e.g.
                          "sensofar - Height profile 650".

    Returns:
        Flat dictionary suitable for `plot_profile_collection`.

    Raises:
        VisualizationError: If `nested_profiles` is empty, or if any of
            its values is not itself a dictionary (pass a flat dict
            directly to `plot_profile_collection` in that case).
    """
    if not nested_profiles:
        raise VisualizationError("Cannot flatten an empty profile dictionary.")

    flat: dict[str, np.ndarray] = {}
    for group_label, sub_profiles in nested_profiles.items():
        if not isinstance(sub_profiles, dict):
            raise VisualizationError(
                f"Expected a nested dict of profiles for '{group_label}', "
                f"got {type(sub_profiles).__name__}. Pass a flat dict "
                "directly to plot_profile_collection instead."
            )
        for sub_label, profile in sub_profiles.items():
            flat[f"{group_label}{separator}{sub_label}"] = profile

    return flat


def plot_profile_collection(
    profiles: dict[str, np.ndarray],
    pixel_size: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
    pixel_size_unit: str = "µm",
    profile_axis: str = "x",
    colors: dict[str, str] | list[str] | None = None,
    linestyles: dict[str, str] | list[str] | None = None,
    linewidth: float = 2.0,
    alpha: float = 0.9,
    title: str = "Profile Comparison",
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend_loc: str = "best",
    fontsize: int = 14,
    figsize: tuple[float, float] = (10, 6),
    grid: bool = True,
    ax: plt.Axes | None = None,
    dpi: int = 300,
    save_path: str | None = None,
    show_plot: bool = False,
) -> plt.Figure:
    """
    Overlay an arbitrary number of 1D profiles on a single shared y-axis.

    Use this when every profile represents the same physical quantity in
    the same unit (e.g. several height profiles in µm) - a single shared
    axis makes their absolute values directly comparable at a glance.
    For two profiles of *different* quantities or scales that still need
    to be compared shape-for-shape (e.g. two phase profiles with
    different calibration factors), use `plot_dual_profile` instead,
    which plots on two independently-scaled, zero- or level-aligned
    axes.

    Args:
        profiles:          Flat dictionary of {label: 1D array}. Build
                           this with `flatten_profile_dict` if you are
                           starting from a nested {group: {sub_label:
                           profile}} structure.
        pixel_size:        (dy, dx) pixel size in physical units,
                           applied to every profile, OR a dict giving a
                           per-profile (dy, dx) (useful when profiles
                           come from structures with slightly different
                           pixel sizes). None plots against sample index
                           instead of a physical x-axis.
        pixel_size_unit:   Physical unit string, e.g. "µm". Only used
                           when `pixel_size` is given.
        profile_axis:      "x" (step size = dx) or "y" (step size =
                           dy). Must match how the profiles were
                           originally extracted.
        colors:            Per-curve line color: a dict keyed by label,
                           a list applied in `profiles` order (cycled if
                           shorter), or None for the default color
                           cycle.
        linestyles:        Per-curve linestyle, same three accepted
                           shapes as `colors`. Defaults to solid lines.
        linewidth:         Line width, shared by all curves.
        alpha:             Line opacity, shared by all curves.
        title:             Plot title.
        xlabel:            X-axis label override. Auto-derived from
                           `profile_axis`/`pixel_size_unit` if omitted.
        ylabel:            Y-axis label. Left blank if omitted - set
                           this explicitly, since profiles of different
                           physical quantities should not usually share
                           one axis in the first place.
        legend_loc:        Legend location, forwarded to `ax.legend`.
        fontsize:          Base font size - see `_style_axes`.
        figsize:           Figure size, ignored if `ax` is given.
        grid:              Whether to draw a light dotted grid.
        ax:                Optional existing axes to plot on, e.g. to
                           embed this as one panel of a larger figure.
        dpi:                Resolution used when saving.
        save_path:         Optional path to save the figure to.
        show_plot:          If True, call `plt.show()` before returning.

    Returns:
        Figure object (the parent of `ax`, if `ax` was given).

    Raises:
        VisualizationError: If `profiles` is empty or plotting fails.
    """
    try:
        if not profiles:
            raise VisualizationError("Cannot plot an empty profile collection.")

        labels = list(profiles.keys())
        default_colors = [f"C{i}" for i in range(10)]
        resolved_colors = _resolve_style_mapping(colors, labels, default_colors)
        resolved_styles = _resolve_style_mapping(linestyles, labels, ["-"])

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        for label, profile_data in profiles.items():
            profile_data = np.asarray(profile_data)

            curve_pixel_size = (
                pixel_size.get(label) if isinstance(pixel_size, dict) else pixel_size
            )

            if curve_pixel_size is not None:
                dy, dx = curve_pixel_size
                step = dx if profile_axis == "x" else dy
                x_coords = np.arange(len(profile_data)) * step
            else:
                x_coords = np.arange(len(profile_data))

            ax.plot(
                x_coords,
                profile_data,
                color=resolved_colors[label],
                linestyle=resolved_styles[label],
                linewidth=linewidth,
                alpha=alpha,
                label=label,
            )

        if grid:
            ax.grid(True, alpha=0.3, linestyle=":")

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif pixel_size is not None:
            axis_letter = "x" if profile_axis == "x" else "y"
            ax.set_xlabel(f"{axis_letter} [{pixel_size_unit}]")
        else:
            ax.set_xlabel("sample index")

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        ax.set_title(title)
        ax.legend(loc=legend_loc)
        _style_axes(ax, fontsize)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        if show_plot:
            plt.show()

        return fig

    except Exception as e:
        if isinstance(e, VisualizationError):
            raise
        raise VisualizationError(f"Failed to plot profile collection: {e!s}")


@dataclass
class DualProfileResult:
    """
    Container for the outputs of `plot_dual_profile`.

    Bundles the figure and both axes (instead of a plain tuple) so the
    call site stays readable and there is room to add fields later
    without breaking existing code - the same reasoning behind
    `HeatmapProfileResult` above.

    Attributes:
        fig:    The figure containing both axes.
        ax_1:   Axes carrying `profile_1` (left y-axis).
        ax_2:   Axes carrying `profile_2` (right y-axis, `ax_1.twinx()`).
        levels: Per-profile (background, foreground) levels used for
                alignment. Only set when `align_mode="levels"`.
    """

    fig: plt.Figure
    ax_1: plt.Axes
    ax_2: plt.Axes
    levels: tuple[tuple[float, float], tuple[float, float]] | None = None


def plot_dual_profile(
    profile_1: np.ndarray,
    profile_2: np.ndarray,
    label_1: str = "Profile 1",
    label_2: str = "Profile 2",
    color_1: str = "tab:blue",
    color_2: str = "tab:red",
    pixel_size: tuple[float, float] | None = None,
    pixel_size_unit: str = "µm",
    profile_axis: str = "x",
    align_mode: str = "zero",
    data_1: np.ndarray | None = None,
    data_2: np.ndarray | None = None,
    level_margin: float = 0.1,
    padding: float = 0.08,
    show_level_lines: bool = False,
    title: str = "Profile Comparison",
    xlabel: str | None = None,
    ylabel_1: str | None = None,
    ylabel_2: str | None = None,
    linewidth: float = 2.5,
    alpha: float = 0.9,
    fontsize: int = 14,
    figsize: tuple[float, float] = (10, 6),
    grid: bool = True,
    dpi: int = 300,
    save_path: str | None = None,
    show_plot: bool = False,
) -> DualProfileResult:
    """
    Plot two 1D profiles of potentially different scale on one shared
    x-axis, each on its own y-axis, visually aligned so their shapes
    become directly comparable.

    This is the two-curve counterpart to `plot_structure_comparison`'s
    multi-axis profile row, built on the very same alignment machinery
    (`_align_zero_multi` / `_align_levels_multi`) so both stay in sync
    as that machinery evolves, instead of duplicating the alignment
    logic here.

    Args:
        profile_1, profile_2: The two 1D profile arrays to compare.
        label_1, label_2:      Legend labels for each curve.
        color_1, color_2:      Line (and matching axis/tick) color for
                               each curve.
        pixel_size:            (dy, dx) pixel size in physical units,
                               shared by both profiles. None plots
                               against sample index.
        pixel_size_unit:       Physical unit string, e.g. "µm".
        profile_axis:          "x" or "y" - selects dx or dy as the
                               step size from `pixel_size`.
        align_mode:            How the two y-axes are aligned:
                               - "zero":   Zero sits at the same height
                                 on both axes (`_align_zero_multi`).
                                 Good default when both profiles
                                 oscillate around a meaningful zero.
                               - "levels": Robust background/foreground
                                 levels (per-profile Otsu class median,
                                 `_align_levels_multi`) sit at the same
                                 height on both axes. Better when the
                                 two quantities use different
                                 calibration factors and zero is not
                                 the meaningful reference point.
                                 Requires `data_1` and `data_2`.
                               - "none":   Both axes autoscale
                                 independently.
        data_1, data_2:        Full 2D arrays backing each profile,
                               needed only for `align_mode="levels"`
                               (more pixels than the 1D profile give a
                               far more stable robust-level estimate).
        level_margin:          Fractional axis position of the
                               background/foreground levels. Only used
                               for `align_mode="levels"`.
        padding:               Headroom fraction kept around each
                               profile's own data range, regardless of
                               alignment mode.
        show_level_lines:      If True and `align_mode="levels"`, draw
                               dashed reference lines at the detected
                               levels.
        title:                  Plot title.
        xlabel:                 X-axis label override.
        ylabel_1, ylabel_2:     Y-axis label for each curve. Defaults
                               to `label_1`/`label_2`.
        linewidth:              Line width, shared by both curves.
        alpha:                  Line opacity, shared by both curves.
        fontsize:               Base font size - see `_style_axes`.
        figsize:                Figure size.
        grid:                   Whether to draw a light dotted grid.
        dpi:                     Resolution used when saving.
        save_path:              Optional path to save the figure to.
        show_plot:               If True, call `plt.show()` before
                                returning.

    Returns:
        `DualProfileResult` with the figure, both axes, and (for
        `align_mode="levels"`) the detected levels.

    Raises:
        VisualizationError: If `align_mode` is invalid, `align_mode=
            "levels"` is used without `data_1`/`data_2`, or plotting
            fails for any other reason.
    """
    try:
        if align_mode not in ("zero", "levels", "none"):
            raise VisualizationError(
                f"Invalid align_mode '{align_mode}'. Expected 'zero', "
                "'levels', or 'none'."
            )
        if align_mode == "levels" and (data_1 is None or data_2 is None):
            raise VisualizationError(
                "align_mode='levels' requires the full 2D arrays data_1 "
                "and data_2 to estimate robust reference levels."
            )

        profile_1 = np.asarray(profile_1)
        profile_2 = np.asarray(profile_2)

        if pixel_size is not None:
            dy, dx = pixel_size
            step = dx if profile_axis == "x" else dy
            x_1 = np.arange(len(profile_1)) * step
            x_2 = np.arange(len(profile_2)) * step
        else:
            x_1 = np.arange(len(profile_1))
            x_2 = np.arange(len(profile_2))

        fig, ax_1 = plt.subplots(figsize=figsize)
        ax_2 = ax_1.twinx()

        ax_1.plot(
            x_1,
            profile_1,
            color=color_1,
            linewidth=linewidth,
            alpha=alpha,
            label=label_1,
        )
        ax_2.plot(
            x_2,
            profile_2,
            color=color_2,
            linewidth=linewidth,
            alpha=alpha,
            label=label_2,
        )

        ax_1.set_ylabel(ylabel_1 or label_1, color=color_1)
        ax_1.tick_params(axis="y", labelcolor=color_1)
        ax_2.set_ylabel(ylabel_2 or label_2, color=color_2)
        ax_2.tick_params(axis="y", labelcolor=color_2)

        if xlabel is not None:
            ax_1.set_xlabel(xlabel)
        elif pixel_size is not None:
            axis_letter = "x" if profile_axis == "x" else "y"
            ax_1.set_xlabel(f"{axis_letter} [{pixel_size_unit}]")
        else:
            ax_1.set_xlabel("sample index")

        if grid:
            ax_1.grid(True, alpha=0.3, linestyle=":")

        ax_1.set_title(title)

        levels = None
        if align_mode == "zero":
            _align_zero_multi([ax_1, ax_2], [profile_1, profile_2], padding=padding)
        elif align_mode == "levels":
            level_pair = _align_levels_multi(
                [ax_1, ax_2],
                [data_1, data_2],
                [profile_1, profile_2],
                level_margin=level_margin,
                padding=padding,
            )
            levels = (level_pair[0], level_pair[1])
            if show_level_lines:
                for ax, (lo, hi), color in zip(
                    (ax_1, ax_2), level_pair, (color_1, color_2)
                ):
                    ax.axhline(lo, color=color, linestyle="--", linewidth=1, alpha=0.5)
                    ax.axhline(hi, color=color, linestyle="--", linewidth=1, alpha=0.5)
        # align_mode == "none": leave both axes on their independent
        # autoscaled ranges.

        # --- Combined legend from both axes ---
        lines_1, labels_1 = ax_1.get_legend_handles_labels()
        lines_2, labels_2 = ax_2.get_legend_handles_labels()
        ax_1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

        _style_axes(ax_1, fontsize)
        ax_2.set_ylabel(ax_2.get_ylabel(), color=color_2, fontsize=max(fontsize - 2, 1))
        ax_2.tick_params(axis="y", labelcolor=color_2, labelsize=max(fontsize - 4, 1))

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        if show_plot:
            plt.show()

        return DualProfileResult(fig=fig, ax_1=ax_1, ax_2=ax_2, levels=levels)

    except Exception as e:
        if isinstance(e, VisualizationError):
            raise
        raise VisualizationError(f"Failed to plot dual profile: {e!s}")


def plot_structure_comparison(
    data_list: list[np.ndarray],
    pixel_size_list: list[tuple[float, float]],
    titles: list[str] | None = None,
    cbar_labels: list[str] | None = None,
    x_label_heatmap: list[str | None] | None = None,
    y_label_heatmap: list[str | None] | None = None,
    profile_labels: list[str] | None = None,
    profile_colors: list[str] | None = None,
    x_label_profile: str | None = "X Position",
    y_label_profile: list[str | None] | None = None,
    pixel_size_unit: str = "m",
    profile_axis: str = "x",
    profile_location: str | int = "mid",
    cmap: str = "jet",
    clip_percentile: bool | tuple[float, float] = True,
    vmin: list[float | None] | None = None,
    vmax: list[float | None] | None = None,
    same_y_lim: bool = False,
    align_zero: bool = True,
    align_levels: bool = False,
    align_levels_margin: float = 0.1,
    profile_padding: float = 0.08,
    show_level_lines: bool = True,
    show_center_line: bool = True,
    suptitle: str | None = None,
    figsize: tuple[float, float] | None = None,
    height_ratios: tuple[float, float] = (3, 2.5),
    fontsize: int = 14,
    dpi: int = 300,
    save_path: str | None = None,
    show_plot: bool = False,
) -> tuple[plt.Figure, dict[str, list[plt.Axes]]]:
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
                             identical limits, derived from the combined
                             min/max across all structures (plus
                             `profile_padding` headroom). Mutually
                             exclusive in effect with `align_zero` and
                             `align_levels` (same_y_lim takes
                             precedence).
        align_zero:          If True (and `same_y_lim` is False), align
                             the zero-lines of all profile y-axes so
                             structures can be visually compared even
                             though their value ranges differ.
        align_levels:        If True, align each structure's robust
                             background/foreground levels (see
                             `_align_levels_multi`) at the same relative
                             axis height, instead of just the zero line
                             or a shared absolute range. Takes
                             precedence over `align_zero`, but not over
                             `same_y_lim`.
        align_levels_margin: Fractional axis height at which the
                             background/foreground levels are placed
                             when `align_levels` is True (e.g. 0.1 places
                             them at 10% / 90% of the axis height).
        profile_padding:     Fractional headroom added to a profile
                             y-axis whenever the actual plotted curve
                             would otherwise reach the edge of the range
                             chosen by `same_y_lim`, `align_levels`, or
                             `align_zero`. This is what prevents noisy
                             peaks from being clipped; it leaves
                             already-comfortable axes untouched.
        show_level_lines:    If True and `align_levels` is True, draw a
                             faint dashed horizontal line at each
                             structure's robust background and
                             foreground level, color-matched to its
                             profile line - makes the alignment itself
                             visually inspectable and interpretable.
        show_center_line:   If True, draw a horizontal/vertical reference
                             line on each heatmap marking the location of
                             the extracted profile.
        suptitle:            Optional overall figure title.
        figsize:             Optional explicit figure size. Defaults to a
                             size scaled by the number of structures,
                             tall enough for the profile row to be
                             readable.
        height_ratios:       Relative height of (heatmap row, profile
                             row) in the GridSpec. Increase the second
                             value to give the profile plot more room.
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
            ("titles", titles),
            ("cbar_labels", cbar_labels),
            ("x_label_heatmap", x_label_heatmap),
            ("y_label_heatmap", y_label_heatmap),
            ("profile_labels", profile_labels),
            ("profile_colors", profile_colors),
            ("y_label_profile", y_label_profile),
            ("vmin", vmin),
            ("vmax", vmax),
        ):
            if len(lst) != n:
                raise VisualizationError(
                    f"'{name}' must have exactly {n} entries, got {len(lst)}."
                )

        # --- Resolve percentile clipping settings ---
        if isinstance(clip_percentile, tuple):
            percentile_low, percentile_high = clip_percentile
            use_percentile_clip = True
        else:
            percentile_low, percentile_high = 1, 99
            use_percentile_clip = bool(clip_percentile)

        # --- Figure and layout ---
        # Default height raised from 11 to 12.5, and the profile row given
        # more relative space (default height_ratios changed from 3:2 to
        # 3:2.5), so noisy, high-dynamic-range profile curves have enough
        # vertical room to read without needing to zoom in afterwards.
        if figsize is None:
            figsize = (5.5 * n, 12.5)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            2, n, height_ratios=list(height_ratios), hspace=0.4, wspace=0.35
        )

        heatmap_axes: list[plt.Axes] = [fig.add_subplot(gs[0, i]) for i in range(n)]
        profile_axes: list[plt.Axes] = []
        # Profiles are extracted once in the heatmap loop below and reused
        # in the profile-plot loop, instead of being extracted twice.
        profile_data_list: list[np.ndarray] = []

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
            )
            heatmap_axes[i].set_title(titles[i], fontsize=fontsize)

            # Extract the profile here so we know the exact row/column
            # used, and cache it for reuse in the profile row below
            # (avoids extracting the same profile twice).
            profile_data, idx = _extract_profile(data, profile_axis, profile_location)
            profile_data_list.append(profile_data)
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
            dy, dx = pixel_size_list[i]
            profile_data = profile_data_list[i]

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
            ax.plot(
                x_coords,
                profile_data,
                color=color,
                linewidth=2.5,
                label=profile_labels[i],
                alpha=0.85,
            )
            ax.set_ylabel(y_label_profile[i], color=color, fontsize=fontsize - 2)
            ax.tick_params(axis="y", labelcolor=color, labelsize=fontsize - 4)

        ax_primary.set_xlabel(x_label_profile, fontsize=fontsize - 2)
        ax_primary.tick_params(axis="x", labelsize=fontsize - 4)
        ax_primary.grid(True, alpha=0.3, linestyle=":")
        ax_primary.set_title("Profile Comparison", fontsize=fontsize, pad=20)

        # --- Y-axis scaling across profile axes ---
        # Every branch below guarantees `profile_padding` headroom around
        # the actually plotted data, regardless of alignment mode. This is
        # what previously allowed peaks to be clipped at the axis edge -
        # most visible with `align_levels`, since its range is anchored to
        # robust background/foreground levels rather than the data's own
        # min/max.
        if same_y_lim:
            all_min = min(float(np.nanmin(p)) for p in profile_data_list)
            all_max = max(float(np.nanmax(p)) for p in profile_data_list)
            span = all_max - all_min
            pad = profile_padding * span if span > 0 else profile_padding
            reference_ylim = (all_min - pad, all_max + pad)
            for ax in profile_axes:
                ax.set_ylim(reference_ylim)
        elif align_levels:
            levels = _align_levels_multi(
                profile_axes,
                data_list,
                profile_data_list,
                level_margin=align_levels_margin,
                padding=profile_padding,
            )
            if show_level_lines:
                for ax, (lo, hi), color in zip(profile_axes, levels, profile_colors):
                    ax.axhline(lo, color=color, linestyle="--", linewidth=1, alpha=0.5)
                    ax.axhline(hi, color=color, linestyle="--", linewidth=1, alpha=0.5)
        elif align_zero:
            _align_zero_multi(profile_axes, profile_data_list, padding=profile_padding)

        # --- Combined legend from all profile axes ---
        lines, labels = [], []
        for ax in profile_axes:
            l, lab = ax.get_legend_handles_labels()
            lines += l
            labels += lab
        ax_primary.legend(lines, labels, loc="upper left", fontsize=fontsize - 4)

        if suptitle:
            fig.suptitle(suptitle, fontsize=fontsize + 2, y=0.99)
            # Reserve headroom explicitly: tight_layout() alone does not
            # always account for a suptitle, which could otherwise overlap
            # the top row of heatmaps.
            plt.tight_layout(rect=(0, 0, 1, 0.96))
        else:
            plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")

        if show_plot:
            plt.show()

        return fig, {"heatmap_axes": heatmap_axes, "profile_axes": profile_axes}

    except Exception as e:
        if isinstance(e, VisualizationError):
            raise
        raise VisualizationError(f"Failed to plot structure comparison: {e!s}")


def create_masked_overlay(
    base_image: ImageData | np.ndarray,
    mask: np.ndarray,
    color: str = "red",
    alpha: float = 0.5,
    title: str = "Masked Overlay",
    ax: plt.Axes | None = None,
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
        raise VisualizationError(f"Failed to create masked overlay: {e!s}")
