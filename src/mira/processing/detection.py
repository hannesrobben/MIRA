"""
Structure detection module.

This module provides functions for detecting structures, edges, and corners
in image data using various methods.
"""

import numpy as np
import numpy.ma as ma
from enum import Enum
from typing import Union, Optional, Tuple, Dict, List, Literal, Any
from dataclasses import dataclass

from ..core.image import ImageData
from ..core.structure import Structure
from ..utils.exceptions import ProcessingError, ValidationError


def detect_structure(
    image_data: Union[ImageData, np.ndarray],
    method: Literal["threshold", "intensity", "adaptive"] = "threshold",
    intensity_image: Optional[Union[ImageData, np.ndarray]] = None,
    pixel_size: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> Structure:
    """
    Detect structure in image data using the specified method.

    Args:
        image_data: Image data for structure detection
        method: Detection method to use
        intensity_image: Optional intensity image (needed for some methods)
        pixel_size: Pixel size in physical units
        **kwargs: Additional parameters for the detection method

    Returns:
        Structure object representing the detected structure

    Raises:
        ProcessingError: If detection fails
        ValidationError: If inputs are invalid
    """
    # Extract numpy array if ImageData object
    data = image_data.data if isinstance(image_data, ImageData) else image_data

    # Get pixel size if available
    if pixel_size is None and isinstance(image_data, ImageData):
        pixel_size = image_data.pixel_size
    else:
        pixel_size = (1.0, 1.0) if pixel_size is None else pixel_size

    # Dispatch to appropriate detection method
    try:
        if method == "threshold":
            mask = threshold_detection(data, **kwargs)
        elif method == "intensity":
            if intensity_image is None:
                raise ValidationError(
                    "Intensity image must be provided for intensity-based detection"
                )

            # Extract intensity data
            intensity_data = (
                intensity_image.data
                if isinstance(intensity_image, ImageData)
                else intensity_image
            )
            mask = intensity_detection(data, intensity_data, **kwargs)
        elif method == "adaptive":
            mask = adaptive_detection(data, **kwargs)
        else:
            raise ValidationError(f"Unknown detection method: {method}")

        # Clean up the mask (remove small objects, fill holes)
        mask = clean_structure_mask(mask, **kwargs)

        # Create masked data for the structure
        masked_data = ma.masked_array(data, mask=~mask)

        # Create and return Structure object
        return Structure(data=masked_data, pixel_size=pixel_size)

    except Exception as e:
        if isinstance(e, (ProcessingError, ValidationError)):
            raise
        raise ProcessingError(f"Structure detection failed: {str(e)}")


def threshold_detection(
    data: np.ndarray,
    threshold_method: Literal["otsu", "manual", "percentile"] = "otsu",
    threshold_value: Optional[float] = None,
    percentile: float = 10.0,
    **kwargs,
) -> np.ndarray:
    """
    Detect structure using thresholding.

    Args:
        data: Image data
        threshold_method: Method for determining threshold
        threshold_value: Manual threshold value (for 'manual' method)
        percentile: Percentile for 'percentile' method
        **kwargs: Additional parameters (ignored)

    Returns:
        Boolean mask (True where structure is detected)

    Raises:
        ProcessingError: If thresholding fails
        ValidationError: If inputs are invalid
    """
    try:
        if threshold_method == "otsu":
            # Use Otsu's method for automatic thresholding
            try:
                from skimage.filters import threshold_otsu

                threshold = threshold_otsu(data)
            except ImportError:
                # Fallback to simple percentile-based threshold
                threshold = np.percentile(data, 90)

        elif threshold_method == "manual":
            if threshold_value is None:
                raise ValidationError(
                    "threshold_value must be provided for manual thresholding"
                )
            threshold = threshold_value

        elif threshold_method == "percentile":
            if not 0 <= percentile <= 100:
                raise ValidationError("percentile must be between 0 and 100")
            threshold = np.percentile(data, percentile)

        else:
            raise ValidationError(f"Unknown threshold method: {threshold_method}")

        # print(threshold)

        # Apply threshold to create mask
        mask = data > threshold

        return mask

    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ProcessingError(f"Threshold detection failed: {str(e)}")


def intensity_detection(
    data: np.ndarray,
    intensity_data: np.ndarray,
    smoothing_sigma: float = 2.0,
    border_width: Optional[int] = None,
    low_threshold_ratio: float = 0.4,
    seed_gap_closing: Optional[int] = None,
    safety_margin_px: int = 0,
    min_size: int = 100,
    **kwargs,
) -> np.ndarray:
    """
    Detect structure using marker-controlled watershed segmentation on the
    intensity image, with hysteresis-thresholded edges for marker seeding.

    Why watershed instead of "find edges, then close + fill":
    The original approach detected edges (Sobel), tried to morphologically
    close small gaps in them, and then filled whatever region that closed
    contour enclosed. This only works if the edge forms an unbroken loop --
    a single gap (a weak-contrast stretch of boundary, a speckle dropout)
    lets "inside" leak into "outside", so `binary_fill_holes` has nothing to
    fill and you're left with disconnected edge fragments.

    Marker-controlled watershed reframes this as a flooding problem instead
    of a contour-closing problem. The (smoothed) intensity gradient is a
    terrain: ridges follow the true structure boundary, valleys are the
    flat interior/background. Two "water sources" flood it simultaneously
    -- one seeded in the outer border (assumption - near border = always
    background), one seeded inside the structure -- and every pixel gets
    claimed by whichever source reaches it first (lowest cumulative
    "elevation" along the path). The result is a complete, filled
    partition of the image by construction, with no separate "fill the
    hole" step to fail, and no assumption about the structure's geometry.

    Why hysteresis thresholding for finding the interior seed (instead of
    thresholding the intensity itself, or a single hard gradient cutoff):
    On real measurement data the interior of a cell and the background
    often sit at nearly the same *intensity level* -- only the boundary
    line differs, so thresholding intensity directly (e.g. Otsu) tends to
    pick out an unrelated bright/dark patch (measurement artifact) rather
    than the structure. And a single global gradient cutoff has the same
    fragility as your original `edge_threshold`: one weak stretch of the
    real boundary (contrast dips below the cutoff for a while, e.g. from a
    local measurement artifact) breaks the loop there and leaks the
    interior to the border. `skimage.filters.apply_hysteresis_threshold`
    (the same idea Canny uses internally) fixes this: a pixel is kept if
    it clears the high threshold OR if it's connected, via other above-low
    pixels, to one that does. A weak-but-continuous stretch of boundary
    survives because it's chained to strong stretches elsewhere, without
    lowering the threshold everywhere (which would let background texture
    through too). This is also why your Canny attempts likely underwhelmed
    Sobel: Canny's hysteresis is only as good as the two threshold values
    it's given, and its defaults aren't scaled to your data's gradient
    range -- here both are derived adaptively from the image itself (Otsu
    for the high threshold, a fraction of it for the low one).

    A second real effect on this kind of grating/cell structure: every
    sub-cell is individually outlined, not just the outer perimeter, so
    "the single largest region enclosed by edges" is just one cell, not
    the whole footprint. The fix is to seed the foreground marker from the
    *union* of every edge-enclosed region that isn't connected to the
    border -- all cells together -- and let watershed grow that union
    outward to the true outer boundary and swallow the thin dividing lines
    between cells, producing one connected filled area.

    If the structure runs off the edge of the frame (breaking the
    background-at-border assumption), a warning is raised.

    Args:
        data: Phase/height data (only used for shape validation; kept for
            API compatibility with `detect_structure`)
        intensity_data: Intensity image used for segmentation
        smoothing_sigma: Gaussian smoothing applied before computing the
            gradient, to suppress speckle/measurement noise.
        border_width: Width (px) of the outer frame used as the
            guaranteed-background marker. Defaults to ~2% of the smaller
            image dimension.
        low_threshold_ratio: Hysteresis low threshold, as a fraction of
            the (Otsu-derived) high threshold. Lower values bridge bigger
            gaps in the boundary but risk pulling in background texture;
            0.3-0.5 is a reasonable range to try.
        seed_gap_closing: Small dilation (px) applied to the hysteresis
            edge mask purely to bridge pixel-level gaps when identifying
            enclosed (non-border-connected) regions for seeding -- this
            does NOT affect the final boundary, which is refined by
            watershed on the un-dilated gradient. Defaults to ~1% of the
            smaller image dimension (at least 2px).
        safety_margin_px: Grows the final mask outward by this many pixels
            (isotropically, via a disk-shaped structuring element) after
            detection. Use this to keep background-only operations safely
            away from the structure boundary -- e.g. `safety_margin_px=10`
            means every "background" pixel is guaranteed at least 10 px
            from the true edge, at every point around the structure (not
            just along straight stretches -- a plain iterated dilation
            with the default cross-shaped structuring element would grow
            a diamond and under-cover the margin near corners, which is
            exactly where a multi-corner structure like yours needs it
            most). 0 (default) disables it -- returns the mask as
            detected.
        min_size: Kept for API compatibility; final cleanup (hole filling,
            largest-component selection) is done by `clean_structure_mask`.
        **kwargs: Additional parameters (ignored)

    Returns:
        Boolean mask (True where structure is -- including the safety
        margin, if set)

    Raises:
        ProcessingError: If detection fails or scikit-image is unavailable
        ValidationError: If inputs are invalid
    """
    try:
        from skimage.filters import sobel, threshold_otsu, apply_hysteresis_threshold
        from skimage.segmentation import watershed
        from skimage.morphology import disk
        from scipy.ndimage import (
            gaussian_filter,
            binary_dilation,
            binary_fill_holes,
            label,
        )
    except ImportError:
        raise ProcessingError(
            "Intensity-based detection requires scikit-image "
            "(marker-controlled watershed, hysteresis thresholding)."
        )

    # Ensure data shapes match
    if data.shape != intensity_data.shape:
        raise ValidationError(
            f"Data shapes don't match: {data.shape} vs {intensity_data.shape}"
        )

    try:
        h, w = intensity_data.shape
        min_dim = min(h, w)
        if border_width is None:
            border_width = max(2, int(round(0.02 * min_dim)))
        if seed_gap_closing is None:
            seed_gap_closing = max(2, int(round(0.01 * min_dim)))

        # 1. Suppress measurement noise before computing the gradient
        smoothed = gaussian_filter(intensity_data.astype(float), sigma=smoothing_sigma)

        # 2. "Elevation map" that watershed floods across -- edges become ridges.
        #    This is the landscape used for the FINAL boundary -- always the
        #    un-dilated, un-thresholded gradient, so the result isn't limited
        #    to the resolution of the binary edge mask below.
        elevation = sobel(smoothed)

        # 3. Guaranteed-background marker: an outer frame (your stated assumption)
        border_mask = np.zeros(intensity_data.shape, dtype=bool)
        border_mask[:border_width, :] = True
        border_mask[-border_width:, :] = True
        border_mask[:, :border_width] = True
        border_mask[:, -border_width:] = True

        # 4. Hysteresis-thresholded edges: robust to locally weak/dashed
        #    stretches of the real boundary (see docstring)
        high_thresh = threshold_otsu(elevation)
        low_thresh = low_threshold_ratio * high_thresh
        edge_mask = apply_hysteresis_threshold(elevation, low_thresh, high_thresh)

        # 5. Foreground marker: the union of every edge-enclosed region that
        #    is NOT connected to the border -- i.e. every individual cell,
        #    combined -- found via a small dilation purely for gap-bridging
        closed_edges = binary_dilation(edge_mask, iterations=seed_gap_closing)
        non_edge = ~closed_edges
        labeled_regions, _ = label(non_edge)
        border_labels = set(np.unique(labeled_regions[border_mask])) - {0}
        foreground_seed = non_edge.copy()
        for bl in border_labels:
            foreground_seed[labeled_regions == bl] = False

        if not foreground_seed.any():
            raise ProcessingError(
                "Could not find any interior seed region -- every "
                "edge-enclosed area is connected to the border. Try "
                "lowering low_threshold_ratio, or check that the "
                "structure doesn't run off the image edge."
            )

        # Sanity check on the border-is-background assumption
        border_overlap_fraction = edge_mask[border_mask].mean()
        if (~foreground_seed | edge_mask)[
            border_mask
        ].all() and border_overlap_fraction > 0.3:
            import warnings

            warnings.warn(
                "The border is densely covered in detected edges -- the "
                "structure may extend past the image edge, which breaks "
                "the background-at-border assumption this method relies on.",
                stacklevel=2,
            )

        # 6. Flood both markers across the true gradient landscape; where
        #    they meet becomes the boundary, and every pixel is assigned
        markers = np.zeros(intensity_data.shape, dtype=np.int32)
        markers[border_mask] = 1  # background
        markers[foreground_seed] = 2  # structure
        labels = watershed(elevation, markers)
        mask = labels == 2

        # 7. Fill any remaining small internal holes and keep the largest
        #    connected component (also done downstream by
        #    `clean_structure_mask`, but kept here so this function returns
        #    a clean, usable mask on its own)
        filled = binary_fill_holes(mask)
        labeled_final, num_final = label(filled)
        if num_final == 0:
            result = filled
        else:
            sizes = np.bincount(labeled_final.ravel())
            sizes[0] = 0
            result = labeled_final == np.argmax(sizes)

        # 8. Optional safety margin: grow the mask outward isotropically so
        #    background-only operations stay clear of the true boundary
        if safety_margin_px > 0:
            result = binary_dilation(result, structure=disk(safety_margin_px))

        return result

    except Exception as e:
        if isinstance(e, (ProcessingError, ValidationError)):
            raise
        raise ProcessingError(f"Intensity-based detection failed: {str(e)}")


def adaptive_detection(
    data: np.ndarray, block_size: int = 35, offset: float = 0.0, **kwargs
) -> np.ndarray:
    """
    Detect structure using adaptive thresholding.

    Args:
        data: Image data
        block_size: Size of blocks for adaptive thresholding
        offset: Offset for threshold value
        **kwargs: Additional parameters (ignored)

    Returns:
        Boolean mask (True where structure is detected)

    Raises:
        ProcessingError: If detection fails
    """
    try:
        # Use skimage's adaptive threshold
        try:
            from skimage.filters import threshold_local

            threshold = threshold_local(data, block_size, offset=offset)
            mask = data > threshold

        except ImportError:
            # Fallback to manual implementation
            from scipy.ndimage import uniform_filter

            # Ensure block_size is odd
            if block_size % 2 == 0:
                block_size += 1

            # Calculate local mean
            local_mean = uniform_filter(data, size=block_size)

            # Apply threshold
            mask = data > (local_mean + offset)

        return mask

    except Exception as e:
        raise ProcessingError(f"Adaptive detection failed: {str(e)}")


def clean_structure_mask(
    mask: np.ndarray,
    remove_small: bool = True,
    min_size: int = 100,
    fill_holes: bool = True,
    keep_largest: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Clean a structure mask by removing small objects and filling holes.

    Args:
        mask: Boolean mask to clean
        remove_small: Whether to remove small objects
        min_size: Minimum size of objects to keep
        fill_holes: Whether to fill holes in the mask
        keep_largest: Whether to keep only the largest connected component
        **kwargs: Additional parameters (ignored)

    Returns:
        Cleaned boolean mask

    Raises:
        ProcessingError: If cleaning fails
    """
    try:
        cleaned_mask = mask.copy()

        # Fill holes
        if fill_holes:
            try:
                from scipy.ndimage import binary_fill_holes

                cleaned_mask = binary_fill_holes(cleaned_mask)
            except Exception as e:
                raise ProcessingError(f"Failed to fill holes: {str(e)}")

        # Label connected components
        try:
            from scipy.ndimage import label

            labeled_array, num_features = label(cleaned_mask)

            # If no features found, return the original mask
            if num_features == 0:
                return mask

            # Keep only the largest component if requested
            if keep_largest:
                # Get the size of each component
                component_sizes = np.bincount(labeled_array.ravel())

                # Skip background (index 0)
                if len(component_sizes) > 1:
                    # Find the largest component (excluding background)
                    largest_component = np.argmax(component_sizes[1:]) + 1

                    # Create a new mask with only the largest component
                    cleaned_mask = labeled_array == largest_component

            # Remove small objects if requested
            elif remove_small:
                try:
                    from skimage.morphology import remove_small_objects

                    cleaned_mask = remove_small_objects(cleaned_mask, min_size=min_size)
                except ImportError:
                    # Manual implementation
                    for i in range(1, num_features + 1):
                        component = labeled_array == i
                        if np.sum(component) < min_size:
                            cleaned_mask[component] = False

        except Exception as e:
            raise ProcessingError(f"Failed to process connected components: {str(e)}")

        return cleaned_mask

    except Exception as e:
        if isinstance(e, ProcessingError):
            raise
        raise ProcessingError(f"Failed to clean structure mask: {str(e)}")


def detect_edges(
    image_data: Union[ImageData, np.ndarray],
    method: Literal["sobel", "canny", "gradient"] = "sobel",
    threshold: Optional[float] = None,
    sigma: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """
    Detect edges in image data.

    Args:
        image_data: Image data for edge detection
        method: Edge detection method
        threshold: Threshold for edge detection (None for automatic)
        sigma: Sigma for Gaussian smoothing
        **kwargs: Additional parameters for detection methods

    Returns:
        Boolean mask of detected edges

    Raises:
        ProcessingError: If edge detection fails
        ValidationError: If inputs are invalid
    """
    # Extract numpy array if ImageData object
    data = image_data.data if isinstance(image_data, ImageData) else image_data

    try:
        if method == "sobel":
            # Use Sobel edge detection
            from scipy import ndimage

            # Apply Gaussian smoothing
            if sigma > 0:
                data = ndimage.gaussian_filter(data, sigma=sigma)

            # Calculate gradients
            sx = ndimage.sobel(data, axis=0)
            sy = ndimage.sobel(data, axis=1)

            # Calculate gradient magnitude
            magnitude = np.hypot(sx, sy)

            # Normalize and threshold
            if threshold is None:
                # Automatic threshold using Otsu's method
                try:
                    from skimage.filters import threshold_otsu

                    threshold = threshold_otsu(magnitude)
                except ImportError:
                    threshold = 0.2 * np.max(magnitude)
            else:
                threshold = threshold * np.max(magnitude)

            edges = magnitude > threshold

        elif method == "canny":
            # Use Canny edge detection
            try:
                from skimage.feature import canny

                edges = canny(data, sigma=sigma, **kwargs)
            except ImportError:
                raise ProcessingError("Canny edge detection requires scikit-image")

        elif method == "gradient":
            # Simple gradient magnitude thresholding
            from scipy import ndimage

            # Calculate gradients
            gy, gx = np.gradient(data)

            # Calculate gradient magnitude
            magnitude = np.sqrt(gx**2 + gy**2)

            # Normalize and threshold
            if threshold is None:
                threshold = 0.2 * np.max(magnitude)
            else:
                threshold = threshold * np.max(magnitude)

            edges = magnitude > threshold

        else:
            raise ValidationError(f"Unknown edge detection method: {method}")

        return edges

    except Exception as e:
        if isinstance(e, (ProcessingError, ValidationError)):
            raise
        raise ProcessingError(f"Edge detection failed: {str(e)}")


def detect_corners(
    image_data: Union[ImageData, np.ndarray],
    method: Literal["harris", "shi_tomasi"] = "harris",
    min_distance: int = 5,
    threshold: float = 0.01,
    **kwargs,
) -> np.ndarray:
    """
    Detect corners in image data.

    Args:
        image_data: Image data for corner detection
        method: Corner detection method
        min_distance: Minimum distance between corners
        threshold: Threshold for corner detection
        **kwargs: Additional parameters for detection methods

    Returns:
        Array of corner coordinates, shape (N, 2) with (y, x) format

    Raises:
        ProcessingError: If corner detection fails
        ValidationError: If inputs are invalid
    """
    # Extract numpy array if ImageData object
    data = image_data.data if isinstance(image_data, ImageData) else image_data

    try:
        # Import corner detection functions
        try:
            from skimage.feature import corner_harris, corner_shi_tomasi, corner_peaks
        except ImportError:
            raise ProcessingError("Corner detection requires scikit-image")

        if method == "harris":
            # Use Harris corner detection
            corner_response = corner_harris(data, **kwargs)
            corners = corner_peaks(
                corner_response, min_distance=min_distance, threshold_rel=threshold
            )

        elif method == "shi_tomasi":
            # Use Shi-Tomasi corner detection
            corner_response = corner_shi_tomasi(data, **kwargs)
            corners = corner_peaks(
                corner_response, min_distance=min_distance, threshold_rel=threshold
            )

        else:
            raise ValidationError(f"Unknown corner detection method: {method}")

        return corners

    except Exception as e:
        if isinstance(e, (ProcessingError, ValidationError)):
            raise
        raise ProcessingError(f"Corner detection failed: {str(e)}")
