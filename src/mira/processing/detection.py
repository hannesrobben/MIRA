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

        print(threshold)

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
    edge_threshold: float = 0.3,
    min_size: int = 100,
    **kwargs,
) -> np.ndarray:
    """
    Detect structure using edge detection on intensity data.

    Args:
        data: Phase/height data
        intensity_data: Intensity image used for edge detection
        edge_threshold: Threshold for edge detection
        min_size: Minimum structure size in pixels
        **kwargs: Additional parameters (ignored)

    Returns:
        Boolean mask (True where structure is detected)

    Raises:
        ProcessingError: If detection fails
    """
    try:
        # Ensure data shapes match
        if data.shape != intensity_data.shape:
            raise ValidationError(
                f"Data shapes don't match: {data.shape} vs {intensity_data.shape}"
            )

        # Calculate edges using Canny edge detection
        try:
            from skimage.feature import canny

            edges = canny(intensity_data, sigma=1.0)
        except ImportError:
            # Fallback to Sobel edge detection
            from scipy import ndimage

            sx = ndimage.sobel(intensity_data, axis=0)
            sy = ndimage.sobel(intensity_data, axis=1)
            edges = np.hypot(sx, sy) > edge_threshold * np.max(np.hypot(sx, sy))

        # Fill enclosed regions
        try:
            from skimage.morphology import binary_closing, remove_small_objects
            from scipy.ndimage import binary_fill_holes

            # Close gaps in edges
            closed_edges = binary_closing(edges)

            # Fill holes
            filled = binary_fill_holes(closed_edges)

            # Remove small objects
            mask = remove_small_objects(filled, min_size=min_size)

        except ImportError:
            # Fallback using scipy
            from scipy.ndimage import binary_closing, binary_fill_holes, label

            # Close gaps in edges
            closed_edges = binary_closing(edges)

            # Fill holes
            filled = binary_fill_holes(closed_edges)

            # Remove small objects
            labeled, num = label(filled)
            sizes = np.bincount(labeled.ravel())
            mask_sizes = sizes > min_size
            mask_sizes[0] = 0  # Don't remove background
            mask = mask_sizes[labeled]

        return mask

    except Exception as e:
        if isinstance(e, ValidationError):
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
