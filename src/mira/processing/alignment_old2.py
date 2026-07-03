"""
Image alignment module.

This module provides functions for aligning structures and images, including
rotation, translation, scaling, and pixel size normalization.
"""

from __future__ import annotations
import numpy as np
import numpy.ma as ma

from typing import TYPE_CHECKING, Dict, List, Literal, Tuple, Optional

import cv2


from ..core.structure import Structure
from ..utils.exceptions import AlignmentError, ValidationError

if TYPE_CHECKING:
    # Nur für Typ-Checker sichtbar, kein echter Import zur Laufzeit
    from ..processor import StructureProcessor


def align_structures(
    source_structure: Structure,
    target_structure: Structure,
    method: Literal["edge", "mask", "feature"] = "edge",
    scaling: Literal["upsampling", "downsampling"] = "downsampling",
    return_transform: bool = False,
    **kwargs,
) -> Union[Tuple[Structure, Dict], Structure]:
    """
    Align a source structure with a target structure.

    Args:
        source_structure: Structure to be aligned
        target_structure: Reference structure to align with
        method: Alignment method
        scaling: Scaling approach
        return_transform: Whether to return transformation info
        **kwargs: Additional parameters for alignment

    Returns:
        Aligned structure or tuple of (aligned structure, transform info)

    Raises:
        AlignmentError: If alignment fails
        ValidationError: If inputs are invalid
    """
    try:
        # Copy structures to avoid modifying originals
        source = Structure(
            data=source_structure.data.copy(), pixel_size=source_structure.pixel_size
        )

        target = Structure(
            data=target_structure.data.copy(), pixel_size=target_structure.pixel_size
        )

        # Initialize transform info
        transform_info = {
            "method": method,
            "scaling": scaling,
            "original_source_shape": source.data.shape,
            "original_target_shape": target.data.shape,
            "original_source_pixel_size": source.pixel_size,
            "original_target_pixel_size": target.pixel_size,
        }

        # Step 1: Normalize pixel size if needed
        if source.pixel_size != target.pixel_size:
            if scaling == "upsampling":
                # Make source match target's pixel size
                source_data, zoom_factors = normalize_pixel_size(
                    source.data.data, source.pixel_size, target.pixel_size
                )

                # Update source data and pixel size
                source.data = ma.masked_array(
                    source_data, mask=~resize_mask(~source.data.mask, zoom_factors)
                )
                source.pixel_size = target.pixel_size

                transform_info["pixel_scaling"] = {
                    "zoom_factors": zoom_factors,
                    "direction": "source_to_target",
                }

            elif scaling == "downsampling":
                # Make target match source's pixel size
                target_data, zoom_factors = normalize_pixel_size(
                    target.data.data, target.pixel_size, source.pixel_size
                )

                # Update target data and pixel size
                target.data = ma.masked_array(
                    target_data, mask=~resize_mask(~target.data.mask, zoom_factors)
                )
                target.pixel_size = source.pixel_size

                transform_info["pixel_scaling"] = {
                    "zoom_factors": zoom_factors,
                    "direction": "target_to_source",
                }

            else:  # 'none'
                # Just record the difference
                transform_info["pixel_scaling"] = {
                    "source_pixel_size": source.pixel_size,
                    "target_pixel_size": target.pixel_size,
                    "no_scaling_applied": True,
                }

        # Step 2: Make sure structures have the same dimensions
        source_data, target_data = pad_to_same_size(source.data.data, target.data.data)

        source_mask = pad_to_same_size(~source.data.mask, ~target.data.mask)[0]

        target_mask = pad_to_same_size(~target.data.mask, ~source.data.mask)[1]

        # Update data with padded versions
        source.data = ma.masked_array(source_data, mask=~source_mask)
        target.data = ma.masked_array(target_data, mask=~target_mask)

        transform_info["padding"] = {"padded_shape": source.data.shape}

        # Step 3: Apply specific alignment method
        if method == "edge":
            aligned_source, method_info = align_by_edges(source, target, **kwargs)
        elif method == "mask":
            aligned_source, method_info = align_by_mask(source, target, **kwargs)
        elif method == "feature":
            aligned_source, method_info = align_by_features(source, target, **kwargs)
        else:
            raise ValidationError(f"Unknown alignment method: {method}")

        # Update transform info
        transform_info.update(method_info)

        # Return result
        if return_transform:
            return aligned_source, transform_info
        else:
            return aligned_source

    except Exception as e:
        if isinstance(e, (AlignmentError, ValidationError)):
            raise
        raise AlignmentError(f"Structure alignment failed: {str(e)}")


def align_by_edges(
    source: Structure,
    target: Structure,
    rotation_method: Literal["hough", "minimize", "radon"] = "hough",
    detect_edges_kwargs: Optional[Dict] = None,
    **kwargs,
) -> Tuple[Structure, Dict]:
    """
    Align structures using edge detection and rotation.

    Args:
        source: Source structure to be aligned
        target: Target structure to align with
        rotation_method: Method for calculating rotation
        detect_edges_kwargs: Parameters for edge detection
        **kwargs: Additional parameters

    Returns:
        Tuple of (aligned structure, method info)

    Raises:
        AlignmentError: If alignment fails
    """
    try:
        from .detection import detect_edges

        # Set default edge detection parameters
        edge_params = {"method": "sobel", "threshold": 0.2}
        if detect_edges_kwargs:
            edge_params.update(detect_edges_kwargs)

        # Detect edges
        source_edges = detect_edges(source.data, **edge_params)
        target_edges = detect_edges(target.data, **edge_params)

        # Calculate rotation
        rotation_angle = calculate_rotation(
            source_edges, target_edges, method=rotation_method, **kwargs
        )

        # Apply rotation to source
        from scipy.ndimage import rotate

        rotated_data = rotate(
            source.data.filled(0),
            rotation_angle,
            reshape=False,
            mode="constant",
            cval=0,
        )

        rotated_mask = (
            rotate(
                ~source.data.mask,
                rotation_angle,
                reshape=False,
                mode="constant",
                cval=0,
            )
            > 0.5
        )  # Threshold to get boolean mask

        # Create aligned structure
        aligned_source = Structure(
            data=ma.masked_array(rotated_data, mask=~rotated_mask),
            pixel_size=source.pixel_size,
        )

        # Center the structures
        aligned_source = center_structures(aligned_source, target)[0]

        # Return aligned structure and info
        method_info = {
            "rotation_angle": rotation_angle,
            "rotation_method": rotation_method,
            "center_adjusted": True,
        }

        return aligned_source, method_info

    except Exception as e:
        raise AlignmentError(f"Edge-based alignment failed: {str(e)}")


def align_by_mask(
    source: Structure, target: Structure, **kwargs
) -> Tuple[Structure, Dict]:
    """
    Align structures using mask-based registration.

    Args:
        source: Source structure to be aligned
        target: Target structure to align with
        **kwargs: Additional parameters

    Returns:
        Tuple of (aligned structure, method info)

    Raises:
        AlignmentError: If alignment fails
    """
    try:
        # Use phase correlation for translation
        try:
            from skimage.registration import phase_cross_correlation
        except ImportError:
            raise AlignmentError("Mask-based alignment requires scikit-image")

        # Get masks
        source_mask = ~source.data.mask
        target_mask = ~target.data.mask

        # Find translation using phase correlation
        shift, error, diffphase = phase_cross_correlation(
            source_mask.astype(float), target_mask.astype(float), **kwargs
        )

        # Apply translation to source
        from scipy.ndimage import shift as apply_shift

        shifted_data = apply_shift(
            source.data.filled(0), shift=shift, mode="constant", cval=0
        )

        shifted_mask = (
            apply_shift(source_mask, shift=shift, mode="constant", cval=0) > 0.5
        )  # Threshold to get boolean mask

        # Create aligned structure
        aligned_source = Structure(
            data=ma.masked_array(shifted_data, mask=~shifted_mask),
            pixel_size=source.pixel_size,
        )

        # Return aligned structure and info
        method_info = {
            "translation_shift": shift.tolist(),
            "correlation_error": float(error),
            "phase_difference": float(diffphase),
        }

        return aligned_source, method_info

    except Exception as e:
        if isinstance(e, AlignmentError):
            raise
        raise AlignmentError(f"Mask-based alignment failed: {str(e)}")


def align_by_features(
    source: Structure, target: Structure, **kwargs
) -> Tuple[Structure, Dict]:
    """
    Align structures using feature-based registration.

    Args:
        source: Source structure to be aligned
        target: Target structure to align with
        **kwargs: Additional parameters

    Returns:
        Tuple of (aligned structure, method info)

    Raises:
        AlignmentError: If alignment fails
    """
    try:
        from .detection import detect_corners

        # Detect corners
        source_corners = detect_corners(source.data, **kwargs)
        target_corners = detect_corners(target.data, **kwargs)

        if len(source_corners) < 3 or len(target_corners) < 3:
            raise AlignmentError(
                "Not enough corners detected for feature-based alignment"
            )

        # Match corners and estimate transform
        try:
            from skimage.transform import estimate_transform
        except ImportError:
            raise AlignmentError("Feature-based alignment requires scikit-image")

        # Simple matching - this is a naive approach that assumes corners
        # are detected in similar order. For a real application, use RANSAC
        # or a more robust matching algorithm.
        n_points = min(
            len(source_corners), len(target_corners), 10
        )  # Limit to 10 points

        # Estimate similarity transform (scale, rotation, translation)
        transform = estimate_transform(
            "similarity", src=source_corners[:n_points], dst=target_corners[:n_points]
        )

        # Apply transform to source
        try:
            from skimage.transform import warp
        except ImportError:
            raise AlignmentError("Feature-based alignment requires scikit-image")

        warped_data = warp(
            source.data.filled(0),
            inverse_map=transform.inverse,
            mode="constant",
            cval=0,
        )

        warped_mask = (
            warp(
                (~source.data.mask).astype(float),
                inverse_map=transform.inverse,
                mode="constant",
                cval=0,
            )
            > 0.5
        )  # Threshold to get boolean mask

        # Create aligned structure
        aligned_source = Structure(
            data=ma.masked_array(warped_data, mask=~warped_mask),
            pixel_size=source.pixel_size,
        )

        # Return aligned structure and info
        method_info = {
            "transform_matrix": transform.params.tolist(),
            "transform_type": "similarity",
            "num_matched_corners": n_points,
        }

        return aligned_source, method_info

    except Exception as e:
        if isinstance(e, AlignmentError):
            raise
        raise AlignmentError(f"Feature-based alignment failed: {str(e)}")


def calculate_rotation(
    source_edges: np.ndarray,
    target_edges: np.ndarray,
    method: Literal["hough", "minimize", "radon"] = "hough",
    **kwargs,
) -> float:
    """
    Calculate rotation angle between two edge images.

    Args:
        source_edges: Source edge image
        target_edges: Target edge image
        method: Method for calculating rotation
        **kwargs: Additional parameters for the method

    Returns:
        Rotation angle in degrees

    Raises:
        AlignmentError: If calculation fails
        ValidationError: If inputs are invalid
    """
    try:
        if method == "hough":
            return calculate_rotation_hough(source_edges, target_edges, **kwargs)
        elif method == "minimize":
            return calculate_rotation_minimize(source_edges, target_edges, **kwargs)
        elif method == "radon":
            return calculate_rotation_radon(source_edges, target_edges, **kwargs)
        else:
            raise ValidationError(f"Unknown rotation calculation method: {method}")

    except Exception as e:
        if isinstance(e, (AlignmentError, ValidationError)):
            raise
        raise AlignmentError(f"Rotation calculation failed: {str(e)}")


def calculate_rotation_hough(
    source_edges: np.ndarray, target_edges: np.ndarray, **kwargs
) -> float:
    """
    Calculate rotation using Hough transform.

    Args:
        source_edges: Source edge image
        target_edges: Target edge image
        **kwargs: Additional parameters

    Returns:
        Rotation angle in degrees

    Raises:
        AlignmentError: If calculation fails
    """
    try:
        # Use skimage's Hough transform
        try:
            from skimage.transform import hough_line, hough_line_peaks
        except ImportError:
            raise AlignmentError("Hough transform requires scikit-image")

        # Get Hough transform for both images
        h1, theta1, d1 = hough_line(source_edges)
        h2, theta2, d2 = hough_line(target_edges)

        # Get dominant lines
        _, angles1, _ = hough_line_peaks(h1, theta1, d1)
        _, angles2, _ = hough_line_peaks(h2, theta2, d2)

        # If no lines found, return 0
        if len(angles1) == 0 or len(angles2) == 0:
            return 0.0

        # Calculate rotation difference
        angle_diff = np.median(angles1) - np.median(angles2)

        # Convert to degrees
        return np.degrees(angle_diff)

    except Exception as e:
        if isinstance(e, AlignmentError):
            raise
        raise AlignmentError(f"Hough transform rotation calculation failed: {str(e)}")


def calculate_rotation_minimize(
    source_edges: np.ndarray,
    target_edges: np.ndarray,
    angle_range: Tuple[float, float] = (-30, 30),
    angle_step: float = 0.5,
    **kwargs,
) -> float:
    """
    Calculate rotation by minimizing difference.

    Args:
        source_edges: Source edge image
        target_edges: Target edge image
        angle_range: Range of angles to try
        angle_step: Step size for angles
        **kwargs: Additional parameters

    Returns:
        Rotation angle in degrees

    Raises:
        AlignmentError: If calculation fails
    """
    try:
        from scipy.ndimage import rotate

        # Ensure source and target have the same shape
        if source_edges.shape != target_edges.shape:
            source_edges, target_edges = pad_to_same_size(source_edges, target_edges)

        # Initialize variables
        best_angle = 0.0
        best_score = -np.inf

        # Try different angles
        angles = np.arange(angle_range[0], angle_range[1] + angle_step, angle_step)

        for angle in angles:
            # Rotate source edges
            rotated = rotate(
                source_edges, angle, reshape=False, mode="constant", cval=0
            )

            # Calculate normalized cross-correlation
            score = np.sum(rotated * target_edges) / np.sqrt(
                np.sum(rotated**2) * np.sum(target_edges**2)
            )

            # Update best angle
            if score > best_score:
                best_score = score
                best_angle = angle

        return best_angle

    except Exception as e:
        raise AlignmentError(f"Minimization rotation calculation failed: {str(e)}")


def calculate_rotation_radon(
    source_edges: np.ndarray,
    target_edges: np.ndarray,
    angle_range: Tuple[float, float] = (-90, 90),
    **kwargs,
) -> float:
    """
    Calculate rotation using Radon transform.

    Args:
        source_edges: Source edge image
        target_edges: Target edge image
        angle_range: Range of angles for transform
        **kwargs: Additional parameters

    Returns:
        Rotation angle in degrees

    Raises:
        AlignmentError: If calculation fails
    """
    try:
        # Use skimage's Radon transform
        try:
            from skimage.transform import radon
        except ImportError:
            raise AlignmentError("Radon transform requires scikit-image")

        # Create theta range
        theta = np.linspace(angle_range[0], angle_range[1], 180)

        # Compute Radon transform
        sinogram1 = radon(source_edges, theta=theta)
        sinogram2 = radon(target_edges, theta=theta)

        # Find the rotation angle that gives maximum correlation
        correlation = np.zeros(len(theta))

        for i in range(len(theta)):
            correlation[i] = np.corrcoef(sinogram1[:, 0], sinogram2[:, i])[0, 1]

        # Get angle with maximum correlation
        rotation_angle = theta[np.argmax(correlation)]

        return rotation_angle

    except Exception as e:
        if isinstance(e, AlignmentError):
            raise
        raise AlignmentError(f"Radon transform rotation calculation failed: {str(e)}")


def normalize_pixel_size(
    data: np.ndarray,
    source_pixel_size: Tuple[float, float],
    target_pixel_size: Tuple[float, float],
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Resize data to match target pixel size.

    Args:
        data: Image data to resize
        source_pixel_size: Original pixel size
        target_pixel_size: Target pixel size

    Returns:
        Tuple of (resized data, zoom factors)

    Raises:
        AlignmentError: If resizing fails
        ValidationError: If inputs are invalid
    """
    try:
        # Validate pixel sizes
        if any(p <= 0 for p in source_pixel_size + target_pixel_size):
            raise ValidationError("Pixel sizes must be positive")

        # Calculate zoom factors
        zoom_y = source_pixel_size[1] / target_pixel_size[1]
        zoom_x = source_pixel_size[0] / target_pixel_size[0]

        # Apply zoom
        from scipy.ndimage import zoom

        resized_data = zoom(data, (zoom_y, zoom_x), mode="reflect")

        return resized_data, (zoom_y, zoom_x)

    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise AlignmentError(f"Pixel size normalization failed: {str(e)}")


def resize_mask(mask: np.ndarray, zoom_factors: Tuple[float, float]) -> np.ndarray:
    """
    Resize a binary mask using the given zoom factors.

    Args:
        mask: Binary mask to resize
        zoom_factors: Zoom factors (y, x)

    Returns:
        Resized binary mask

    Raises:
        AlignmentError: If resizing fails
    """
    try:
        from scipy.ndimage import zoom

        # Apply zoom
        resized = zoom(mask.astype(float), zoom_factors, mode="constant", cval=0)

        # Threshold to get binary mask
        return resized > 0.5

    except Exception as e:
        raise AlignmentError(f"Mask resizing failed: {str(e)}")


def pad_to_same_size(
    image1: np.ndarray,
    image2: np.ndarray,
    mode: str = "constant",
    constant_value: float = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad two images to the same size.

    Args:
        image1: First image
        image2: Second image
        mode: Padding mode
        constant_value: Value for constant padding

    Returns:
        Tuple of (padded_image1, padded_image2)

    Raises:
        AlignmentError: If padding fails
    """
    try:
        # Get shapes
        shape1, shape2 = image1.shape, image2.shape

        # Calculate maximum dimensions
        max_shape = np.maximum(shape1, shape2)

        # Calculate padding for image1
        pad1_y = max(0, max_shape[0] - shape1[0])
        pad1_x = max(0, max_shape[1] - shape1[1])

        # Calculate padding for image2
        pad2_y = max(0, max_shape[0] - shape2[0])
        pad2_x = max(0, max_shape[1] - shape2[1])

        # Calculate padding for each side
        pad1_top = pad1_y // 2
        pad1_bottom = pad1_y - pad1_top
        pad1_left = pad1_x // 2
        pad1_right = pad1_x - pad1_left

        pad2_top = pad2_y // 2
        pad2_bottom = pad2_y - pad2_top
        pad2_left = pad2_x // 2
        pad2_right = pad2_x - pad2_left

        # Apply padding
        padded1 = np.pad(
            image1,
            ((pad1_top, pad1_bottom), (pad1_left, pad1_right)),
            mode=mode,
            constant_values=constant_value,
        )

        padded2 = np.pad(
            image2,
            ((pad2_top, pad2_bottom), (pad2_left, pad2_right)),
            mode=mode,
            constant_values=constant_value,
        )

        return padded1, padded2

    except Exception as e:
        raise AlignmentError(f"Image padding failed: {str(e)}")


def center_structures(
    source: Structure, target: Structure
) -> Tuple[Structure, Structure]:
    """
    Center two structures based on their centers of mass.

    Args:
        source: Source structure
        target: Target structure

    Returns:
        Tuple of (centered_source, centered_target)

    Raises:
        AlignmentError: If centering fails
    """
    try:
        # Get centers
        source_center = source.center
        target_center = target.center

        # Calculate shifts
        height, width = source.data.shape
        center_point = (height // 2, width // 2)

        source_shift = np.subtract(center_point, source_center)
        target_shift = np.subtract(center_point, target_center)

        # Apply shifts
        from scipy.ndimage import shift as apply_shift

        # Shift source
        shifted_source_data = apply_shift(
            source.data.filled(0), shift=source_shift, mode="constant", cval=0
        )

        shifted_source_mask = (
            apply_shift(
                (~source.data.mask).astype(float),
                shift=source_shift,
                mode="constant",
                cval=0,
            )
            > 0.5
        )

        # Shift target
        shifted_target_data = apply_shift(
            target.data.filled(0), shift=target_shift, mode="constant", cval=0
        )

        shifted_target_mask = (
            apply_shift(
                (~target.data.mask).astype(float),
                shift=target_shift,
                mode="constant",
                cval=0,
            )
            > 0.5
        )

        # Create centered structures
        centered_source = Structure(
            data=ma.masked_array(shifted_source_data, mask=~shifted_source_mask),
            pixel_size=source.pixel_size,
        )

        centered_target = Structure(
            data=ma.masked_array(shifted_target_data, mask=~shifted_target_mask),
            pixel_size=target.pixel_size,
        )

        return centered_source, centered_target

    except Exception as e:
        raise AlignmentError(f"Structure centering failed: {str(e)}")


def _warp_structure_mask(
    mask: np.ndarray,
    M: np.ndarray,
    output_dim: Tuple[int, int],
) -> np.ndarray:
    """
    Apply an affine transformation to a binary structure mask.

    The mask is treated as a float image, warped, and then thresholded
    back to boolean. This ensures the structure mask stays consistent
    with the transformed raw data without requiring a new detect_structure()
    call.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask where True indicates structure pixels.
    M : np.ndarray
        2x3 affine transformation matrix (as used by cv2.warpAffine).
    output_dim : tuple of (width, height)
        Output canvas size in (width, height) order (cv2 convention).

    Returns
    -------
    np.ndarray
        Boolean mask of shape (height, width) after transformation.
    """
    warped = cv2.warpAffine(
        mask.astype(np.float32),
        M,
        output_dim,
        flags=cv2.INTER_NEAREST,  # no interpolation artefacts on binary data
        borderValue=0,
    )
    return warped > 0.5


def align_processor_pair(
    source: "StructureProcessor",
    reference: "StructureProcessor",
    scaling: Literal["upsampling", "downsampling"] = "downsampling",
    visualize_line_detection: bool = False,
    keep_source_fixed: bool = False,
) -> Tuple["StructureProcessor", Dict]:
    """
    Align source with reference via cv2.warpAffine on raw data.

    Both processors are mutated in-place (raw data, pixel_size, and
    structure mask). The structure mask is propagated through the same
    affine transform as the raw data, so no re-detection is needed after
    this call.

    The background of both processors must already be zeroed before
    calling this function (see StructureProcessor._zero_background_inplace),
    so that newly introduced border pixels are consistently zero.

    The affine transform is built as Rotate -> Scale -> Translate, where
    the rotation pivots around each structure's own (pre-transform) center
    and the translation is computed AFTER scaling/rotation from the actual
    (rotated, scaled) center position. This guarantees the structure center
    lands exactly at the output canvas center, for any rotation angle or
    zoom factor (see `_build_matrix` below for details / rationale).

    Parameters
    ----------
    source : StructureProcessor
        The processor to be aligned. Must have structure detected.
    reference : StructureProcessor
        The reference processor. Must have structure detected.
    scaling : {'upsampling', 'downsampling'}
        Determines which image is rescaled to match pixel sizes. Ignored
        if `keep_source_fixed` is True.
    visualize_line_detection : bool
        If True, show debug visualisation of line detection.
    keep_source_fixed : bool
        If True, `source` is treated as an immutable reference canvas:
        it is never warped, rescaled, or resized, only `reference` is
        warped to match it (output canvas = source's current shape).
        Use this when aligning multiple targets against the same source
        one after another, so that earlier targets are not invalidated
        by a later target moving `source` again (see
        `align_processors_many`).

    Returns
    -------
    source, transform_info
        source is mutated in-place; transform_info holds all
        transformation parameters for downstream use.

    Raises
    ------
    AlignmentError
        If the transformation cannot be computed or applied.
    ValidationError
        If either processor has no structure detected.
    """
    if source.structure is None:
        raise ValidationError("align_processor_pair: source has no detected structure.")
    if reference.structure is None:
        raise ValidationError(
            "align_processor_pair: reference has no detected structure."
        )

    # --- Determine which image is 'target' (the fixed/output canvas) ------
    if keep_source_fixed:
        # source never moves; reference is always the one being warped.
        target, src, is_source_target = source, reference, True
    else:
        dim_source = source.shape[0] * source.shape[1]
        dim_ref = reference.shape[0] * reference.shape[1]

        if scaling == "upsampling":
            if dim_ref < dim_source:
                target, src, is_source_target = reference, source, False
            else:
                target, src, is_source_target = source, reference, True
        elif scaling == "downsampling":
            if dim_ref < dim_source:
                target, src, is_source_target = source, reference, True
            else:
                target, src, is_source_target = reference, source, False
        else:
            raise NotImplementedError(
                f"scaling='{scaling}' is not implemented. Use 'upsampling' or 'downsampling'."
            )

    # --- Zoom factors ------------------------------------------------------
    zoom_factor_target = [
        target.pixel_size[1] / src.pixel_size[1],  # row (y)
        target.pixel_size[0] / src.pixel_size[0],  # col (x)
    ]
    zoom_factor_source = [1.0, 1.0]

    # --- Rotation angles from corner points [tl, tr, br, bl] -------------
    tl_t, tr_t, br_t, bl_t = target.structure.corner_points
    tl_s, tr_s, br_s, bl_s = src.structure.corner_points

    h_angle_top_target = np.degrees(np.arctan2(tr_t[0] - tl_t[0], tr_t[1] - tl_t[1]))
    v_angle_left_target = -np.degrees(np.arctan2(bl_t[1] - tl_t[1], bl_t[0] - tl_t[0]))
    h_angle_top_source = np.degrees(np.arctan2(tr_s[0] - tl_s[0], tr_s[1] - tl_s[1]))
    v_angle_left_source = -np.degrees(np.arctan2(bl_s[1] - tl_s[1], bl_s[0] - tl_s[0]))

    avg_angle_target = (h_angle_top_target + v_angle_left_target) / 2
    avg_angle_source = (h_angle_top_source + v_angle_left_source) / 2

    # --- Output canvas dimensions -------------------------------------------
    # Determined BEFORE building the transformation matrices, because the
    # translation step below needs to know the exact final canvas center.
    if keep_source_fixed:
        # source's canvas is immutable; reference is warped into it as-is.
        output_dim = (source.shape[1], source.shape[0])  # (width, height)
    else:
        target_output = (
            int(target.shape[1] * zoom_factor_target[1]),  # width
            int(target.shape[0] * zoom_factor_target[0]),  # height
        )
        output_dim = (
            max(target_output[0], src.shape[1]),
            max(target_output[1], src.shape[0]),
        )
    image_center = (output_dim[1] / 2, output_dim[0] / 2)  # (row, col)

    # --- Affine transformation matrices: Rotate -> Scale -> Translate -----
    # Order matters: we rotate FIRST, pivoting exactly around the
    # structure's own (pre-transform) center, which leaves that center's
    # position unchanged by the rotation. We then scale around the origin
    # (which moves the now-fixed center predictably to
    # `structure_center * zoom`). Only THEN do we compute the translation,
    # directly as `image_center - structure_center * zoom`. This way the
    # structure center always lands exactly on the output canvas center,
    # regardless of rotation angle or zoom factor - unlike a
    # Scale -> Translate -> Rotate order with a translation that is
    # computed BEFORE the rotation is known, which silently drifts the
    # structure off-center (this was the root cause of the alignment bug;
    # see `Structure.alignment_transform` for the same correct pattern
    # already used elsewhere in this codebase, here extended with zoom).
    def _build_matrix(zoom, structure_center, angle):
        # structure_center, image_center are (row, col); cv2 wants (col, row)
        cx, cy = float(structure_center[1]), float(structure_center[0])

        R = cv2.getRotationMatrix2D(center=(cx, cy), angle=angle, scale=1.0)
        R = np.vstack([R, [0.0, 0.0, 1.0]])  # promote 2x3 -> 3x3 for multi_dot

        S = np.array([[zoom[1], 0.0, 0.0], [0.0, zoom[0], 0.0], [0.0, 0.0, 1.0]])

        # Rotation leaves (cx, cy) fixed; scaling about the origin then
        # moves it to (cx * zoom_x, cy * zoom_y). Translate from there.
        scaled_cx, scaled_cy = cx * zoom[1], cy * zoom[0]
        T = np.array(
            [
                [1.0, 0.0, image_center[1] - scaled_cx],
                [0.0, 1.0, image_center[0] - scaled_cy],
                [0.0, 0.0, 1.0],
            ]
        )

        return np.linalg.multi_dot([T, S, R])

    M_target = _build_matrix(
        zoom_factor_target, target.structure.center, avg_angle_target
    )
    M_source = _build_matrix(zoom_factor_source, src.structure.center, avg_angle_source)

    # --- Warp raw data (border = 0, background was zeroed upstream) -------
    def _warp(data: Optional[np.ndarray], M: np.ndarray) -> Optional[np.ndarray]:
        if data is None:
            return None
        return cv2.warpAffine(data, M[:2, :], output_dim, borderValue=0)

    warped_target_phase = _warp(
        target.phase_data.data if target.phase_data else None, M_target
    )
    warped_target_height = _warp(
        target.height_data.data if target.height_data else None, M_target
    )
    warped_target_intensity = _warp(
        target.intensity_data.data if target.intensity_data else None, M_target
    )

    warped_source_phase = _warp(
        src.phase_data.data if src.phase_data else None, M_source
    )
    warped_source_height = _warp(
        src.height_data.data if src.height_data else None, M_source
    )
    warped_source_intensity = _warp(
        src.intensity_data.data if src.intensity_data else None, M_source
    )

    # --- Warp structure masks (same matrix, nearest-neighbour) ------------
    # This avoids re-running detect_structure() on the new shape.
    # The structure mask is a binary image — applying the same affine
    # transform keeps it perfectly consistent with the raw data.
    target_struct_mask_warped = _warp_structure_mask(
        ~target.structure.data.mask, M_target[:2, :], output_dim
    )
    source_struct_mask_warped = _warp_structure_mask(
        ~src.structure.data.mask, M_source[:2, :], output_dim
    )

    # --- Align pixel sizes ------------------------------------------------
    if is_source_target:
        source.update_pixel_size(reference.pixel_size)
    else:
        reference.update_pixel_size(source.pixel_size)

    # --- Write raw data back ----------------------------------------------
    def _write_back(proc, phase, height, intensity):
        if phase is not None:
            proc.update_data(phase_data=phase)
        if height is not None:
            proc.update_data(height_data=height)
        if intensity is not None:
            proc.update_data(intensity_data=intensity)

    _write_back(
        target, warped_target_phase, warped_target_height, warped_target_intensity
    )
    _write_back(src, warped_source_phase, warped_source_height, warped_source_intensity)

    # --- Update structure masks in-place ----------------------------------
    # Rebuild the structure's masked array from the warped mask.
    # All other structure properties (center, bounds, corner_points) are
    # derived lazily from the mask, so they stay correct automatically.
    def _update_structure_mask(proc, warped_bool_mask: np.ndarray):
        old = proc.structure
        proc.structure = Structure(
            data=ma.masked_array(
                proc._active_data.data,  # fresh raw data after write-back
                mask=~warped_bool_mask,
            ),
            pixel_size=old.pixel_size,
        )
        # Preserve segments if any (their masks would need separate handling)
        if old.segments:
            proc.structure.segments = old.segments

    _update_structure_mask(target, target_struct_mask_warped)
    _update_structure_mask(src, source_struct_mask_warped)

    # NOTE: No separate re-centering step is needed here. The corrected
    # `_build_matrix` above computes the translation AFTER rotation and
    # scaling, directly from the actual transformed center position, so
    # the structure always lands exactly at the output canvas center in
    # a single warpAffine call - for any rotation angle, zoom factor, or
    # `output_dim`. The old `_recenter` post-processing step (and the
    # `recenter_target`/`recenter_source` flags) is therefore obsolete
    # and has been removed; it also previously called
    # `update_data(phase=...)` instead of `update_data(phase_data=...)`,
    # which would raise a TypeError whenever it actually triggered.

    # --- Transform info ---------------------------------------------------
    transform_info = {
        "scaling": scaling,
        "transformation_order": "rotate -> scale -> translate",
        "output_shape": output_dim,
        "zoom_factor_source": (
            zoom_factor_target if is_source_target else zoom_factor_source
        ),
        "zoom_factor_ref": (
            zoom_factor_source if is_source_target else zoom_factor_target
        ),
        "rotation_angle_source": (
            avg_angle_target if is_source_target else avg_angle_source
        ),
        "rotation_angle_ref": (
            avg_angle_source if is_source_target else avg_angle_target
        ),
        "keep_source_fixed": keep_source_fixed,
    }

    return source, transform_info


def align_processors_many(
    source: "StructureProcessor",
    targets: "List[StructureProcessor]",
    scaling: Literal["upsampling", "downsampling"] = "downsampling",
    **kwargs,
) -> Tuple[List["StructureProcessor"], Dict]:
    """
    Align all targets pairwise with source, then synchronize the canvas.

    Workflow
    --------
    1. source <-> targets[0]: both are aligned against each other (this is
       the only step where `source` itself is allowed to move/resize).
    2. source <-> targets[1], [2], ...: `source` is now treated as a FIXED
       reference (`keep_source_fixed=True`); only each remaining target is
       warped to match it. This is essential - without it, every iteration
       would re-warp `source` again relative to that one target, silently
       invalidating the alignment of all previously processed targets
       (this was the cause of structures appearing clipped/off-center when
       more than one target was aligned in a single call).
    3. Pad all raw data to the same shared canvas size (symmetric padding).
    4. Re-run detect_structure() on all objects (their masks are propagated
       through the warp already, but padding in step 3 changes the shape).

    Parameters
    ----------
    source : StructureProcessor
        The reference processor. Must have structure detected (or it will
        be auto-detected).
    targets : list of StructureProcessor
        One or more processors to align against `source`.
    scaling : {'upsampling', 'downsampling'}
        Only applies to the first source<->targets[0] alignment, which is
        the only pair where `source` may still be rescaled. All later
        pairs always rescale the *target*, never `source`.
    **kwargs
        Additional keyword arguments forwarded to `align_processor_pair`
        (e.g. `visualize_line_detection`).

    Returns
    -------
    [source, *aligned_targets], transform_dict
    """
    if not targets:
        raise ValidationError("targets list must not be empty.")

    combined_transform: Dict = {}

    first_target, *remaining_targets = targets

    if first_target.structure is None:
        first_target.detect_structure()

    # Step 1: source and the first target are aligned mutually - this is
    # the only call allowed to move/resize `source`.
    _, t_info = align_processor_pair(source, first_target, scaling=scaling, **kwargs)
    combined_transform[0] = t_info

    # Step 2: every further target is warped into source's now-fixed frame.
    # `source` itself must not be touched again, or the alignment of
    # `first_target` (and any earlier remaining target) would be silently
    # invalidated.
    for idx, tgt in enumerate(remaining_targets, start=1):
        if tgt.structure is None:
            tgt.detect_structure()

        _, t_info = align_processor_pair(
            source, tgt, scaling=scaling, keep_source_fixed=True, **kwargs
        )
        combined_transform[idx] = t_info

    all_procs = [source] + list(targets)
    canvas_shape = sync_raw_data_canvas(all_procs)
    combined_transform["canvas_shape"] = canvas_shape

    return all_procs, combined_transform


def sync_raw_data_canvas(
    processors: "List[StructureProcessor]",
) -> Tuple[int, int]:
    """
    Pad the raw data of all processors to the same shared canvas size.

    Padding is symmetric (the image stays centered), new pixels = 0. The
    structure mask of each processor is padded with the exact same
    pad_spec, so it stays perfectly shape-consistent with the raw data
    afterward - no `detect_structure()` call is needed (or should be
    performed) after this function runs, since that would discard any
    custom detection settings (e.g. a manually chosen percentile
    threshold) the structure was originally detected with, silently
    replacing it with a fresh, default-settings detection.

    Returns
    -------
    canvas_shape : (height, width)
    """
    max_h = max(p._active_data.data.shape[0] for p in processors)
    max_w = max(p._active_data.data.shape[1] for p in processors)

    for proc in processors:
        h, w = proc._active_data.data.shape
        if h == max_h and w == max_w:
            continue

        pad_top = (max_h - h) // 2
        pad_bottom = (max_h - h) - pad_top
        pad_left = (max_w - w) // 2
        pad_right = (max_w - w) - pad_left
        pad_spec = ((pad_top, pad_bottom), (pad_left, pad_right))

        def _pad(arr):
            return np.pad(arr, pad_spec, mode="constant", constant_values=0)

        if proc.phase_data is not None:
            proc.update_data(phase_data=_pad(proc.phase_data.data))
        if proc.height_data is not None:
            proc.update_data(height_data=_pad(proc.height_data.data))
        if proc.intensity_data is not None:
            proc.update_data(intensity_data=_pad(proc.intensity_data.data))

        # --- Keep the structure mask in lock-step with the padded data ---
        if proc.structure is not None:
            structure_present = ~proc.structure.data.mask
            padded_structure_present = _pad(structure_present.astype(bool))
            proc.structure = Structure(
                data=ma.masked_array(
                    proc._active_data.data,  # fresh raw data after padding
                    mask=~padded_structure_present,
                ),
                pixel_size=proc.structure.pixel_size,
            )

    return (max_h, max_w)
