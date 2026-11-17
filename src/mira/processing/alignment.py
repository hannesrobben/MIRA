"""
Image alignment module.

This module provides functions for aligning structures and images, including
rotation, translation, scaling, and pixel size normalization.
"""

from __future__ import annotations
import numpy as np
import numpy.ma as ma

from typing import TYPE_CHECKING, Dict, List, Literal, Tuple, Optional, Union



from ..core.structure import Structure, Segment
from ..utils.exceptions import AlignmentError, ValidationError

if TYPE_CHECKING:
    from ..processor import StructureProcessor
    from .registration import RegistrationRefiner


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


def _structure_tilt_degrees(corner_points) -> float:
    """
    Average tilt of a (near-rectangular) structure, in degrees.

    Uses the two horizontal edges (top tl->tr and bottom bl->br), matching the
    convention in ``Structure.alignment_transform`` (single source of truth for
    the angle convention across the codebase). arctan2 avoids division-by-zero
    and quadrant ambiguity.
    """
    tl, tr, br, bl = corner_points  # each (row, col)
    phi_top = np.arctan2(tr[0] - tl[0], tr[1] - tl[1])
    phi_bot = np.arctan2(br[0] - bl[0], br[1] - bl[1])
    return float(np.degrees((phi_top + phi_bot) / 2))


def _rotated_scaled_canvas(
    shape: Tuple[int, int], angle_deg: float, zoom: List[float]
) -> Tuple[int, int]:
    """
    Bounding-box size (width, height) of a frame of ``shape`` = (height, width)
    after rotating by ``angle_deg`` and scaling by ``zoom`` = [zoom_y, zoom_x].

    Used to size the output canvas so a rotated structure is never clipped
    (a pure ``shape * zoom`` estimate ignores that rotation grows the box).
    """
    h, w = shape
    a = np.radians(angle_deg)
    cos_a, sin_a = abs(np.cos(a)), abs(np.sin(a))
    out_w = (w * cos_a + h * sin_a) * zoom[1]
    out_h = (h * cos_a + w * sin_a) * zoom[0]
    return int(np.ceil(out_w)), int(np.ceil(out_h))


def _build_alignment_matrix(
    zoom: List[float],
    structure_center: Tuple[float, float],
    angle: float,
    image_center: Tuple[float, float],
) -> np.ndarray:
    """
    Build a 3x3 affine matrix as Rotate -> Scale -> Translate.

    We rotate FIRST, pivoting around the structure's own (pre-transform) center,
    which leaves that center fixed. We then scale about the origin (moving the
    center predictably to ``center * zoom``) and finally translate it straight
    onto ``image_center``. This lands the structure center exactly at the canvas
    center for any rotation angle or zoom factor, in a single warpAffine call -
    unlike a Scale -> Translate -> Rotate order whose translation is computed
    before the rotation is known and silently drifts the structure off-center.

    All centers are (row, col); cv2 uses (col, row).
    """
    import cv2

    cx, cy = float(structure_center[1]), float(structure_center[0])

    R = cv2.getRotationMatrix2D(center=(cx, cy), angle=angle, scale=1.0)
    R = np.vstack([R, [0.0, 0.0, 1.0]])  # promote 2x3 -> 3x3 for multi_dot

    S = np.array([[zoom[1], 0.0, 0.0], [0.0, zoom[0], 0.0], [0.0, 0.0, 1.0]])

    scaled_cx, scaled_cy = cx * zoom[1], cy * zoom[0]
    T = np.array(
        [
            [1.0, 0.0, image_center[1] - scaled_cx],
            [0.0, 1.0, image_center[0] - scaled_cy],
            [0.0, 0.0, 1.0],
        ]
    )
    return np.linalg.multi_dot([T, S, R])


def _apply_warp_to_processor(
    proc: "StructureProcessor",
    M: np.ndarray,
    output_dim: Tuple[int, int],
) -> None:
    """
    Warp all raw channels and the structure (mask + segments) of ``proc`` with
    the affine matrix ``M`` onto a canvas of ``output_dim`` = (width, height).

    Thin adapter around ``StructureProcessor.apply_affine_transform`` - the
    single, authoritative geometry gateway. It is kept so existing call sites in
    this module can keep their (width, height) OpenCV-style ``output_dim``
    convention, while all the actual warping (raw channels, structure mask,
    segment masks) and the raw/structure consistency guarantee live in one
    place. Everything is mutated in-place and kept mutually consistent.
    """
    width, height = output_dim
    proc.apply_affine_transform(M, output_shape=(height, width))


def align_processor_pair(
    source: "StructureProcessor",
    reference: "StructureProcessor",
    scaling: Literal["upsampling", "downsampling"] = "downsampling",
    visualize_line_detection: bool = False,
    keep_source_fixed: bool = False,
    refiner: Optional["RegistrationRefiner"] = None,
) -> Tuple["StructureProcessor", Dict]:
    """
    Align source with reference via cv2.warpAffine on raw data.

    Both processors are mutated in-place (raw data, pixel_size, structure mask
    and segments). The structure mask is propagated through the same affine
    transform as the raw data, so no re-detection is needed after this call.

    Strategy: each moving structure is de-rotated to be axis-aligned (rotating
    about its own center) and its center is translated onto the output-canvas
    center; exactly one of the two is additionally rescaled so the pixel sizes
    match. The background of both processors must already be zeroed (see
    StructureProcessor._zero_background_inplace) so border pixels (=0) are clean.

    Parameters
    ----------
    source : StructureProcessor
        The processor to be aligned. Must have a detected structure.
    reference : StructureProcessor
        The reference processor. Must have a detected structure.
    scaling : {'upsampling', 'downsampling'}
        Which image is rescaled to match pixel sizes. Ignored if
        ``keep_source_fixed`` is True (reference always adopts source's grid).
    visualize_line_detection : bool
        Reserved for debug visualisation of line detection.
    keep_source_fixed : bool
        If True, ``source`` is an immutable reference canvas: it is never
        warped, rescaled, or resized; only ``reference`` is warped to match it
        (output canvas = source's current shape). Use this when aligning
        several targets against the same source in sequence so that earlier
        targets are not invalidated by a later one moving source again
        (see ``align_processors_many``).
    refiner : RegistrationRefiner, optional
        If given, a content-based refinement step is applied on top of the
        geometric pre-alignment (coarse-to-fine), estimating a residual
        affine correction from image intensities.

    Returns
    -------
    source, transform_info
        ``source`` is returned (mutated in-place unless keep_source_fixed);
        ``transform_info`` holds the transformation parameters.

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

    # ---------------------------------------------------------------------
    # Case 1: source is fixed - only reference is warped onto source's canvas.
    # ---------------------------------------------------------------------
    if keep_source_fixed:
        zoom_reference = [
            reference.pixel_size[1] / source.pixel_size[1],  # y (row)
            reference.pixel_size[0] / source.pixel_size[0],  # x (col)
        ]
        angle_reference = _structure_tilt_degrees(reference.structure.corner_points)

        output_dim = (source.shape[1], source.shape[0])  # (width, height), fixed
        image_center = (output_dim[1] / 2, output_dim[0] / 2)  # (row, col)

        M_reference = _build_alignment_matrix(
            zoom_reference, reference.structure.center, angle_reference, image_center
        )

        _apply_warp_to_processor(reference, M_reference, output_dim)
        reference.update_pixel_size(source.pixel_size)
        reference.structure.pixel_size = source.pixel_size

        if refiner is not None:
            residual = refiner.estimate(
                reference._active_data.data, source._active_data.data
            )
            _apply_warp_to_processor(reference, residual, output_dim)

        transform_info = {
            "scaling": scaling,
            "transformation_order": "rotate -> scale -> translate",
            "output_shape": output_dim,
            "keep_source_fixed": True,
            "rotation_angle_reference": angle_reference,
            "zoom_factor_reference": zoom_reference,
        }
        return source, transform_info

    # ---------------------------------------------------------------------
    # Case 2: both move. One processor is rescaled to the other's pixel grid;
    # both are then de-rotated and centered onto a shared canvas.
    # ---------------------------------------------------------------------
    dim_source = source.shape[0] * source.shape[1]
    dim_ref = reference.shape[0] * reference.shape[1]

    if scaling == "upsampling":
        # Upsample the smaller image onto the larger pixel grid.
        scaled_proc, kept_proc = (
            (reference, source) if dim_ref < dim_source else (source, reference)
        )
    elif scaling == "downsampling":
        # Downsample the larger image onto the smaller pixel grid.
        scaled_proc, kept_proc = (
            (source, reference) if dim_ref < dim_source else (reference, source)
        )
    else:
        raise NotImplementedError(
            f"scaling='{scaling}' is not implemented. "
            "Use 'upsampling' or 'downsampling'."
        )

    zoom_scaled = [
        scaled_proc.pixel_size[1] / kept_proc.pixel_size[1],  # y (row)
        scaled_proc.pixel_size[0] / kept_proc.pixel_size[0],  # x (col)
    ]
    zoom_kept = [1.0, 1.0]

    angle_scaled = _structure_tilt_degrees(scaled_proc.structure.corner_points)
    angle_kept = _structure_tilt_degrees(kept_proc.structure.corner_points)

    # Output canvas large enough for both rotated + scaled frames (no clipping).
    box_scaled = _rotated_scaled_canvas(scaled_proc.shape, angle_scaled, zoom_scaled)
    box_kept = _rotated_scaled_canvas(kept_proc.shape, angle_kept, zoom_kept)
    output_dim = (max(box_scaled[0], box_kept[0]), max(box_scaled[1], box_kept[1]))
    image_center = (output_dim[1] / 2, output_dim[0] / 2)  # (row, col)

    M_scaled = _build_alignment_matrix(
        zoom_scaled, scaled_proc.structure.center, angle_scaled, image_center
    )
    M_kept = _build_alignment_matrix(
        zoom_kept, kept_proc.structure.center, angle_kept, image_center
    )

    _apply_warp_to_processor(scaled_proc, M_scaled, output_dim)
    _apply_warp_to_processor(kept_proc, M_kept, output_dim)

    # The rescaled processor adopts the kept processor's pixel size.
    scaled_proc.update_pixel_size(kept_proc.pixel_size)
    scaled_proc.structure.pixel_size = kept_proc.pixel_size

    if refiner is not None:
        residual = refiner.estimate(
            scaled_proc._active_data.data, kept_proc._active_data.data
        )
        _apply_warp_to_processor(scaled_proc, residual, output_dim)

    transform_info = {
        "scaling": scaling,
        "transformation_order": "rotate -> scale -> translate",
        "output_shape": output_dim,
        "keep_source_fixed": False,
        "scaled_processor": "source" if scaled_proc is source else "reference",
        "rotation_angle_source": (
            angle_scaled if scaled_proc is source else angle_kept
        ),
        "rotation_angle_reference": (
            angle_scaled if scaled_proc is reference else angle_kept
        ),
        "zoom_factor_source": zoom_scaled if scaled_proc is source else zoom_kept,
        "zoom_factor_reference": (
            zoom_scaled if scaled_proc is reference else zoom_kept
        ),
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
    4. Pad the structure mask (and segment masks) of every object with the same
       pad_spec used for the raw data, so they stay shape-consistent without any
       detect_structure() call (which would discard the original detection
       settings).

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
    structure mask (and any segment masks) of each processor are padded with the
    exact same pad_spec, so they stay perfectly shape-consistent with the raw
    data afterward - no `detect_structure()` call is needed (or should be
    performed) after this function runs, since that would discard any custom
    detection settings (e.g. a manually chosen percentile threshold) the
    structure was originally detected with, silently replacing it with a fresh,
    default-settings detection.

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

        def _pad(arr, spec=pad_spec):
            return np.pad(arr, spec, mode="constant", constant_values=0)

        if proc.phase_data is not None:
            proc.update_data(phase_data=_pad(proc.phase_data.data))
        if proc.height_data is not None:
            proc.update_data(height_data=_pad(proc.height_data.data))
        if proc.intensity_data is not None:
            proc.update_data(intensity_data=_pad(proc.intensity_data.data))

        # --- Keep structure mask AND segment masks in lock-step with data ---
        if proc.structure is not None:
            new_raw = proc._active_data.data  # fresh raw data after padding
            padded_structure = _pad((~proc.structure.data.mask).astype(bool))

            padded_segments = []
            for seg in proc.structure.segments:
                seg_present = _pad((~seg.data.mask).astype(bool))
                padded_segments.append(
                    Segment(
                        data=ma.masked_array(new_raw, mask=~seg_present),
                        pixel_size=seg.pixel_size,
                        id=seg.id,
                    )
                )

            proc.structure = Structure(
                data=ma.masked_array(new_raw, mask=~padded_structure),
                pixel_size=proc.structure.pixel_size,
            )
            proc.structure.segments = padded_segments

    return (max_h, max_w)
