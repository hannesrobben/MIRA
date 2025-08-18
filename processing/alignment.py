"""
Image alignment module.

This module provides functions for aligning structures and images, including
rotation, translation, scaling, and pixel size normalization.
"""

import numpy as np
import numpy.ma as ma
from typing import Union, Optional, Tuple, List, Dict, Literal, Any

from ..core.image import ImageData
from ..core.structure import Structure
from ..utils.exceptions import ProcessingError, AlignmentError, ValidationError


def align_structures(source_structure: Structure,
                     target_structure: Structure,
                     method: Literal['edge', 'mask', 'feature'] = 'edge',
                     scaling: Literal['upsampling', 'downsampling', 'none'] = 'none',
                     return_transform: bool = False,
                     **kwargs) -> Union[Tuple[Structure, Dict], Structure]:
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
            data=source_structure.data.copy(),
            pixel_size=source_structure.pixel_size
        )

        target = Structure(
            data=target_structure.data.copy(),
            pixel_size=target_structure.pixel_size
        )

        # Initialize transform info
        transform_info = {
            'method': method,
            'scaling': scaling,
            'original_source_shape': source.data.shape,
            'original_target_shape': target.data.shape,
            'original_source_pixel_size': source.pixel_size,
            'original_target_pixel_size': target.pixel_size
        }

        # Step 1: Normalize pixel size if needed
        if source.pixel_size != target.pixel_size:
            if scaling == 'upsampling':
                # Make source match target's pixel size
                source_data, zoom_factors = normalize_pixel_size(
                    source.data.data,
                    source.pixel_size,
                    target.pixel_size
                )

                # Update source data and pixel size
                source.data = ma.masked_array(
                    source_data,
                    mask=~resize_mask(~source.data.mask, zoom_factors)
                )
                source.pixel_size = target.pixel_size

                transform_info['pixel_scaling'] = {
                    'zoom_factors': zoom_factors,
                    'direction': 'source_to_target'
                }

            elif scaling == 'downsampling':
                # Make target match source's pixel size
                target_data, zoom_factors = normalize_pixel_size(
                    target.data.data,
                    target.pixel_size,
                    source.pixel_size
                )

                # Update target data and pixel size
                target.data = ma.masked_array(
                    target_data,
                    mask=~resize_mask(~target.data.mask, zoom_factors)
                )
                target.pixel_size = source.pixel_size

                transform_info['pixel_scaling'] = {
                    'zoom_factors': zoom_factors,
                    'direction': 'target_to_source'
                }

            else:  # 'none'
                # Just record the difference
                transform_info['pixel_scaling'] = {
                    'source_pixel_size': source.pixel_size,
                    'target_pixel_size': target.pixel_size,
                    'no_scaling_applied': True
                }

        # Step 2: Make sure structures have the same dimensions
        source_data, target_data = pad_to_same_size(
            source.data.data,
            target.data.data
        )

        source_mask = pad_to_same_size(
            ~source.data.mask,
            ~target.data.mask
        )[0]

        target_mask = pad_to_same_size(
            ~target.data.mask,
            ~source.data.mask
        )[1]

        # Update data with padded versions
        source.data = ma.masked_array(source_data, mask=~source_mask)
        target.data = ma.masked_array(target_data, mask=~target_mask)

        transform_info['padding'] = {
            'padded_shape': source.data.shape
        }

        # Step 3: Apply specific alignment method
        if method == 'edge':
            aligned_source, method_info = align_by_edges(source, target, **kwargs)
        elif method == 'mask':
            aligned_source, method_info = align_by_mask(source, target, **kwargs)
        elif method == 'feature':
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


def align_by_edges(source: Structure,
                   target: Structure,
                   rotation_method: Literal['hough', 'minimize', 'radon'] = 'hough',
                   detect_edges_kwargs: Optional[Dict] = None,
                   **kwargs) -> Tuple[Structure, Dict]:
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
        from ..processing.detection import detect_edges

        # Set default edge detection parameters
        edge_params = {'method': 'sobel', 'threshold': 0.2}
        if detect_edges_kwargs:
            edge_params.update(detect_edges_kwargs)

        # Detect edges
        source_edges = detect_edges(source.data, **edge_params)
        target_edges = detect_edges(target.data, **edge_params)

        # Calculate rotation
        rotation_angle = calculate_rotation(
            source_edges,
            target_edges,
            method=rotation_method,
            **kwargs
        )

        # Apply rotation to source
        from scipy.ndimage import rotate

        rotated_data = rotate(
            source.data.filled(0),
            rotation_angle,
            reshape=False,
            mode='constant',
            cval=0
        )

        rotated_mask = rotate(
            ~source.data.mask,
            rotation_angle,
            reshape=False,
            mode='constant',
            cval=0
        ) > 0.5  # Threshold to get boolean mask

        # Create aligned structure
        aligned_source = Structure(
            data=ma.masked_array(rotated_data, mask=~rotated_mask),
            pixel_size=source.pixel_size
        )

        # Center the structures
        aligned_source = center_structures(aligned_source, target)[0]

        # Return aligned structure and info
        method_info = {
            'rotation_angle': rotation_angle,
            'rotation_method': rotation_method,
            'center_adjusted': True
        }

        return aligned_source, method_info

    except Exception as e:
        raise AlignmentError(f"Edge-based alignment failed: {str(e)}")


def align_by_mask(source: Structure,
                  target: Structure,
                  **kwargs) -> Tuple[Structure, Dict]:
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
            source_mask.astype(float),
            target_mask.astype(float),
            **kwargs
        )

        # Apply translation to source
        from scipy.ndimage import shift as apply_shift

        shifted_data = apply_shift(
            source.data.filled(0),
            shift=shift,
            mode='constant',
            cval=0
        )

        shifted_mask = apply_shift(
            source_mask,
            shift=shift,
            mode='constant',
            cval=0
        ) > 0.5  # Threshold to get boolean mask

        # Create aligned structure
        aligned_source = Structure(
            data=ma.masked_array(shifted_data, mask=~shifted_mask),
            pixel_size=source.pixel_size
        )

        # Return aligned structure and info
        method_info = {
            'translation_shift': shift.tolist(),
            'correlation_error': float(error),
            'phase_difference': float(diffphase)
        }

        return aligned_source, method_info

    except Exception as e:
        if isinstance(e, AlignmentError):
            raise
        raise AlignmentError(f"Mask-based alignment failed: {str(e)}")


def align_by_features(source: Structure,
                      target: Structure,
                      **kwargs) -> Tuple[Structure, Dict]:
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
        from ..processing.detection import detect_corners

        # Detect corners
        source_corners = detect_corners(source.data, **kwargs)
        target_corners = detect_corners(target.data, **kwargs)

        if len(source_corners) < 3 or len(target_corners) < 3:
            raise AlignmentError("Not enough corners detected for feature-based alignment")

        # Match corners and estimate transform
        try:
            from skimage.transform import estimate_transform
        except ImportError:
            raise AlignmentError("Feature-based alignment requires scikit-image")

        # Simple matching - this is a naive approach that assumes corners
        # are detected in similar order. For a real application, use RANSAC
        # or a more robust matching algorithm.
        n_points = min(len(source_corners), len(target_corners), 10)  # Limit to 10 points

        # Estimate similarity transform (scale, rotation, translation)
        transform = estimate_transform(
            'similarity',
            src=source_corners[:n_points],
            dst=target_corners[:n_points]
        )

        # Apply transform to source
        try:
            from skimage.transform import warp
        except ImportError:
            raise AlignmentError("Feature-based alignment requires scikit-image")

        warped_data = warp(
            source.data.filled(0),
            inverse_map=transform.inverse,
            mode='constant',
            cval=0
        )

        warped_mask = warp(
            (~source.data.mask).astype(float),
            inverse_map=transform.inverse,
            mode='constant',
            cval=0
        ) > 0.5  # Threshold to get boolean mask

        # Create aligned structure
        aligned_source = Structure(
            data=ma.masked_array(warped_data, mask=~warped_mask),
            pixel_size=source.pixel_size
        )

        # Return aligned structure and info
        method_info = {
            'transform_matrix': transform.params.tolist(),
            'transform_type': 'similarity',
            'num_matched_corners': n_points
        }

        return aligned_source, method_info

    except Exception as e:
        if isinstance(e, AlignmentError):
            raise
        raise AlignmentError(f"Feature-based alignment failed: {str(e)}")


def calculate_rotation(source_edges: np.ndarray,
                       target_edges: np.ndarray,
                       method: Literal['hough', 'minimize', 'radon'] = 'hough',
                       **kwargs) -> float:
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
        if method == 'hough':
            return calculate_rotation_hough(source_edges, target_edges, **kwargs)
        elif method == 'minimize':
            return calculate_rotation_minimize(source_edges, target_edges, **kwargs)
        elif method == 'radon':
            return calculate_rotation_radon(source_edges, target_edges, **kwargs)
        else:
            raise ValidationError(f"Unknown rotation calculation method: {method}")

    except Exception as e:
        if isinstance(e, (AlignmentError, ValidationError)):
            raise
        raise AlignmentError(f"Rotation calculation failed: {str(e)}")


def calculate_rotation_hough(source_edges: np.ndarray,
                             target_edges: np.ndarray,
                             **kwargs) -> float:
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


def calculate_rotation_minimize(source_edges: np.ndarray,
                                target_edges: np.ndarray,
                                angle_range: Tuple[float, float] = (-30, 30),
                                angle_step: float = 0.5,
                                **kwargs) -> float:
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
            rotated = rotate(source_edges, angle, reshape=False, mode='constant', cval=0)

            # Calculate normalized cross-correlation
            score = np.sum(rotated * target_edges) / np.sqrt(np.sum(rotated ** 2) * np.sum(target_edges ** 2))

            # Update best angle
            if score > best_score:
                best_score = score
                best_angle = angle

        return best_angle

    except Exception as e:
        raise AlignmentError(f"Minimization rotation calculation failed: {str(e)}")


def calculate_rotation_radon(source_edges: np.ndarray,
                             target_edges: np.ndarray,
                             angle_range: Tuple[float, float] = (-90, 90),
                             **kwargs) -> float:
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


def normalize_pixel_size(data: np.ndarray,
                         source_pixel_size: Tuple[float, float],
                         target_pixel_size: Tuple[float, float]) -> Tuple[np.ndarray, Tuple[float, float]]:
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

        resized_data = zoom(data, (zoom_y, zoom_x), mode='reflect')

        return resized_data, (zoom_y, zoom_x)

    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise AlignmentError(f"Pixel size normalization failed: {str(e)}")


def resize_mask(mask: np.ndarray,
                zoom_factors: Tuple[float, float]) -> np.ndarray:
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
        resized = zoom(mask.astype(float), zoom_factors, mode='constant', cval=0)

        # Threshold to get binary mask
        return resized > 0.5

    except Exception as e:
        raise AlignmentError(f"Mask resizing failed: {str(e)}")


def pad_to_same_size(image1: np.ndarray,
                     image2: np.ndarray,
                     mode: str = 'constant',
                     constant_value: float = 0) -> Tuple[np.ndarray, np.ndarray]:
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
            constant_values=constant_value
        )

        padded2 = np.pad(
            image2,
            ((pad2_top, pad2_bottom), (pad2_left, pad2_right)),
            mode=mode,
            constant_values=constant_value
        )

        return padded1, padded2

    except Exception as e:
        raise AlignmentError(f"Image padding failed: {str(e)}")


def center_structures(source: Structure,
                      target: Structure) -> Tuple[Structure, Structure]:
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
            source.data.filled(0),
            shift=source_shift,
            mode='constant',
            cval=0
        )

        shifted_source_mask = apply_shift(
            (~source.data.mask).astype(float),
            shift=source_shift,
            mode='constant',
            cval=0
        ) > 0.5

        # Shift target
        shifted_target_data = apply_shift(
            target.data.filled(0),
            shift=target_shift,
            mode='constant',
            cval=0
        )

        shifted_target_mask = apply_shift(
            (~target.data.mask).astype(float),
            shift=target_shift,
            mode='constant',
            cval=0
        ) > 0.5

        # Create centered structures
        centered_source = Structure(
            data=ma.masked_array(shifted_source_data, mask=~shifted_source_mask),
            pixel_size=source.pixel_size
        )

        centered_target = Structure(
            data=ma.masked_array(shifted_target_data, mask=~shifted_target_mask),
            pixel_size=target.pixel_size
        )

        return centered_source, centered_target

    except Exception as e:
        raise AlignmentError(f"Structure centering failed: {str(e)}")