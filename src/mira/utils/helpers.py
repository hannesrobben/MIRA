"""
Helper functions for the image processing library.

This module provides utility functions used across the library
to perform common tasks and operations.
"""

import logging
import os
import numpy as np
from typing import Tuple, List, Union, Optional, Any, Dict
from pathlib import Path


def get_logger(name: str = "image_processing",
               logfile: Optional[str] = None,
               level: int = logging.INFO) -> logging.Logger:
    """
    Set up and return a logger with the given name and level.

    Args:
        name: Name of the logger
        logfile: Optional path to log file
        level: Logging level (default: logging.INFO)

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if logfile is specified
    if logfile:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(logfile)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def ensure_numpy_array(data: Any) -> np.ndarray:
    """
    Convert input data to a numpy array if it isn't already.

    Args:
        data: Input data to convert

    Returns:
        Data as numpy array

    Raises:
        ValueError: If data cannot be converted to numpy array
    """
    if isinstance(data, np.ndarray):
        return data

    try:
        # Try to convert to numpy array
        array = np.asarray(data)

        # Check if conversion was successful
        if array.dtype == np.dtype('O') and array.size > 0:
            # Object arrays usually mean conversion didn't work as expected
            raise ValueError("Could not convert input to valid numeric array")

        return array
    except Exception as e:
        raise ValueError(f"Failed to convert input to numpy array: {str(e)}")


def normalize_pixel_size(pixel_size: Union[float, Tuple[float, float], List[float]]) -> Tuple[float, float]:
    """
    Normalize pixel size to a tuple of two floats.

    Args:
        pixel_size: Pixel size as float, tuple, or list

    Returns:
        Normalized pixel size as tuple (width, height)

    Raises:
        ValueError: If pixel_size cannot be normalized
    """
    if isinstance(pixel_size, (int, float)):
        return (float(pixel_size), float(pixel_size))

    elif isinstance(pixel_size, (list, tuple)):
        if len(pixel_size) == 1:
            val = float(pixel_size[0])
            return (val, val)
        elif len(pixel_size) == 2:
            return (float(pixel_size[0]), float(pixel_size[1]))
        else:
            raise ValueError(f"Expected 1 or 2 values for pixel_size, got {len(pixel_size)}")

    else:
        raise ValueError(f"Expected pixel_size to be number or sequence, got {type(pixel_size).__name__}")


def butterworth_filter(shape: Tuple[int, int], cutoff: float, order: int) -> np.ndarray:
    """
    Create a 2D Butterworth low-pass filter.

    Args:
        shape: Image dimensions (height, width)
        cutoff: Cutoff frequency of the filter
        order: Order of the filter (higher means sharper cutoff)

    Returns:
        2D filter array
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    # Create coordinate matrices
    y, x = np.ogrid[-crow:rows - crow, -ccol:cols - ccol]
    d = np.sqrt(x * x + y * y)

    # Create the filter
    butterworth = 1 / (1 + (d / cutoff) ** (2 * order))

    return butterworth


def compute_cross_product(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Compute the cross product of vectors p1p2 and p1p3.

    A positive value means p3 is counterclockwise from p2 relative to p1.

    Args:
        p1: First point
        p2: Second point
        p3: Third point

    Returns:
        Cross product value
    """
    v1 = p2 - p1
    v2 = p3 - p1
    return np.cross(v1, v2)


def point_in_quadrilateral(point: np.ndarray, quad_points: List[np.ndarray]) -> bool:
    """
    Determine if a point lies inside a quadrilateral using cross products.

    Args:
        point: Point to test [y, x]
        quad_points: List of 4 corner points defining the quadrilateral

    Returns:
        True if point is inside the quadrilateral, False otherwise
    """
    if len(quad_points) != 4:
        raise ValueError("quad_points must contain exactly 4 points")

    def sign(p1, p2, p3):
        # Cross product sign for points in (y, x) format
        return (p2[1] - p1[1]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[1] - p1[1])

    # Check if point is on the same side of all edges
    sign1 = sign(quad_points[0], quad_points[1], point)
    sign2 = sign(quad_points[1], quad_points[2], point)
    sign3 = sign(quad_points[2], quad_points[3], point)
    sign4 = sign(quad_points[3], quad_points[0], point)

    # Point is inside if all signs are positive or all are negative
    return ((sign1 >= 0 and sign2 >= 0 and sign3 >= 0 and sign4 >= 0) or
            (sign1 <= 0 and sign2 <= 0 and sign3 <= 0 and sign4 <= 0))


def find_line_intersection(p1: Tuple[float, float],
                           p2: Tuple[float, float],
                           p3: Tuple[float, float],
                           p4: Tuple[float, float]) -> Optional[np.ndarray]:
    """
    Find the intersection point of two lines defined by points (p1,p2) and (p3,p4).

    Args:
        p1: First point of line 1 (y, x)
        p2: Second point of line 1 (y, x)
        p3: First point of line 2 (y, x)
        p4: Second point of line 2 (y, x)

    Returns:
        Intersection point as [y, x] or None if lines are parallel
    """
    r1, c1 = p1  # row, column for point 1
    r2, c2 = p2  # row, column for point 2
    r3, c3 = p3  # row, column for point 3
    r4, c4 = p4  # row, column for point 4

    denominator = (c1 - c2) * (r3 - r4) - (r1 - r2) * (c3 - c4)

    if denominator == 0:  # Lines are parallel
        return None

    t = ((c1 - c3) * (r3 - r4) - (r1 - r3) * (c3 - c4)) / denominator

    intersection_row = r1 + t * (r2 - r1)
    intersection_col = c1 + t * (c2 - c1)

    return np.array([intersection_row, intersection_col])


def phase_to_height(phase_data: np.ndarray,
                    n_resin: float,
                    wavelength: float = 0.6749e-6) -> np.ndarray:
    """
    Convert phase data to height profile.

    Based on the theoretical investigation by Nguyen et al. (2016)

    Args:
        phase_data: Phase data array
        n_resin: Refractive index of the resin
        wavelength: Light wavelength in meters (default: 0.6749 µm)

    Returns:
        Height profile array
    """
    n_air = 1.0
    delta_n = n_resin - n_air  # change in refractive index

    height_profile = wavelength * phase_data / (2 * np.pi * delta_n)
    return height_profile


def height_to_phase(height_data: np.ndarray,
                    n_resin: float,
                    wavelength: float = 0.6749e-6) -> np.ndarray:
    """
    Convert height profile to phase data.

    Args:
        height_data: Height profile in meters
        n_resin: Refractive index of the resin
        wavelength: Light wavelength in meters (default: 0.6749 µm)

    Returns:
        Phase data array
    """
    n_air = 1.0
    delta_n = n_resin - n_air  # change in refractive index

    phase_data = 2 * np.pi * height_data * delta_n / wavelength
    return phase_data