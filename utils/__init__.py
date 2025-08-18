"""
Utility functions and helpers for the image processing library.

This module provides common utilities, helper functions, and exception classes
used throughout the library.
"""

from .exceptions import (
    DataLoadError, ValidationError, ProcessingError,
    StructureError, SegmentationError, AlignmentError
)

from .helpers import (
    get_logger, normalize_pixel_size, ensure_numpy_array,
    butterworth_filter, compute_cross_product, point_in_quadrilateral
)

__all__ = [
    'DataLoadError', 'ValidationError', 'ProcessingError',
    'StructureError', 'SegmentationError', 'AlignmentError',
    'get_logger', 'normalize_pixel_size', 'ensure_numpy_array',
    'butterworth_filter', 'compute_cross_product', 'point_in_quadrilateral'
]