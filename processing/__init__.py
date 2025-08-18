"""
Processing module for the image processing library.

This module provides functions for structure detection, segmentation,
alignment, and phase retrieval operations on image data.
"""

from .detection import (
    detect_structure, detect_edges, detect_corners,
    threshold_detection, intensity_detection, adaptive_detection
)

from .segmentation import (
    segment_structure, segment_by_grid, segment_by_kmeans,
    segment_by_watershed, create_labeled_mask
)

from .alignment import (
    align_structures, calculate_rotation, normalize_pixel_size,
    pad_to_same_size, center_structures
)

from .phase_retrieval import (
    convert_phase_to_height, convert_height_to_phase,
    unwrap_phase, full_phase_retrieval, adjust_background
)

__all__ = [
    # Detection functions
    'detect_structure', 'detect_edges', 'detect_corners',
    'threshold_detection', 'intensity_detection', 'adaptive_detection',

    # Segmentation functions
    'segment_structure', 'segment_by_grid', 'segment_by_kmeans',
    'segment_by_watershed', 'create_labeled_mask',

    # Alignment functions
    'align_structures', 'calculate_rotation', 'normalize_pixel_size',
    'pad_to_same_size', 'center_structures',

    # Phase retrieval functions
    'convert_phase_to_height', 'convert_height_to_phase',
    'unwrap_phase', 'full_phase_retrieval', 'adjust_background'
]