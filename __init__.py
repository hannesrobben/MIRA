"""
Image Processing Library

A comprehensive library for processing, analyzing, and visualizing image data,
with a focus on holographic microscopy and phase/height data.

This library provides tools for structure detection, segmentation, alignment,
phase retrieval, and visualization of various types of image data.
"""

__version__ = '0.1.0'
__author__ = 'Hannes Robben'

# Import main interfaces
from .processor import StructureProcessor

# Import core components
from .core.image import ImageData
from .core.structure import Structure, Segment, Background

# Import exceptions for better error handling
from .utils.exceptions import (
    DataLoadError, ValidationError, ProcessingError,
    StructureError, SegmentationError, AlignmentError
)

# Define public API
__all__ = [
    # Main processor
    'StructureProcessor',

    # Core classes
    'ImageData', 'Structure', 'Segment', 'Background',

    # Exceptions
    'DataLoadError', 'ValidationError', 'ProcessingError',
    'StructureError', 'SegmentationError', 'AlignmentError'
]