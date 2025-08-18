"""
Core components for the image processing library.

This module contains the foundational classes for representing and
manipulating image data and structures.
"""

from .image import ImageData
from .structure import Structure, Segment, Background

__all__ = ['ImageData', 'Structure', 'Segment', 'Background']