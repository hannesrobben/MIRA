"""
Exception classes for the image processing library.

This module contains custom exception classes used throughout the library
to provide more specific error information to users.
"""


class DataLoadError(Exception):
    """
    Exception raised when data loading fails.

    This could be due to file not found, format issues, or corruption.
    """
    pass


class ValidationError(Exception):
    """
    Exception raised when input validation fails.

    This includes type checking, range validation, or other constraints.
    """
    pass


class ProcessingError(Exception):
    """
    Exception raised when image processing operations fail.

    This is a general error for processing failures.
    """
    pass


class StructureError(Exception):
    """
    Exception raised when structure detection or manipulation fails.

    This includes failures in structure creation, bounds calculation, etc.
    """
    pass


class SegmentationError(Exception):
    """
    Exception raised when structure segmentation fails.

    This could be due to algorithm failures, invalid parameters, etc.
    """
    pass


class AlignmentError(Exception):
    """
    Exception raised when image alignment fails.

    This includes failures in finding reference points, transformation, etc.
    """
    pass


class PhaseRetrievalError(Exception):
    """
    Exception raised when phase retrieval operations fail.

    This includes phase unwrapping, conversion issues, etc.
    """
    pass


class VisualizationError(Exception):
    """
    Exception raised when visualization operations fail.

    This could be due to missing libraries, invalid parameters, etc.
    """
    pass