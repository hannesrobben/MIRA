"""
Core image data representation module.

This module provides the base class for all image data types in the library.
It handles loading, validation, and basic manipulation of image data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Union, Literal
from dataclasses import dataclass, field
from enum import Enum
from scipy.ndimage import shift, zoom


class Axis(Enum):
    """Enumeration for axis selection."""

    X = "X"
    Y = "Y"


class DataLoadError(Exception):
    """Exception raised for errors during data loading."""

    pass


class ValidationError(Exception):
    """Exception raised for data validation errors."""

    pass


class ProcessingError(Exception):
    """Exception raised for errors during data processing."""

    pass


@dataclass
class ImageData:
    """
    Base class for image data representation.

    This class provides functionality for handling 2D image data with associated
    pixel size information. It includes methods for basic image manipulation,
    data access, and file operations.

    Attributes:
        data (np.ndarray): The underlying numpy array containing image data
        pixel_size (Tuple[float, float]): Size of pixels in x and y directions (width, height)
    """

    data: np.ndarray
    pixel_size: Tuple[float, float] = field(default=(1.0, 1.0))
    _flipped: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Validate input data after initialization and store original data."""
        self._validate_input()
        # Store a copy of the original data and pixel size for reference
        self.original_data = self.data.copy()
        self.original_pixel_size = self.pixel_size

    def _validate_input(self) -> None:
        """
        Validate input data and pixel size.

        Raises:
            ValidationError: If data or pixel_size fail validation
        """
        # Validate data
        if not isinstance(self.data, np.ndarray):
            raise ValidationError("Data must be a numpy array")
        if self.data.size == 0:
            raise ValidationError("Data array cannot be empty")
        if self.data.ndim != 2:
            raise ValidationError("Data must be a 2D array")

        # Normalize and validate pixel_size
        if not isinstance(self.pixel_size, tuple) or len(self.pixel_size) != 2:
            if isinstance(self.pixel_size, (int, float)):
                self.pixel_size = (self.pixel_size, self.pixel_size)
            elif isinstance(self.pixel_size, list) and len(self.pixel_size) == 2:
                self.pixel_size = tuple(self.pixel_size)
            elif isinstance(self.pixel_size, np.ndarray):
                self.pixel_size = (self.pixel_size[0], self.pixel_size[1])
            else:
                raise ValidationError(
                    "Pixel size must be a tuple of two floats, a single number, or a list of two numbers"
                )

        # Validate pixel size values
        if not all(isinstance(x, (int, float)) and x > 0 for x in self.pixel_size):
            raise ValidationError("Pixel size values must be positive numbers")

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get array shape."""
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        """Get array data type."""
        return self.data.dtype

    @property
    def x_dim(self) -> int:
        """Get x dimension size (width)."""
        return self.shape[1]

    @property
    def y_dim(self) -> int:
        """Get y dimension size (height)."""
        return self.shape[0]

    @property
    def size(self) -> int:
        """Get total array size."""
        return self.data.size

    @property
    def is_quadratic(self) -> bool:
        """Check if pixels are quadratic (same size in x and y)."""
        return self.pixel_size[0] == self.pixel_size[1]

    def update_data(self, data, pixel_size=None):
        self.data = data
        if pixel_size is not None:
            self.pixel_size = pixel_size

    @classmethod
    def from_file(
        cls, file_path: Union[str, Path], pixel_size: Tuple[float, float] = (1.0, 1.0)
    ) -> ImageData:
        """
        Create instance from a file.

        Args:
            file_path: Path to the data file
            pixel_size: Size of pixels in x and y directions

        Returns:
            ImageData instance

        Raises:
            DataLoadError: If file cannot be loaded
        """
        try:
            file_path = Path(file_path)

            # Handle different file types
            if file_path.suffix.lower() == ".csv":
                data = pd.read_csv(file_path, delimiter=";").to_numpy()
            elif file_path.suffix.lower() == ".npy":
                data = np.load(file_path)
            else:
                # Default to numpy's loadtxt for other formats
                data = np.loadtxt(file_path)

            return cls(data=data, pixel_size=pixel_size)
        except Exception as e:
            raise DataLoadError(f"Failed to load data from {file_path}: {str(e)}")

    def flip(self) -> None:
        """
        Transpose array data (flip x and y dimensions).

        This operation is performed in-place.
        """
        self.data = np.transpose(self.data)
        self._flipped = not self._flipped
        # Also flip pixel size to maintain physical dimensions
        self.pixel_size = (self.pixel_size[1], self.pixel_size[0])

    def get_profile(
        self, axis: Union[Axis, str], location: Union[Literal["mid"], int]
    ) -> np.ndarray:
        """
        Get profile along specified axis at given location.

        Args:
            axis: Axis along which to get profile (Axis.X, Axis.Y, "X", or "Y")
            location: Position or 'mid' for middle

        Returns:
            1D array of profile data

        Raises:
            ValueError: If location is invalid
            ProcessingError: If profile extraction fails
        """
        try:
            # Convert string axis to Enum if needed
            if isinstance(axis, str):
                axis = Axis[axis.upper()]

            if axis == Axis.X:
                if isinstance(location, int):
                    if not 0 <= location < self.y_dim:
                        raise ValueError(
                            f"Y location must be between 0 and {self.y_dim - 1}"
                        )
                    return self.data[location, :]
                elif isinstance(location, str) and location.lower() == "mid":
                    return self.data[self.y_dim // 2, :]
                else:
                    raise ValueError(f"Invalid location: {location}")
            else:  # Axis.Y
                if isinstance(location, int):
                    if not 0 <= location < self.x_dim:
                        raise ValueError(
                            f"X location must be between 0 and {self.x_dim - 1}"
                        )
                    return self.data[:, location]
                elif isinstance(location, str) and location.lower() == "mid":
                    return self.data[:, self.x_dim // 2]
                else:
                    raise ValueError(f"Invalid location: {location}")
        except Exception as e:
            raise ProcessingError(f"Error getting profile: {str(e)}")

    def center_data(self, center_point: Tuple[float, float]) -> None:
        """
        Center the data around a specified point.

        Args:
            center_point: (y, x) coordinates of the point to center

        Raises:
            ValueError: If center_point is not valid
        """
        if len(center_point) != 2:
            raise ValueError("Center point must have 2 dimensions (y, x)")

        # Calculate shift vector
        matrix_center = np.array(self.shape) / 2
        shift_vector = matrix_center - center_point

        # Apply shift (in-place)
        self.data = shift(self.data, shift_vector, mode="constant", cval=0)

    def pad_to_size(
        self,
        target_shape: Tuple[int, int],
        mode: str = "constant",
        constant_value: float = 0.0,
    ) -> None:
        """
        Pad the image to the target shape.

        Args:
            target_shape: (height, width) target shape
            mode: Padding mode (see numpy.pad documentation)
            constant_value: Value to use for padding in constant mode

        Raises:
            ValueError: If target_shape is smaller than current shape
        """
        current_shape = self.shape

        if any(t < c for t, c in zip(target_shape, current_shape)):
            raise ValueError(
                "Target shape must be greater than or equal to current shape"
            )

        # Calculate padding widths
        pad_y = target_shape[0] - current_shape[0]
        pad_x = target_shape[1] - current_shape[1]

        # Calculate padding for each side
        pad_top = pad_y // 2
        pad_bottom = pad_y - pad_top
        pad_left = pad_x // 2
        pad_right = pad_x - pad_left

        # Apply padding
        self.data = np.pad(
            self.data,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode=mode,
            constant_values=constant_value,
        )

    @staticmethod
    def pad_to_same_size(
        image1: Union[ImageData, np.ndarray],
        image2: Union[ImageData, np.ndarray],
        mode: str = "constant",
        constant_value: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pad two images to make them the same size.

        Args:
            image1: First image
            image2: Second image
            mode: Padding mode
            constant_value: Value to use for padding in constant mode

        Returns:
            Tuple of padded arrays (padded_image1, padded_image2)

        Raises:
            TypeError: If inputs are not valid image types
        """
        # Extract numpy arrays if ImageData objects were provided
        array1 = image1.data if isinstance(image1, ImageData) else image1
        array2 = image2.data if isinstance(image2, ImageData) else image2

        if not (isinstance(array1, np.ndarray) and isinstance(array2, np.ndarray)):
            raise TypeError("Both inputs must be numpy arrays or ImageData objects")

        # Find maximum dimensions
        max_height = max(array1.shape[0], array2.shape[0])
        max_width = max(array1.shape[1], array2.shape[1])

        # Create output arrays
        result1 = np.zeros((max_height, max_width), dtype=array1.dtype)
        result2 = np.zeros((max_height, max_width), dtype=array2.dtype)

        # Calculate padding
        pad_height1 = (max_height - array1.shape[0]) // 2
        pad_width1 = (max_width - array1.shape[1]) // 2
        pad_height2 = (max_height - array2.shape[0]) // 2
        pad_width2 = (max_width - array2.shape[1]) // 2

        # Apply padding by copying data to the center of the output arrays
        result1[
            pad_height1 : pad_height1 + array1.shape[0],
            pad_width1 : pad_width1 + array1.shape[1],
        ] = array1
        result2[
            pad_height2 : pad_height2 + array2.shape[0],
            pad_width2 : pad_width2 + array2.shape[1],
        ] = array2

        return result1, result2

    def resize(self, target_pixel_size: Tuple[float, float]) -> None:
        """
        Resize the image to match a target pixel size.

        Args:
            target_pixel_size: Target pixel size (width, height)

        Raises:
            ValueError: If pixel sizes are invalid
        """
        if any(p <= 0 for p in target_pixel_size):
            raise ValueError("Pixel sizes must be positive")

        # Calculate zoom factors
        zoom_y = self.pixel_size[1] / target_pixel_size[1]
        zoom_x = self.pixel_size[0] / target_pixel_size[0]

        # Apply zoom
        self.data = zoom(self.data, (zoom_y, zoom_x), mode="reflect")

        # Update pixel size
        self.pixel_size = target_pixel_size

    def save(self, file_path: Union[str, Path], format: str = "npy") -> None:
        """
        Save the image data to a file.

        Args:
            file_path: Path to save the file
            format: File format ('npy', 'csv', or 'txt')

        Raises:
            ValueError: If format is not supported
        """
        file_path = Path(file_path)

        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "npy":
            np.save(file_path, self.data)
        elif format.lower() == "csv":
            pd.DataFrame(self.data).to_csv(file_path, sep=";", index=False)
        elif format.lower() == "txt":
            np.savetxt(file_path, self.data)
        else:
            raise ValueError(
                f"Unsupported format: {format}. Use 'npy', 'csv', or 'txt'."
            )

        # Save pixel size information
        pixel_info_path = file_path.with_name(f"{file_path.stem}_pixel_info.txt")
        with open(pixel_info_path, "w") as f:
            f.write(f"pixel_size = {self.pixel_size}\n")
