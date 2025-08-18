"""
Structure representation module.

This module provides classes for representing structures detected in images,
including the main Structure class and its components (Segment, Background).
"""

from __future__ import annotations

import numpy as np
import numpy.ma as ma
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Union

class StructureError(Exception):
    """Exception raised for errors related to structure operations."""
    pass


@dataclass
class Background:
    """
    Class representing the background area of an image.

    The background is the area outside of the detected structure.

    Attributes:
        data (ma.MaskedArray): Masked array containing background data
        pixel_size (Tuple[float, float]): Size of pixels in x and y directions
    """
    data: ma.MaskedArray
    pixel_size: Tuple[float, float] = field(default=(1.0, 1.0))

    def __post_init__(self):
        """Validate input data after initialization."""
        self._validate_input()

    def _validate_input(self):
        """Validate input data and pixel size."""
        if not isinstance(self.data, ma.MaskedArray):
            raise StructureError("Background data must be a masked array")
        if not isinstance(self.pixel_size, tuple) or len(self.pixel_size) != 2:
            raise StructureError("Pixel size must be a tuple of two floats")
        if not all(isinstance(x, (int, float)) and x > 0 for x in self.pixel_size):
            raise StructureError("Pixel size values must be positive numbers")

    @property
    def mean_value(self) -> float:
        """Get the mean value of the background."""
        return float(self.data.mean())

    @property
    def std_value(self) -> float:
        """Get the standard deviation of the background."""
        return float(self.data.std())

    @property
    def mask(self) -> np.ndarray:
        """Get the background mask (True where background exists)."""
        return ~self.data.mask

    def get_margin_mask(self, margin_size: int = 5) -> np.ndarray:
        """
        Get a mask for the margin area around the background.

        Args:
            margin_size: Size of the margin in pixels

        Returns:
            Boolean mask array (True for margin pixels)
        """
        from scipy.ndimage import binary_erosion

        # Create a mask that's True for background pixels
        bg_mask = ~self.data.mask

        # Erode the mask by margin_size
        eroded_mask = binary_erosion(bg_mask, iterations=margin_size)

        # The margin is the difference between the original and eroded masks
        margin_mask = bg_mask & ~eroded_mask

        return margin_mask


@dataclass
class Segment:
    """
    Class representing a segment of a structure.

    A segment is a portion of the structure, usually divided based on geometry
    or functionality.

    Attributes:
        data (ma.MaskedArray): Masked array containing segment data
        pixel_size (Tuple[float, float]): Size of pixels in x and y directions
        id (int): Segment identifier
        bounds (Tuple[slice, slice]): Segment boundaries as slices
    """
    data: ma.MaskedArray
    pixel_size: Tuple[float, float] = field(default=(1.0, 1.0))
    id: int = field(default=0)
    bounds: Tuple[slice, slice] = field(default=None, init=False)

    def __post_init__(self):
        """Initialize and validate the segment."""
        self._validate_input()
        self._calculate_bounds()

    def _validate_input(self):
        """Validate input data and pixel size."""
        if not isinstance(self.data, ma.MaskedArray):
            raise StructureError("Segment data must be a masked array")
        if not isinstance(self.pixel_size, tuple) or len(self.pixel_size) != 2:
            raise StructureError("Pixel size must be a tuple of two floats")
        if not all(isinstance(x, (int, float)) and x > 0 for x in self.pixel_size):
            raise StructureError("Pixel size values must be positive numbers")
        if not isinstance(self.id, int):
            raise StructureError("Segment ID must be an integer")

    def _calculate_bounds(self):
        """Calculate the segment's bounding box."""
        if self.data.mask.all():  # If all data is masked
            self.bounds = (slice(0, 0), slice(0, 0))
            return

        # Get the indices where the mask is False (i.e., where data exists)
        rows, cols = np.where(~self.data.mask)

        if len(rows) == 0 or len(cols) == 0:
            self.bounds = (slice(0, 0), slice(0, 0))
            return

        # Calculate bounds
        min_row, max_row = rows.min(), rows.max() + 1
        min_col, max_col = cols.min(), cols.max() + 1

        self.bounds = (slice(min_row, max_row), slice(min_col, max_col))

    @property
    def inner_data(self) -> ma.MaskedArray:
        """
        Get the inner region of the segment, excluding edge pixels.

        Returns:
            MaskedArray with only inner region data
        """
        from scipy.ndimage import binary_erosion

        # Create a mask of valid data points
        valid_mask = ~self.data.mask

        # Erode to get inner region
        inner_mask = binary_erosion(valid_mask, iterations=1)

        # Create a new masked array with inner region only
        result = ma.masked_array(self.data.data, mask=~inner_mask)
        return result

    @property
    def mean_value(self) -> float:
        """Get the mean value of the segment."""
        return float(self.data.mean())

    @property
    def area(self) -> float:
        """
        Get the segment area in physical units.

        Returns:
            Area in square units (based on pixel_size)
        """
        # Count non-masked pixels
        pixel_count = np.sum(~self.data.mask)

        # Convert to physical units
        physical_area = pixel_count * self.pixel_size[0] * self.pixel_size[1]

        return physical_area

    def extract_with_offset(self, offset_percent: float = 10) -> ma.MaskedArray:
        """
        Extract segment data with an offset from the edges.

        Args:
            offset_percent: Percentage of segment dimensions to exclude

        Returns:
            Masked array with data from inner region
        """
        if self.data.mask.all():
            return self.data.copy()

        # Get non-masked coordinates
        rows, cols = np.where(~self.data.mask)

        if len(rows) == 0:
            return self.data.copy()

        # Calculate offsets
        row_offset = int((max(rows) - min(rows)) * offset_percent / 100)
        col_offset = int((max(cols) - min(cols)) * offset_percent / 100)

        # Create inner mask
        inner_mask = ~self.data.mask.copy()

        # Zero out the edges
        if row_offset > 0:
            inner_mask[min(rows):min(rows) + row_offset, :] = False
            inner_mask[max(rows) - row_offset:max(rows) + 1, :] = False

        if col_offset > 0:
            inner_mask[:, min(cols):min(cols) + col_offset] = False
            inner_mask[:, max(cols) - col_offset:max(cols) + 1] = False

        # Create result masked array
        result = ma.masked_array(self.data.data, mask=~inner_mask)
        return result


@dataclass
class Structure:
    """
    Class representing a structure detected in an image.

    A structure is a cohesive region of interest in an image, which may be
    further divided into segments.

    Attributes:
        data (ma.MaskedArray): Masked array containing structure data
        pixel_size (Tuple[float, float]): Size of pixels in x and y directions
        segments (List[Segment]): List of segments that make up the structure
    """
    data: ma.MaskedArray
    pixel_size: Tuple[float, float] = field(default=(1.0, 1.0))
    segments: List[Segment] = field(default_factory=list)
    _center: Optional[Tuple[float, float]] = field(default=None, init=False)
    _bounds: Optional[Tuple[slice, slice]] = field(default=None, init=False)
    _corner_points: Optional[np.ndarray] = field(default=None, init=False)

    def __post_init__(self):
        """Initialize and validate the structure."""
        self._validate_input()

    def _validate_input(self):
        """Validate input data and pixel size."""
        if not isinstance(self.data, ma.MaskedArray):
            raise StructureError("Structure data must be a masked array")
        if not isinstance(self.pixel_size, tuple) or len(self.pixel_size) != 2:
            raise StructureError("Pixel size must be a tuple of two floats")
        if not all(isinstance(x, (int, float)) and x > 0 for x in self.pixel_size):
            raise StructureError("Pixel size values must be positive numbers")

    @classmethod
    def from_mask(cls, data: np.ndarray, mask: np.ndarray,
                 pixel_size: Tuple[float, float] = (1.0, 1.0)) -> Structure:
        """
        Create a Structure instance from data and a boolean mask.

        Args:
            data: Raw image data
            mask: Boolean mask (True where structure exists)
            pixel_size: Pixel size in physical units

        Returns:
            Structure instance
        """
        masked_data = ma.masked_array(data, mask=~mask)
        return cls(data=masked_data, pixel_size=pixel_size)

    @property
    def mask(self) -> np.ndarray:
        """Get the structure mask (True where structure exists)."""
        return ~self.data.mask

    @property
    def center(self) -> Tuple[float, float]:
        """
        Get the center of mass of the structure.

        Returns:
            (y, x) coordinates of the center
        """
        if self._center is not None:
            return self._center

        # Calculate center of mass
        y_indices, x_indices = np.where(~self.data.mask)

        if len(y_indices) == 0:
            # Return center of image if no structure is found
            self._center = (self.data.shape[0] // 2, self.data.shape[1] // 2)
        else:
            self._center = (float(y_indices.mean()), float(x_indices.mean()))

        return self._center

    @property
    def bounds(self) -> Tuple[slice, slice]:
        """
        Get the bounding box of the structure as slices.

        Returns:
            (y_slice, x_slice) for accessing the bounded region
        """
        if self._bounds is not None:
            return self._bounds

        # Find bounding box
        y_indices, x_indices = np.where(~self.data.mask)

        if len(y_indices) == 0:
            self._bounds = (slice(0, 0), slice(0, 0))
        else:
            min_y, max_y = y_indices.min(), y_indices.max() + 1
            min_x, max_x = x_indices.min(), x_indices.max() + 1
            self._bounds = (slice(min_y, max_y), slice(min_x, max_x))

        return self._bounds

    @property
    def corner_points(self) -> np.ndarray:
        """
        Get the corner points of the structure.

        Returns:
            Array of (y, x) corner coordinates
        """
        if self._corner_points is not None:
            return self._corner_points

        # Use OpenCV to find corners if available
        try:
            import cv2

            # Convert mask to uint8
            mask = (~self.data.mask).astype(np.uint8) * 255

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                raise StructureError("No contours found in the structure mask")

            # Approximate contour with a polygon
            epsilon = 0.02 * cv2.arcLength(contours[0], True)
            approx = cv2.approxPolyDP(contours[0], epsilon, True)

            # Convert to (y, x) format and store
            corners = np.array([(pt[0][1], pt[0][0]) for pt in approx])

            # Sort corners (top-left, top-right, bottom-right, bottom-left)
            corners = self._sort_corners(corners)

            self._corner_points = corners
            return corners

        except ImportError:
            # Fallback method if OpenCV is not available
            y_indices, x_indices = np.where(~self.data.mask)

            if len(y_indices) == 0:
                raise StructureError("Empty structure, no corners found")

            # Find extreme points
            min_y, max_y = y_indices.min(), y_indices.max()
            min_x, max_x = x_indices.min(), x_indices.max()

            # Create corners (top-left, top-right, bottom-right, bottom-left)
            corners = np.array([
                (min_y, min_x),
                (min_y, max_x),
                (max_y, max_x),
                (max_y, min_x)
            ])

            self._corner_points = corners
            return corners

    def _sort_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Sort corners in clockwise order starting from top-left.

        Args:
            corners: Array of (y, x) corner coordinates

        Returns:
            Sorted corners array
        """
        # Calculate center
        center = corners.mean(axis=0)

        # Function to calculate angle with respect to center
        def get_angle(point):
            return np.arctan2(point[0] - center[0], point[1] - center[1])

        # Sort by angle
        sorted_corners = sorted(corners, key=get_angle)

        # Find top-left corner (minimum sum of coordinates)
        top_left_idx = np.argmin([p[0] + p[1] for p in sorted_corners])

        # Reorder to start from top-left
        return np.array(sorted_corners[top_left_idx:] + sorted_corners[:top_left_idx])

    @property
    def area(self) -> float:
        """
        Get the total area of the structure in physical units.

        Returns:
            Area in square units (based on pixel_size)
        """
        # Count non-masked pixels
        pixel_count = np.sum(~self.data.mask)

        # Convert to physical units
        physical_area = pixel_count * self.pixel_size[0] * self.pixel_size[1]

        return physical_area

    @property
    def perimeter(self) -> float:
        """
        Get the perimeter of the structure in physical units.

        Returns:
            Perimeter length in units (based on pixel_size)
        """
        try:
            import cv2

            # Convert mask to uint8
            mask = (~self.data.mask).astype(np.uint8) * 255

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return 0.0

            # Calculate perimeter
            pixel_perimeter = cv2.arcLength(contours[0], True)

            # Convert to physical units (approximation)
            physical_perimeter = pixel_perimeter * (self.pixel_size[0] + self.pixel_size[1]) / 2

            return physical_perimeter

        except ImportError:
            # Fallback method if OpenCV is not available
            from scipy.ndimage import binary_erosion

            # Create a mask of valid data points
            valid_mask = ~self.data.mask

            # Erode by one pixel
            eroded_mask = binary_erosion(valid_mask)

            # Perimeter pixels are in valid_mask but not in eroded_mask
            perimeter_pixels = np.sum(valid_mask & ~eroded_mask)

            # Convert to physical units (rough approximation)
            physical_perimeter = perimeter_pixels * (self.pixel_size[0] + self.pixel_size[1]) / 2

            return physical_perimeter

    def segment(self, num_segments: Tuple[int, int] = (1, 1),
               method: str = 'grid') -> List[Segment]:
        """
        Segment the structure into multiple parts.

        Args:
            num_segments: (rows, cols) number of segments to create
            method: Segmentation method ('grid', 'kmeans', or 'watershed')

        Returns:
            List of Segment objects

        Raises:
            StructureError: If segmentation fails
        """
        if method.lower() == 'grid':
            segments = self._segment_grid(num_segments)
        elif method.lower() == 'kmeans':
            segments = self._segment_kmeans(num_segments)
        elif method.lower() == 'watershed':
            segments = self._segment_watershed(num_segments)
        else:
            raise StructureError(f"Unknown segmentation method: {method}")

        self.segments = segments
        return segments

    def _segment_grid(self, num_segments: Tuple[int, int]) -> List[Segment]:
        """
        Segment the structure using a grid-based approach.

        Args:
            num_segments: (rows, cols) number of segments to create

        Returns:
            List of Segment objects
        """
        n_rows, n_cols = num_segments

        if n_rows <= 0 or n_cols <= 0:
            raise StructureError("Number of rows and columns must be positive")

        # Get corner points
        try:
            corners = self.corner_points
        except StructureError:
            # Fallback to bounding box if corners cannot be found
            y_indices, x_indices = np.where(~self.data.mask)
            min_y, max_y = y_indices.min(), y_indices.max()
            min_x, max_x = x_indices.min(), x_indices.max()
            corners = np.array([
                (min_y, min_x),
                (min_y, max_x),
                (max_y, max_x),
                (max_y, min_x)
            ])

        # Extract coordinates
        tl, tr, br, bl = corners

        # Create segments
        segments = []
        segment_id = 0

        # Original image shape
        height, width = self.data.shape

        # Create a labeled array for the segments
        labeled_array = np.full(self.data.shape, -1, dtype=np.int32)
        structure_mask = ~self.data.mask

        for row in range(n_rows):
            for col in range(n_cols):
                # Calculate segment boundaries (interpolating between corner points)
                top_left_y = int(tl[0] + (bl[0] - tl[0]) * row / n_rows)
                top_left_x = int(tl[1] + (tr[1] - tl[1]) * col / n_cols)

                bottom_right_y = int(tl[0] + (bl[0] - tl[0]) * (row + 1) / n_rows)
                bottom_right_x = int(tl[1] + (tr[1] - tl[1]) * (col + 1) / n_cols)

                # Ensure within image bounds
                top_left_y = max(0, min(top_left_y, height - 1))
                top_left_x = max(0, min(top_left_x, width - 1))
                bottom_right_y = max(0, min(bottom_right_y, height - 1))
                bottom_right_x = max(0, min(bottom_right_x, width - 1))

                # Create segment mask
                segment_mask = np.zeros(self.data.shape, dtype=bool)
                segment_mask[top_left_y:bottom_right_y+1, top_left_x:bottom_right_x+1] = True

                # Intersect with structure mask
                segment_mask = segment_mask & structure_mask

                # If segment contains structure pixels
                if np.any(segment_mask):
                    # Update labeled array
                    labeled_array[segment_mask] = segment_id

                    # Create segment
                    segment_data = ma.masked_array(self.data.data, mask=~segment_mask)
                    segment = Segment(data=segment_data, pixel_size=self.pixel_size, id=segment_id)
                    segments.append(segment)
                    segment_id += 1

        return segments

    def _segment_kmeans(self, num_segments: Tuple[int, int]) -> List[Segment]:
        """
        Segment the structure using K-means clustering.

        Args:
            num_segments: (rows, cols) total number of segments is rows*cols

        Returns:
            List of Segment objects
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise StructureError("K-means segmentation requires scikit-learn")

        n_rows, n_cols = num_segments
        n_clusters = n_rows * n_cols

        # Get coordinates of structure pixels
        y_indices, x_indices = np.where(~self.data.mask)

        if len(y_indices) == 0:
            return []

        # Prepare data for clustering
        points = np.column_stack((y_indices, x_indices))

        # Apply K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(points)

        # Create segments
        segments = []
        for i in range(n_clusters):
            # Create mask for this segment
            segment_mask = np.zeros(self.data.shape, dtype=bool)
            segment_indices = (labels == i)
            segment_y = y_indices[segment_indices]
            segment_x = x_indices[segment_indices]

            if len(segment_y) == 0:
                continue

            segment_mask[segment_y, segment_x] = True

            # Create segment
            segment_data = ma.masked_array(self.data.data, mask=~segment_mask)
            segment = Segment(data=segment_data, pixel_size=self.pixel_size, id=i)
            segments.append(segment)

        return segments

    def _segment_watershed(self, num_segments: Tuple[int, int]) -> List[Segment]:
        """
        Segment the structure using watershed algorithm.

        Args:
            num_segments: (rows, cols) number of segments to create

        Returns:
            List of Segment objects
        """
        try:
            from skimage.segmentation import watershed
            from scipy.ndimage import distance_transform_edt
        except ImportError:
            raise StructureError("Watershed segmentation requires scikit-image")

        n_rows, n_cols = num_segments
        n_markers = n_rows * n_cols

        # Get structure mask
        structure_mask = ~self.data.mask

        if not np.any(structure_mask):
            return []

        # Calculate distance transform
        distance = distance_transform_edt(structure_mask)

        # Create markers
        y_indices, x_indices = np.where(structure_mask)
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        markers = np.zeros(self.data.shape, dtype=np.int32)
        marker_id = 1

        for i in range(n_rows):
            for j in range(n_cols):
                y = y_min + (y_max - y_min) * (i + 0.5) / n_rows
                x = x_min + (x_max - x_min) * (j + 0.5) / n_cols

                y, x = int(y), int(x)

                # Ensure the marker is within the structure
                if structure_mask[y, x]:
                    markers[y, x] = marker_id
                    marker_id += 1

        # Apply watershed
        labels = watershed(-distance, markers, mask=structure_mask)

        # Create segments
        segments = []
        for i in range(1, marker_id):
            segment_mask = (labels == i)

            if np.any(segment_mask):
                segment_data = ma.masked_array(self.data.data, mask=~segment_mask)
                segment = Segment(data=segment_data, pixel_size=self.pixel_size, id=i-1)
                segments.append(segment)

        return segments

    def adjust_background(self, background: Background) -> None:
        """
        Adjust data by subtracting background level.

        Args:
            background: Background object with level information
        """
        # Get background mean value
        bg_level = background.mean_value

        # Create adjusted data
        adjusted_data = self.data.data - bg_level

        # Update data
        self.data = ma.masked_array(adjusted_data, mask=self.data.mask)

    def extract_profile(self, start_point: Tuple[int, int],
                       end_point: Tuple[int, int]) -> np.ndarray:
        """
        Extract a profile along a line from start to end point.

        Args:
            start_point: (y, x) starting point
            end_point: (y, x) ending point

        Returns:
            1D array of values along the profile
        """
        try:
            from skimage.measure import profile_line
        except ImportError:
            raise StructureError("Profile extraction requires scikit-image")

        # Extract profile
        profile = profile_line(
            self.data.filled(fill_value=np.nan),
            start_point,
            end_point,
            mode='constant',
            cval=np.nan
        )

        return profile