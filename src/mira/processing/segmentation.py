"""
Structure segmentation module.

This module provides functions for segmenting detected structures into
multiple regions using various methods such as grid-based partitioning,
clustering, and watershed algorithms.
"""

import numpy as np
import numpy.ma as ma
from typing import Union, Optional, Tuple, List, Dict, Literal, Any

from ..core.image import ImageData
from ..core.structure import Structure, Segment
from ..utils.exceptions import ProcessingError, SegmentationError, ValidationError
from ..utils.helpers import find_line_intersection, point_in_quadrilateral


def segment_structure(
    structure: Structure,
    method: Literal["grid", "kmeans", "watershed"] = "grid",
    num_segments: Tuple[int, int] = (2, 2),
    **kwargs,
) -> List[Segment]:
    """
    Segment a structure into multiple parts.

    Args:
        structure: Structure to segment
        method: Segmentation method to use
        num_segments: Number of segments as (rows, cols)
        **kwargs: Additional parameters for the segmentation method

    Returns:
        List of Segment objects

    Raises:
        SegmentationError: If segmentation fails
        ValidationError: If inputs are invalid
    """
    try:
        # Dispatch to appropriate segmentation method
        if method == "grid":
            segments = segment_by_grid(structure, num_segments, **kwargs)
        elif method == "kmeans":
            segments = segment_by_kmeans(structure, num_segments, **kwargs)
        elif method == "watershed":
            segments = segment_by_watershed(structure, num_segments, **kwargs)
        else:
            raise ValidationError(f"Unknown segmentation method: {method}")

        # Update structure's segments, then record the segmentation
        # configuration in structure.info so downstream consumers (e.g.
        # analysis.ensemble) know how the structure was segmented without
        # having to be told again.
        structure.segments = segments
        structure.info["segmentation_method"] = method
        structure.info["segmentation_grid_shape"] = tuple(num_segments)
        structure.refresh_info()

        return segments

    except Exception as e:
        if isinstance(e, (SegmentationError, ValidationError)):
            raise
        raise SegmentationError(f"Structure segmentation failed: {str(e)}")


def segment_by_grid(
    structure: Structure,
    num_segments: Tuple[int, int],
    use_corners: bool = True,
    **kwargs,
) -> List[Segment]:
    """
    Segment a structure using a grid-based approach.

    Args:
        structure: Structure to segment
        num_segments: Number of segments as (rows, cols)
        use_corners: Whether to use corner points for grid
        **kwargs: Additional parameters (ignored)

    Returns:
        List of Segment objects

    Raises:
        SegmentationError: If segmentation fails
    """
    try:
        n_rows, n_cols = num_segments

        if n_rows <= 0 or n_cols <= 0:
            raise ValidationError("Number of rows and columns must be positive")

        # Get structure mask
        structure_mask = ~structure.data.mask

        if not np.any(structure_mask):
            return []

        # Get structure bounds
        height, width = structure.data.shape

        # Create a labeled array for the segments
        labeled_array = np.full(structure.data.shape, -1, dtype=np.int32)

        # If using corners, get corner points
        if use_corners:
            try:
                corners = structure.corner_points

                # Check if we have enough corners
                if len(corners) < 4:
                    raise SegmentationError(
                        "Not enough corners detected for grid segmentation"
                    )

                # Extract corner points (clockwise from top-left)
                tl, tr, br, bl = corners[:4]

            except Exception:
                use_corners = False

        # If not using corners or corner detection failed, use bounding box
        if not use_corners:
            y_indices, x_indices = np.where(structure_mask)
            min_y, max_y = y_indices.min(), y_indices.max()
            min_x, max_x = x_indices.min(), x_indices.max()

            # Create corners (top-left, top-right, bottom-right, bottom-left)
            tl = np.array([min_y, min_x])
            tr = np.array([min_y, max_x])
            br = np.array([max_y, max_x])
            bl = np.array([max_y, min_x])

        # Generate grid points
        col_upper_points = []  # Top edge points (left to right)
        col_lower_points = []  # Bottom edge points (left to right)

        # Vectors for top and bottom edges
        v_top = tr - tl
        v_bottom = br - bl

        # Calculate points along top and bottom edges
        for i in range(n_cols + 1):
            t = i / n_cols
            col_upper_points.append(tl + t * v_top)
            col_lower_points.append(bl + t * v_bottom)

        # Generate row points
        row_left_points = []  # Left edge points (top to bottom)
        row_right_points = []  # Right edge points (top to bottom)

        # Vectors for left and right edges
        v_left = bl - tl
        v_right = br - tr

        # Calculate points along left and right edges
        for i in range(n_rows + 1):
            t = i / n_rows
            row_left_points.append(tl + t * v_left)
            row_right_points.append(tr + t * v_right)

        # Create segments
        segments = []
        segment_id = 0

        for row in range(n_rows):
            for col in range(n_cols):
                # Calculate the four corners of this segment
                segment_corners = [
                    find_line_intersection(
                        row_left_points[row],
                        row_right_points[row],
                        col_upper_points[col],
                        col_lower_points[col],
                    ),
                    find_line_intersection(
                        row_left_points[row],
                        row_right_points[row],
                        col_upper_points[col + 1],
                        col_lower_points[col + 1],
                    ),
                    find_line_intersection(
                        row_left_points[row + 1],
                        row_right_points[row + 1],
                        col_upper_points[col + 1],
                        col_lower_points[col + 1],
                    ),
                    find_line_intersection(
                        row_left_points[row + 1],
                        row_right_points[row + 1],
                        col_upper_points[col],
                        col_lower_points[col],
                    ),
                ]

                # Check if any intersection failed
                if any(corner is None for corner in segment_corners):
                    continue

                # Create mask for this segment
                segment_mask = np.zeros(structure.data.shape, dtype=bool)

                # Find points within the quadrilateral and structure
                for y in range(height):
                    for x in range(width):
                        if structure_mask[y, x]:
                            if point_in_quadrilateral(
                                np.array([y, x]), segment_corners
                            ):
                                segment_mask[y, x] = True
                                labeled_array[y, x] = segment_id

                # Create segment if there are any points
                if np.any(segment_mask):
                    segment_data = ma.masked_array(
                        structure.data.data, mask=~segment_mask
                    )

                    segment = Segment(
                        data=segment_data,
                        pixel_size=structure.pixel_size,
                        id=segment_id,
                    )

                    segments.append(segment)
                    segment_id += 1

        return segments

    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise SegmentationError(f"Grid segmentation failed: {str(e)}")


def segment_by_kmeans(
    structure: Structure, num_segments: Tuple[int, int], random_state: int = 0, **kwargs
) -> List[Segment]:
    """
    Segment a structure using K-means clustering.

    Args:
        structure: Structure to segment
        num_segments: Number of segments as (rows, cols)
        random_state: Random seed for K-means
        **kwargs: Additional parameters for K-means

    Returns:
        List of Segment objects

    Raises:
        SegmentationError: If segmentation fails
    """
    try:
        # Import K-means
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise SegmentationError("K-means segmentation requires scikit-learn")

        n_rows, n_cols = num_segments
        n_clusters = n_rows * n_cols

        # Get structure mask
        structure_mask = ~structure.data.mask

        # Get coordinates of structure pixels
        y_indices, x_indices = np.where(structure_mask)

        if len(y_indices) == 0:
            return []

        # Prepare data for clustering
        points = np.column_stack((y_indices, x_indices))

        # Apply K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, **kwargs)
        labels = kmeans.fit_predict(points)

        # Create segments
        segments = []

        for i in range(n_clusters):
            # Create mask for this segment
            segment_mask = np.zeros(structure.data.shape, dtype=bool)
            segment_indices = labels == i

            if not np.any(segment_indices):
                continue

            segment_y = y_indices[segment_indices]
            segment_x = x_indices[segment_indices]

            segment_mask[segment_y, segment_x] = True

            # Create segment
            segment_data = ma.masked_array(structure.data.data, mask=~segment_mask)

            segment = Segment(data=segment_data, pixel_size=structure.pixel_size, id=i)

            segments.append(segment)

        return segments

    except Exception as e:
        if isinstance(e, (SegmentationError, ValidationError)):
            raise
        raise SegmentationError(f"K-means segmentation failed: {str(e)}")


def segment_by_watershed(
    structure: Structure,
    num_segments: Tuple[int, int],
    compactness: float = 0.0,
    **kwargs,
) -> List[Segment]:
    """
    Segment a structure using the watershed algorithm.

    Args:
        structure: Structure to segment
        num_segments: Rough number of segments as (rows, cols)
        compactness: Compactness parameter for watershed
        **kwargs: Additional parameters for watershed

    Returns:
        List of Segment objects

    Raises:
        SegmentationError: If segmentation fails
    """
    try:
        # Import watershed
        try:
            from skimage.segmentation import watershed
            from scipy.ndimage import distance_transform_edt
        except ImportError:
            raise SegmentationError("Watershed segmentation requires scikit-image")

        n_rows, n_cols = num_segments
        n_markers = n_rows * n_cols

        # Get structure mask
        structure_mask = ~structure.data.mask

        if not np.any(structure_mask):
            return []

        # Calculate distance transform
        distance = distance_transform_edt(structure_mask)

        # Create markers
        markers = np.zeros(structure.data.shape, dtype=np.int32)

        # Get structure bounds
        y_indices, x_indices = np.where(structure_mask)
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        # Place markers in a grid
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
        labels = watershed(
            -distance, markers, mask=structure_mask, compactness=compactness, **kwargs
        )

        # Create segments
        segments = []

        for i in range(1, marker_id):
            # Create mask for this segment
            segment_mask = labels == i

            if not np.any(segment_mask):
                continue

            # Create segment
            segment_data = ma.masked_array(structure.data.data, mask=~segment_mask)

            segment = Segment(
                data=segment_data, pixel_size=structure.pixel_size, id=i - 1
            )

            segments.append(segment)

        return segments

    except Exception as e:
        if isinstance(e, (SegmentationError, ValidationError)):
            raise
        raise SegmentationError(f"Watershed segmentation failed: {str(e)}")


def create_labeled_mask(structure: Structure) -> np.ndarray:
    """
    Create a labeled mask array from structure segments.

    Args:
        structure: Structure with segments

    Returns:
        Integer array where each value is the segment ID (-1 for background)

    Raises:
        SegmentationError: If creation fails
    """
    try:
        # Check if structure has segments
        if not structure.segments:
            raise SegmentationError("Structure has no segments")

        # Create output array
        labeled_mask = np.full(structure.data.shape, -1, dtype=np.int32)

        # Fill in segment IDs
        for segment in structure.segments:
            segment_mask = ~segment.data.mask
            labeled_mask[segment_mask] = segment.id

        return labeled_mask

    except Exception as e:
        if isinstance(e, SegmentationError):
            raise
        raise SegmentationError(f"Failed to create labeled mask: {str(e)}")
