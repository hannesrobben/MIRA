"""
Structure representation module.

This module provides classes for representing structures detected in images,
including the main Structure class and its components (Segment, Background).
"""

from __future__ import annotations

import json
import numpy as np
import numpy.ma as ma
import cv2
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union, Any


def _erode_mask_by_pixels(mask: np.ndarray, offset_pixels: int) -> np.ndarray:
    """
    Erode a boolean mask inward by a fixed number of pixels.

    Args:
        mask: Boolean mask, True where data is valid.
        offset_pixels: Number of pixels to erode from every edge of the
            valid region. 0 (or negative) returns the mask unchanged.

    Returns:
        Eroded boolean mask (same shape as input).
    """
    if offset_pixels <= 0:
        return mask.copy()
    from scipy.ndimage import binary_erosion

    return binary_erosion(mask, iterations=offset_pixels)


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

    def erode_by_pixels(self, offset_pixels: int = 0) -> ma.MaskedArray:
        """
        Extract background data with a fixed-pixel margin excluded from
        its edges.

        Args:
            offset_pixels: Number of pixels to erode from the background's
                edges. 0 (default) returns the background data unchanged.

        Returns:
            Masked array with only the eroded (inner) background region
            unmasked.
        """
        valid_mask = ~self.data.mask
        eroded_mask = _erode_mask_by_pixels(valid_mask, offset_pixels)
        return ma.masked_array(self.data.data, mask=~eroded_mask)


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
            inner_mask[min(rows) : min(rows) + row_offset, :] = False
            inner_mask[max(rows) - row_offset : max(rows) + 1, :] = False

        if col_offset > 0:
            inner_mask[:, min(cols) : min(cols) + col_offset] = False
            inner_mask[:, max(cols) - col_offset : max(cols) + 1] = False

        # Create result masked array
        result = ma.masked_array(self.data.data, mask=~inner_mask)
        return result

    def erode_by_pixels(self, offset_pixels: int = 0) -> ma.MaskedArray:
        """
        Extract segment data with a fixed-pixel margin excluded from the
        edges.

        Use this instead of `extract_with_offset` when segmentation
        boundaries may be off by a few pixels and you want to exclude
        exactly that many pixels near every segment edge, regardless of
        the segment's overall size.

        Args:
            offset_pixels: Number of pixels to erode from the segment's
                edges. 0 (default) returns the segment data unchanged.

        Returns:
            Masked array with only the eroded (inner) region unmasked.
        """
        valid_mask = ~self.data.mask
        eroded_mask = _erode_mask_by_pixels(valid_mask, offset_pixels)
        return ma.masked_array(self.data.data, mask=~eroded_mask)


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
    info: Dict[str, "Any"] = field(default_factory=dict)
    statistics: Dict[str, "Any"] = field(default_factory=dict)

    # File format version for save_compressed()/load_compressed() and for
    # the structure portion of StructureProcessor.save_snapshot(). Bump
    # this whenever the on-disk payload layout changes in a way that older
    # readers could not handle, so a version mismatch fails with a clear
    # error instead of a cryptic KeyError.
    FORMAT_VERSION = 1

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
    def from_mask(
        cls,
        data: np.ndarray,
        mask: np.ndarray,
        pixel_size: Tuple[float, float] = (1.0, 1.0),
    ) -> Structure:
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
    def alignment_transform(self) -> np.ndarray:
        """
        Compute a 2×3 affine matrix (OpenCV convention) that:
        1. Rotates the structure to be axis-aligned
        2. Translates its center to the image center

        The matrix is directly usable with cv2.warpAffine.

        Returns:
            M (np.ndarray): shape (2, 3), dtype float64

        Notes:
            Rotation angle is averaged from the top and bottom edges,
            using arctan2 to avoid division-by-zero and quadrant ambiguity.
        """
        if len(self.corner_points) == 4:
            tl, tr, br, bl = self.corner_points  # all (row, col)
        else:
            # approxPolyDP did not yield a clean quad (common for gratings):
            # fall back to the axis-aligned bounding box (tilt = 0).
            ys, xs = np.where(~self.data.mask)
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            tl, tr, br, bl = (y0, x0), (y0, x1), (y1, x1), (y1, x0)

        # Average angle from top and bottom edges (arctan2: no div-by-zero)
        phi_top = np.arctan2(tr[0] - tl[0], tr[1] - tl[1])
        phi_bot = np.arctan2(br[0] - bl[0], br[1] - bl[1])
        angle_deg = np.degrees((phi_top + phi_bot) / 2)

        # cv2 rotates around a (x, y) = (col, row) center point
        cx, cy = float(self.center[1]), float(self.center[0])  # col, row

        # Step 1: rotate around structure center to de-rotate
        M_rot = cv2.getRotationMatrix2D(center=(cx, cy), angle=angle_deg, scale=1.0)

        # Step 2: add translation to shift structure center → image center
        H, W = self.data.shape
        img_cx, img_cy = W / 2, H / 2
        M_rot[0, 2] += img_cx - cx  # Δx appended to existing translation column
        M_rot[1, 2] += img_cy - cy  # Δy

        return M_rot

    @property
    def corner_points(self) -> np.ndarray:
        """
        Get the corner points of the structure. (top-left, top-right, bottom-right, bottom-left)

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
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

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
            corners = np.array(
                [(min_y, min_x), (min_y, max_x), (max_y, max_x), (max_y, min_x)]
            )

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
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return 0.0

            # Calculate perimeter
            pixel_perimeter = cv2.arcLength(contours[0], True)

            # Convert to physical units (approximation)
            physical_perimeter = (
                pixel_perimeter * (self.pixel_size[0] + self.pixel_size[1]) / 2
            )

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
            physical_perimeter = (
                perimeter_pixels * (self.pixel_size[0] + self.pixel_size[1]) / 2
            )

            return physical_perimeter

    def refresh_info(self) -> Dict[str, "Any"]:
        """
        Recompute geometry-derived facts and merge them into `self.info`.

        Only the auto-computed keys below are overwritten; any custom keys
        already present (e.g. identifiers set via `set_info`, such as
        "doe_id" or "print_id") are left untouched. Call this again after
        segmenting or resizing the structure to keep the dimensions current.

        Returns:
            The updated `self.info` dict.
        """
        y_slice, x_slice = self.bounds
        height_px = int(y_slice.stop - y_slice.start)
        width_px = int(x_slice.stop - x_slice.start)
        dy, dx = self.pixel_size

        # Cast explicitly to native Python types (not np.int64/np.float64,
        # which self.bounds/self.pixel_size/self.area may hand back) so
        # `self.info` stays directly json.dump-able without a separate
        # to_json_safe() pass, unlike `self.statistics` which holds
        # arbitrary result objects and needs that conversion regardless.
        self.info.update(
            {
                "shape_px": tuple(int(s) for s in self.data.shape),
                "bounding_box_px": (height_px, width_px),
                "physical_width": float(width_px * dx),
                "physical_height": float(height_px * dy),
                "pixel_size": tuple(float(p) for p in self.pixel_size),
                "area": float(self.area),
                "n_segments": len(self.segments),
            }
        )
        return self.info

    def set_info(self, **kwargs) -> None:
        """
        Add or update custom identification metadata (e.g. `doe_id`,
        `print_id`, `structure_type`) in `self.info`, alongside the
        auto-computed geometry keys from `refresh_info`.

        This is the metadata `analysis.ensemble.build_ensemble_dataframe`
        reads its identifying columns from - set here once per structure,
        used everywhere downstream.

        Args:
            **kwargs: Arbitrary key/value pairs to merge into `self.info`.
        """
        self.info.update(kwargs)

    def add_statistic(self, name: str, result: "Any") -> None:
        """
        Store an analysis result under `name` in `self.statistics`.

        Args:
            name: Key to store the result under (e.g. "delta_n_segment_mean",
                "regression_vs_sensofar"). Overwrites any existing entry
                with the same name.
            result: The result to store - typically a dataclass such as
                `DeltaNResult`, `RegressionResult`, or `BlandAltmanResult`,
                but any value is accepted.
        """
        self.statistics[name] = result

    def statistics_summary(self) -> Dict[str, "Any"]:
        """
        JSON-serializable snapshot of everything stored in `self.statistics`.

        Converts dataclass results (and any numpy values inside them) into
        plain Python types via `mira.utils.serialization.to_json_safe`, so
        the returned dict can be passed straight to `json.dump` - this is
        the direct replacement for manually building an `info_dictionary`
        dict before dumping it to `information_dictionary.json`.

        Returns:
            Dict mapping each statistic's name to its JSON-safe value.
        """
        from ..utils.serialization import to_json_safe

        return {name: to_json_safe(result) for name, result in self.statistics.items()}

    def _segments_payload(self) -> Dict[str, np.ndarray]:
        """
        Build the segment part of a serialized payload.

        Returns a stacked boolean mask array (one layer per segment, True =
        segment pixel) plus the matching segment ids, as separate arrays -
        npz has no native concept of "a list of masked arrays", so a list of
        N same-shape 2D masks is stacked into one 3D array instead of saved
        as N separate keys, which would need a variable, run-dependent set
        of key names. Returns empty (zero-length) arrays when there are no
        segments, so callers never need a separate no-segments branch.

        Returns:
            Dict with keys "segment_ids" (shape (N,), int32) and
            "segment_masks" (shape (N, *data.shape), bool).
        """
        if not self.segments:
            return {
                "segment_ids": np.empty((0,), dtype=np.int32),
                "segment_masks": np.empty((0,) + self.data.shape, dtype=bool),
            }
        ids = np.array([seg.id for seg in self.segments], dtype=np.int32)
        masks = np.stack([~seg.data.mask for seg in self.segments], axis=0)
        return {"segment_ids": ids, "segment_masks": masks}

    def _mask_payload(self) -> Dict[str, np.ndarray]:
        """
        Build the geometry-only part of a serialized payload.

        Contains the structure mask, pixel size, segment masks/ids, and the
        `info`/`statistics` dicts - deliberately WITHOUT the underlying data
        array. This is the single source of truth for what "structure
        geometry + metadata" means on disk.

        `info` and `statistics` are stored as JSON strings (0-d numpy
        arrays), matching the pattern already used for
        `StructureProcessor`'s `metadata_json` - npz has no native dict
        type and this repo avoids `allow_pickle=True`. `statistics` is run
        through `to_json_safe` first, since its values are typically result
        dataclasses (DeltaNResult, RegressionResult, ...) rather than
        already-plain types.

        Returns:
            Dict of numpy arrays, ready to be merged into a larger
            ``np.savez_compressed`` payload (see ``save_compressed`` and
            ``StructureProcessor.save_snapshot``).
        """
        from ..utils.serialization import to_json_safe

        payload = {
            "structure_mask": ~self.data.mask,
            "structure_pixel_size": np.array(self.pixel_size, dtype=np.float64),
        }
        payload.update(self._segments_payload())
        # `self.info` is normally already JSON-safe (refresh_info() casts
        # to native types), but `to_json_safe` is applied here too as a
        # cheap safety net against a numpy value slipping in via a custom
        # set_info(...) call.
        payload["info_json"] = np.array(json.dumps(to_json_safe(self.info)))
        payload["statistics_json"] = np.array(json.dumps(to_json_safe(self.statistics)))
        return payload

    @classmethod
    def _from_mask_payload(
        cls, payload: Dict[str, np.ndarray], raw_data: np.ndarray
    ) -> "Structure":
        """
        Reconstruct a Structure from a ``_mask_payload()``-shaped dict and an
        externally supplied raw data array.

        This is the inverse of ``_mask_payload``, used whenever the raw data
        lives elsewhere - e.g. a raw channel array a
        ``StructureProcessor`` snapshot just loaded independently. The mask
        defines the geometry, ``raw_data`` supplies the values, and
        ``Structure.from_mask`` combines the two, exactly as it does for a
        freshly detected structure.

        Args:
            payload: Dict as produced by ``_mask_payload`` (or an
                ``np.load`` result containing the same keys) - must contain
                "structure_mask" and "structure_pixel_size"; "segment_ids"
                and "segment_masks" are optional.
            raw_data: Raw channel array to combine with the stored mask.
                Must have the same shape as the stored mask.

        Returns:
            Reconstructed Structure, with segments restored if present.

        Raises:
            StructureError: If ``raw_data``'s shape does not match the
                stored mask's shape.
        """
        mask = np.asarray(payload["structure_mask"])
        if raw_data.shape != mask.shape:
            raise StructureError(
                f"_from_mask_payload: raw_data shape {raw_data.shape} does "
                f"not match the stored structure mask shape {mask.shape}."
            )

        pixel_size = tuple(np.asarray(payload["structure_pixel_size"]).tolist())
        structure = cls.from_mask(raw_data, mask, pixel_size=pixel_size)

        seg_ids = payload.get("segment_ids", np.empty((0,), dtype=np.int32))
        seg_masks = payload.get(
            "segment_masks", np.empty((0,) + mask.shape, dtype=bool)
        )
        segments = []
        for seg_id, seg_mask in zip(seg_ids, seg_masks):
            segments.append(
                Segment(
                    data=ma.masked_array(raw_data, mask=~seg_mask),
                    pixel_size=pixel_size,
                    id=int(seg_id),
                )
            )
        structure.segments = segments

        info_json = payload.get("info_json")
        if info_json is not None:
            structure.info = json.loads(str(info_json))

        statistics_json = payload.get("statistics_json")
        if statistics_json is not None:
            structure.statistics = json.loads(str(statistics_json))

        return structure

    def save_compressed(self, file_path: Union[str, Path]) -> Path:
        """
        Save this structure as a single, self-contained compressed .npz file.

        Includes the raw data array, so the file can be loaded back into a
        fully usable Structure entirely on its own, independent of any
        StructureProcessor - the right choice for sharing, archiving, or
        inspecting one structure in isolation. If the matching raw channel
        is already being stored separately (e.g. as part of a full
        StructureProcessor snapshot), use ``_mask_payload`` instead so the
        same array is not written to disk twice - see
        ``StructureProcessor.save_snapshot``.

        The write is atomic: the payload is written to a temporary file
        first and only renamed to ``file_path`` on success, so a crash
        mid-write can never leave a corrupt file at the target path.

        Args:
            file_path: Destination path. A ".npz" suffix is appended if
                missing. Parent directories are created as needed.

        Returns:
            The resolved Path the file was written to.
        """
        file_path = Path(file_path)
        if file_path.suffix != ".npz":
            file_path = file_path.with_suffix(".npz")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        payload = self._mask_payload()
        payload["data"] = self.data.data
        payload["format_version"] = np.array(self.FORMAT_VERSION)

        # np.savez_compressed silently appends ".npz" to the filename unless
        # it already ends with ".npz" - so the temp name must end in ".npz"
        # too, or numpy would write "<tmp_path>.npz" instead of tmp_path,
        # and the rename below would fail to find its source file.
        tmp_path = file_path.with_name(file_path.stem + ".tmp.npz")
        np.savez_compressed(tmp_path, **payload)
        tmp_path.replace(file_path)

        return file_path

    @classmethod
    def load_compressed(cls, file_path: Union[str, Path]) -> "Structure":
        """
        Load a Structure previously written by ``save_compressed``.

        Args:
            file_path: Path to the .npz file.

        Returns:
            Reconstructed Structure, including segments if any were saved.

        Raises:
            StructureError: If the file is missing required keys or was
                written by an incompatible format version.
        """
        file_path = Path(file_path)
        with np.load(file_path, allow_pickle=False) as npz:
            if "format_version" not in npz.files or "data" not in npz.files:
                raise StructureError(
                    f"{file_path}: not a valid Structure export "
                    "(missing 'format_version' or 'data')."
                )
            version = int(npz["format_version"])
            if version != cls.FORMAT_VERSION:
                raise StructureError(
                    f"{file_path}: unsupported Structure file format "
                    f"version {version} (expected {cls.FORMAT_VERSION})."
                )
            payload = {key: npz[key] for key in npz.files}

        return cls._from_mask_payload(payload, payload["data"])

    def realign(self) -> Structure:
        """
        Return a new Structure with the data realigned:
        - structure center translated to image center
        - structure de-rotated to be axis-aligned

        The mask is transformed consistently with the data.

        Returns:
            New Structure instance with realigned data and updated mask.
        """
        import cv2

        M = self.alignment_transform
        H, W = self.data.shape

        # Transform data (phase/height values — use linear interpolation)
        aligned_data = cv2.warpAffine(
            self.data.data.astype(np.float64),
            M,
            (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

        # Transform mask with nearest-neighbour to keep it binary
        mask_uint8 = (~self.data.mask).astype(np.uint8) * 255
        aligned_mask_uint8 = cv2.warpAffine(
            mask_uint8,
            M,
            (W, H),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        aligned_mask = aligned_mask_uint8 > 127  # back to bool

        aligned_ma = ma.masked_array(aligned_data, mask=~aligned_mask)
        return Structure(data=aligned_ma, pixel_size=self.pixel_size)

    def segment(
        self, num_segments: Tuple[int, int] = (1, 1), method: str = "grid"
    ) -> List[Segment]:
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
        if method.lower() == "grid":
            segments = self._segment_grid(num_segments)
        elif method.lower() == "kmeans":
            segments = self._segment_kmeans(num_segments)
        elif method.lower() == "watershed":
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
            corners = np.array(
                [(min_y, min_x), (min_y, max_x), (max_y, max_x), (max_y, min_x)]
            )

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
                segment_mask[
                    top_left_y : bottom_right_y + 1, top_left_x : bottom_right_x + 1
                ] = True

                # Intersect with structure mask
                segment_mask = segment_mask & structure_mask

                # If segment contains structure pixels
                if np.any(segment_mask):
                    # Update labeled array
                    labeled_array[segment_mask] = segment_id

                    # Create segment
                    segment_data = ma.masked_array(self.data.data, mask=~segment_mask)
                    segment = Segment(
                        data=segment_data, pixel_size=self.pixel_size, id=segment_id
                    )
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
            segment_indices = labels == i
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
            segment_mask = labels == i

            if np.any(segment_mask):
                segment_data = ma.masked_array(self.data.data, mask=~segment_mask)
                segment = Segment(
                    data=segment_data, pixel_size=self.pixel_size, id=i - 1
                )
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

    def extract_profile(
        self, start_point: Tuple[int, int], end_point: Tuple[int, int]
    ) -> np.ndarray:
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
            mode="constant",
            cval=np.nan,
        )

        return profile
