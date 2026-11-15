"""
Main StructureProcessor class for the image processing library.

This module provides the primary entry point for the library, offering a
unified interface for all operations on image data, structures, and segments.
"""

from __future__ import annotations

from dataclasses import field
from sqlite3 import DataError

import cv2
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from scipy.ndimage import shift as nd_shift
from pathlib import Path
from typing import Union, Optional, Tuple, List, Dict, Literal
import logging

from .core.image import ImageData
from .core.structure import Structure, Background, Segment
from .visualization.visualization_debug import plot_structure_properties

import matplotlib.patches as patches

from .processing.detection import (
    detect_structure,
)

from .processing.segmentation import (
    segment_structure,
)

from .processing.alignment import (
    align_structures,
    align_processor_pair,
    align_processors_many,
    sync_raw_data_canvas,
)

from .processing.phase_retrieval import (
    convert_phase_to_height,
    convert_height_to_phase,
    unwrap_phase,
    full_phase_retrieval,
    adjust_background,
)

from .utils.helpers import get_logger
from .utils.exceptions import (
    DataLoadError,
    ValidationError,
    ProcessingError,
    SegmentationError,
    AlignmentError,
    VisualizationError,
)


class StructureProcessor:
    """
    Main class for processing and analyzing image data.

    This class provides a unified interface for all operations on image data,
    structures, and segments. It handles loading, processing, analysis, and
    visualization of various types of image data.

    Attributes:
        phase_data (Optional[ImageData]): Phase image data, if available
        intensity_data (Optional[ImageData]): Intensity image data, if available
        structure (Optional[Structure]): Detected structure, if available
        logger (logging.Logger): Logger for messages and debugging
    """

    def __init__(
        self,
        phase_data: Optional[Union[ImageData, np.ndarray, Path, str]] = None,
        height_data: Optional[Union[ImageData, np.ndarray, Path, str]] = None,
        intensity_data: Optional[Union[ImageData, np.ndarray, Path, str]] = None,
        pixel_size: Optional[Tuple[float, float]] = None,
        n_resin: float = 1.52,
        wavelength: float = 0.6749e-6,
        log_level: int = logging.INFO,
        adjusting_background: bool = field(default=True),
        data_upsidedown: bool = field(default=False),
        description: [str, dict] = field(default=None),
        segmentation_method: Literal["manual", "automatic"] = field(default="manual"),
        structure_detection_method: Literal["threshold", "intensity"] = field(
            default="threshold"
        ),
    ):
        """
        Initialize StructureProcessor with optional image data.

        Args:
            phase_data: Phase image data or path to file
            intensity_data: Intensity image data or path to file
            pixel_size: Pixel size in physical units (width, height)
            n_resin: Refractive index of the resin
            wavelength: Light wavelength in meters
            log_level: Logging level for the logger
        """
        # Initialize logger
        # ToDo - Rework logger, because of issues?! - testing required
        self.logger = get_logger("StructureProcessor", level=log_level)
        self.logger.info("Initializing StructureProcessor")

        # Store optical parameters
        self.n_resin = n_resin
        self.delta_n = None  # Difference in refractive index - for easier use in the future (printed data for fitting)
        self.wavelength = wavelength

        # Initialize image data attributes
        # Todo - include height data attribute (for use with offline measurement methods like LSM, CM)
        #   include height data and change the methods (phase -> height and height -> phase) accordingly
        #   check the interconnectivity between data (phase, intensity, height) and the structure class
        self.phase_data = None
        self.height_data = None
        self.intensity_data = None
        self.structure = None
        self._segments = []

        self.data_upsidedown = data_upsidedown
        self.adjusting_background = adjusting_background
        self.description = description

        # Load data if provided
        if phase_data is not None:
            self.load_phase_data(
                phase_data, pixel_size, data_upsidedown=self.data_upsidedown
            )
            if intensity_data is not None:
                self.load_intensity_data(
                    intensity_data, pixel_size, data_upsidedown=self.data_upsidedown
                )
        elif height_data is not None:
            self.load_height_data(
                height_data, pixel_size, data_upsidedown=self.data_upsidedown
            )  # todo make function
        else:
            raise DataLoadError("No data given. Provide phase or height data.")

        self.segmentation_method = segmentation_method
        self.structure_detection_method = structure_detection_method

    @property
    def _active_data(self) -> ImageData:
        """
        Return the currently active raw channel (phase or height) as ImageData.

        This is the single choke point every other method uses to reach the
        raw data: ``shape``, ``pixel_size``, ``data``, ``apply_affine_transform``,
        ``adjust_background``, ``_zero_background_inplace``,
        ``_resync_structure_to_raw``, ``sync_raw_data_canvas``, and more. It must
        never return anything other than the ImageData wrapping ``phase_data``
        or ``height_data`` - in particular NOT ``structure.data`` (a masked
        array derived from the raw data, not the raw data itself). Returning
        the wrong type here breaks every one of those callers at once, so the
        contract is enforced explicitly below instead of trusted implicitly.

        Returns:
            The active channel (``phase_data`` if present, else ``height_data``).

        Raises:
            DataLoadError: If neither phase nor height data has been loaded.
            ValidationError: If the active channel is not an ``ImageData``
                instance. This happens when ``phase_data`` / ``height_data`` was
                assigned a raw value directly (e.g.
                ``processor.height_data = some_ndarray``) instead of going
                through ``load_phase_data()`` / ``load_height_data()`` /
                ``update_data()``, which wrap the array in ``ImageData`` and
                keep ``pixel_size`` and other metadata attached.
        """
        # todo future: insert flag or something similar to know what active data is
        if self.phase_data is None and self.height_data is None:
            raise DataLoadError("No data given. Provide phase or height data.")
        if self.phase_data is None and self.height_data is not None:
            active, active_name = self.height_data, "height_data"
        elif self.height_data is None and self.phase_data is not None:
            active, active_name = self.phase_data, "phase_data"
        else:  # height and phase data is given - for now phase data is
            active, active_name = self.phase_data, "phase_data"

        if not isinstance(active, ImageData):
            raise ValidationError(
                f"StructureProcessor.{active_name} must be an ImageData "
                f"instance, got {type(active).__name__} instead. This usually "
                f"means '{active_name}' was assigned a raw value directly "
                f"(e.g. processor.{active_name} = some_array) instead of going "
                "through load_phase_data() / load_height_data() / "
                "update_data(), which wrap the array in ImageData and preserve "
                f"pixel_size. Fix it with: processor.{active_name} = "
                f"ImageData(data=some_array, pixel_size=processor.pixel_size)."
            )
        return active

    @property
    def shape(self):
        return self._active_data.shape

    @property
    def pixel_size(self):
        return self._active_data.pixel_size

    @property
    def data(self):
        return self._active_data.data

    def debug(self):
        import matplotlib.pyplot as plt

        self.detect_structure(threshold_method="percentile", percentile=20)
        plt.imshow(self.structure.mask)
        plt.show()

        test = True
        if test:
            data = self.structure.data.data
            bou = self.structure.bounds
            corners = self.structure.corner_points
            cen = self.structure.center
            roi = data[bou]

            tl, tr, br, bl = corners[0], corners[1], corners[2], corners[3]

            def slope_and_angle(p_left, p_right):
                drow = p_right[0] - p_left[0]
                dcol = p_right[1] - p_left[1]
                m = drow / dcol if dcol != 0 else np.inf
                phi = np.arctan(m)
                return m, phi

            m_top, phi_top = slope_and_angle(tl, tr)
            m_bot, phi_bot = slope_and_angle(bl, br)

            print(f"Top   — slope: {m_top:.4f}, angle: {np.degrees(phi_top):.2f}°")
            print(f"Bottom— slope: {m_bot:.4f}, angle: {np.degrees(phi_bot):.2f}°")

            row_slice, col_slice = bou
            row_off, col_off = row_slice.start, col_slice.start

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Full image
            ax = axes[0]
            ax.imshow(data, cmap="gray", origin="upper")
            for corner in corners:
                ax.plot(corner[1], corner[0], "r+", markersize=8)
            ax.plot(cen[1], cen[0], "bo", markersize=6)
            rect = patches.Rectangle(
                (col_slice.start, row_slice.start),
                col_slice.stop - col_slice.start,
                row_slice.stop - row_slice.start,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.set_title("Full image with ROI")

            # ROI with offset-corrected markers
            ax2 = axes[1]
            ax2.imshow(roi, cmap="gray", origin="upper")
            for corner in corners:
                ax2.plot(corner[1] - col_off, corner[0] - row_off, "r+", markersize=8)
            ax2.plot(cen[1] - col_off, cen[0] - row_off, "bo", markersize=6)
            ax2.set_title(f"ROI  {roi.shape[0]}×{roi.shape[1]} px")

            plt.tight_layout()
            plt.show()

        # self.structure.segment(num_segments=(1, 8), method="kmeans")
        # print(self.structure.segments)
        print("stop")

    def load_phase_data(
        self,
        data: Union[ImageData, np.ndarray, Path, str],
        pixel_size: Optional[Tuple[float, float]] = None,
        data_upsidedown=False,
    ) -> "StructureProcessor":
        """
        Load phase image data.

        Args:
            data: Phase image data or path to file
            pixel_size: Pixel size in physical units (optional if already in data)

        Returns:
            Self for method chaining

        Raises:
            DataLoadError: If data loading fails
        """
        try:
            # Handle different input types
            if isinstance(data, ImageData):
                self.phase_data = data
                if data_upsidedown:
                    self.phase_data.mirror_horizontal()

            elif isinstance(data, np.ndarray):
                if pixel_size is None:
                    raise ValidationError(
                        "Pixel size must be provided for numpy array input"
                    )
                self.phase_data = ImageData(
                    data, pixel_size, data_upsidedown=data_upsidedown
                )

            elif isinstance(data, (str, Path)):
                # Load from file
                try:
                    # todo check loading from file - because pixel size to be 1.0|1.0 is sus
                    self.phase_data = ImageData.from_file(
                        data, pixel_size or (1.0, 1.0), data_upsidedown=data_upsidedown
                    )
                except Exception as e:
                    raise DataLoadError(
                        f"Failed to load phase data from file: {str(e)}"
                    )

            else:
                raise ValidationError(f"Unsupported data type: {type(data)}")

            self.logger.info(f"Loaded phase data with shape {self.phase_data.shape}")
            return self

        except Exception as e:
            if isinstance(e, (DataLoadError, ValidationError)):
                raise
            raise DataLoadError(f"Failed to load phase data: {str(e)}")

    def load_height_data(
        self,
        data: Union[ImageData, np.ndarray, Path, str],
        pixel_size: Optional[Tuple[float, float]] = None,
        data_upsidedown=False,
    ) -> "StructureProcessor":
        """
        Load height image data.

        Args:
            data: Height image data or path to file
            pixel_size: Pixel size in physical units (optional if already in data)

        Returns:
            Self for method chaining

        Raises:
            DataLoadError: If data loading fails
        """
        try:
            # Handle different input types
            if isinstance(data, ImageData):
                self.height_data = data
                if data_upsidedown:
                    self.height_data.mirror_horizontal()

            elif isinstance(data, np.ndarray):
                if pixel_size is None:
                    raise ValidationError(
                        "Pixel size must be provided for numpy array input"
                    )
                self.height_data = ImageData(
                    data, pixel_size, data_upsidedown=data_upsidedown
                )

            elif isinstance(data, (str, Path)):
                # Load from file
                try:
                    self.height_data = ImageData.from_file(
                        data, pixel_size or (1.0, 1.0), data_upsidedown=data_upsidedown
                    )
                except Exception as e:
                    raise DataLoadError(
                        f"Failed to load height data from file: {str(e)}"
                    )

            else:
                raise ValidationError(f"Unsupported data type: {type(data)}")

            self.logger.info(f"Loaded height data with shape {self.height_data.shape}")
            return self

        except Exception as e:
            if isinstance(e, (DataLoadError, ValidationError)):
                raise
            raise DataLoadError(f"Failed to load phase data: {str(e)}")

    def load_intensity_data(
        self,
        data: Union[ImageData, np.ndarray, Path, str],
        pixel_size: Optional[Tuple[float, float]] = None,
        data_upsidedown=False,
    ) -> "StructureProcessor":
        """
        Load intensity image data.

        Args:
            data: Intensity image data or path to file
            pixel_size: Pixel size in physical units (optional if already in data)

        Returns:
            Self for method chaining

        Raises:
            DataLoadError: If data loading fails
        """
        try:
            # Handle different input types
            if isinstance(data, ImageData):
                self.intensity_data = data
                if data_upsidedown:
                    self.intensity_data.mirror_horizontal()

            elif isinstance(data, np.ndarray):
                if pixel_size is None and self.phase_data is not None:
                    # Use pixel size from phase data if available
                    pixel_size = self.phase_data.pixel_size
                elif pixel_size is None:
                    raise ValidationError(
                        "Pixel size must be provided for numpy array input"
                    )

                self.intensity_data = ImageData(
                    data, pixel_size, data_upsidedown=data_upsidedown
                )

            elif isinstance(data, (str, Path)):
                # Load from file
                if pixel_size is None and self.phase_data is not None:
                    # Use pixel size from phase data if available
                    pixel_size = self.phase_data.pixel_size

                try:
                    self.intensity_data = ImageData.from_file(
                        data, pixel_size or (1.0, 1.0), data_upsidedown=data_upsidedown
                    )
                except Exception as e:
                    raise DataLoadError(
                        f"Failed to load intensity data from file: {str(e)}"
                    )

            else:
                raise ValidationError(f"Unsupported data type: {type(data)}")

            self.logger.info(
                f"Loaded intensity data with shape {self.intensity_data.shape}"
            )
            return self

        except Exception as e:
            if isinstance(e, (DataLoadError, ValidationError)):
                raise
            raise DataLoadError(f"Failed to load intensity data: {str(e)}")

    def update_data(self, height_data=None, phase_data=None, intensity_data=None):
        if phase_data is None and height_data is None and intensity_data is None:
            raise DataError("No data given!")

        if phase_data is not None:
            self.phase_data.update_data(data=phase_data)
        if height_data is not None:
            self.height_data.update_data(data=height_data)
        if intensity_data is not None:
            self.intensity_data.update_data(data=intensity_data)

    def get_mean_background_value(self, margin: int = 5) -> Optional[float]:
        if self.structure is None:
            return None
        bg_mask = self.structure.data.mask
        raw = self._active_data.data
        # Shape guard: prevents the original raw/mask shape-mismatch error.
        if raw.shape != bg_mask.shape:
            self.logger.warning(
                f"get_mean_background_value: shape mismatch "
                f"(raw={raw.shape}, mask={bg_mask.shape}); returning None."
            )
            return None
        expanded_mask = binary_dilation(~bg_mask, iterations=margin)
        bg_pixels = raw[~expanded_mask]
        return float(np.mean(bg_pixels)) if len(bg_pixels) > 0 else None

    def _mean_background_value(
        self, channel: ImageData, margin: int = 5
    ) -> Optional[float]:
        """
        Mean background value of a specific raw channel (not necessarily the
        active one), using the current structure mask. Same margin-erosion
        logic as get_mean_background_value(), but parameterized over the
        channel so callers (e.g. apply_affine_transform) can compute a
        per-channel fill value instead of relying on a single active channel.
        """
        if self.structure is None:
            return None
        bg_mask = self.structure.data.mask
        raw = channel.data
        if raw.shape != bg_mask.shape:
            self.logger.warning(
                f"_mean_background_value: shape mismatch "
                f"(raw={raw.shape}, mask={bg_mask.shape}); returning None."
            )
            return None
        expanded_mask = binary_dilation(~bg_mask, iterations=margin)
        bg_pixels = raw[~expanded_mask]
        return float(np.mean(bg_pixels)) if len(bg_pixels) > 0 else None

    def update_pixel_size(self, pixel_size: Tuple[float, float]) -> None:
        """Propagate a new pixel_size to all loaded ImageData objects."""
        for attr in ("phase_data", "height_data", "intensity_data"):
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.pixel_size = pixel_size

    def center_structure(self, matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Shift *matrix* so that the structure center lands at the image center.
        Falls back to _active_data.data when matrix is None.
        """
        if matrix is None:
            matrix = self._active_data.data
        center_point = np.array(self.structure.center)  # (row, col)
        image_center = np.array(matrix.shape) / 2
        shift_vector = image_center - center_point
        fill = self.get_mean_background_value() or 0.0
        return nd_shift(matrix, shift_vector, mode="constant", cval=fill)

    # ------------------------------------------------------------------ #
    # Geometry gateway                                                    #
    # ------------------------------------------------------------------ #
    # Single, authoritative entry point for any raster-geometry change.   #
    # Every operation that moves, rotates, scales or otherwise resamples  #
    # the content (alignment, realignment, future registration refiners)  #
    # must go through apply_affine_transform so that the raw channels and #
    # the structure/segment masks can never silently drift apart.         #
    # ------------------------------------------------------------------ #
    def apply_affine_transform(
        self,
        matrix: np.ndarray,
        output_shape: Optional[Tuple[int, int]] = None,
        *,
        data_interpolation: int = cv2.INTER_LINEAR,
        border_value: Optional[float] = None,
        validate: bool = True,
    ) -> "StructureProcessor":
        """
        Apply one affine transform to all raster representations at once.

        This is the single, authoritative entry point for any geometric
        (raster) change of the processor's content. It warps every raw
        channel (phase / height / intensity) together with the structure
        mask and every segment mask using the *same* matrix, then rebuilds
        the ``Structure`` (and its ``Segment`` objects) from the freshly
        warped active raw array. Because the raw data and the masks are
        produced by the same call, they cannot silently drift apart.

        Any operation that moves, rotates, scales or otherwise resamples the
        content must route through this method instead of transforming
        ``structure.data`` on its own. See ``realign_structure`` and
        ``processing.alignment._apply_warp_to_processor`` for callers.

        Args:
            matrix: Affine matrix, either 2x3 (OpenCV convention) or 3x3 (the
                last row is dropped). Maps input pixel coordinates to output
                pixel coordinates in (x, y) order, as expected by
                ``cv2.warpAffine``.
            output_shape: Target canvas shape as (height, width) in NumPy
                order. Defaults to the current active-data shape (a
                size-preserving transform).
            data_interpolation: OpenCV interpolation flag for the continuous
                raw channels (default ``cv2.INTER_LINEAR``). Masks always use
                nearest-neighbour interpolation so they stay strictly binary.
            border_value: Value used for output pixels mapped from outside the
                source image (default 0.0). Callers relying on a zero
                background should zero it beforehand (see
                ``_zero_background_inplace``).
            validate: If True (default), run
                ``_assert_data_structure_consistency`` after the transform and
                raise on any shape mismatch.

        Returns:
            Self for method chaining.

        Raises:
            ValidationError: If no raw data is available to transform, or the
                matrix has an unsupported shape.
            ProcessingError: If the post-transform consistency check fails.
        """
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape == (3, 3):
            matrix = matrix[:2, :]
        elif matrix.shape != (2, 3):
            raise ValidationError(
                "apply_affine_transform: matrix must be 2x3 or 3x3, "
                f"got shape {matrix.shape}."
            )

        if self.phase_data is None and self.height_data is None:
            raise ValidationError(
                "apply_affine_transform: no raw data available to transform."
            )

        if output_shape is None:
            output_shape = self._active_data.data.shape
        out_h, out_w = int(output_shape[0]), int(output_shape[1])
        cv_size = (out_w, out_h)  # cv2.warpAffine expects (width, height)

        """
        # 1) Warp every available raw channel with identical geometry.
        for attr in ("phase_data", "height_data", "intensity_data"):
            img: Optional[ImageData] = getattr(self, attr, None)
            if img is not None:
                warped = cv2.warpAffine(
                    img.data.astype(np.float64),
                    matrix,
                    cv_size,
                    flags=data_interpolation,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=border_value,
                )
                img.update_data(data=warped)
        """
        # 1) Warp every available raw channel with identical geometry.
        #    Each channel (phase/height/intensity) is a different physical
        #    quantity, so the border fill is computed per channel from its own
        #    background mean - never a shared/global value.
        for attr in ("phase_data", "height_data", "intensity_data"):
            img: Optional[ImageData] = getattr(self, attr, None)
            if img is not None:
                if border_value is None:
                    fill_value = self._mean_background_value(img)
                    if fill_value is None:
                        fill_value = 0.0  # no structure/mask yet - safe fallback
                else:
                    fill_value = border_value

                warped = cv2.warpAffine(
                    img.data.astype(np.float64),
                    matrix,
                    cv_size,
                    flags=data_interpolation,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=fill_value,
                )
                img.update_data(data=warped)

        # If no structure has been detected yet there is nothing to re-sync.
        if self.structure is None:
            if validate:
                self._assert_data_structure_consistency()
            return self

        new_raw = self._active_data.data

        def _warp_mask(bool_mask: np.ndarray) -> np.ndarray:
            """Warp a boolean mask, keeping it strictly binary."""
            warped = cv2.warpAffine(
                bool_mask.astype(np.float32),
                matrix,
                cv_size,
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            return warped > 0.5

        # 2) Warp each segment mask (same matrix), preserving id / pixel_size.
        new_segments: List[Segment] = []
        for seg in self.structure.segments:
            seg_present = _warp_mask(~seg.data.mask)
            new_segments.append(
                Segment(
                    data=ma.masked_array(new_raw, mask=~seg_present),
                    pixel_size=seg.pixel_size,
                    id=seg.id,
                )
            )

        # 3) Warp the structure mask and rebuild it against the new raw data.
        structure_present = _warp_mask(~self.structure.data.mask)
        self.structure = Structure(
            data=ma.masked_array(new_raw, mask=~structure_present),
            pixel_size=self.structure.pixel_size,
        )
        self.structure.segments = new_segments

        if validate:
            self._assert_data_structure_consistency()
        return self

    def _assert_data_structure_consistency(self) -> None:
        """
        Verify that raw data, structure mask and segment masks are shape-aligned.

        The library keeps two representations of the same field: the raw
        channels (``phase_data`` / ``height_data`` / ``intensity_data``) and
        the masked ``structure`` (plus its ``segments``). Every geometric
        operation must keep them in lock-step. This guard turns a silent
        divergence - the class of bug where a plot or measurement quietly uses
        mismatched geometry - into an explicit, immediate error.

        Raises:
            ProcessingError: If any raw channel, the structure mask, or a
                segment mask has a shape different from the active raw data.
        """
        if self.phase_data is None and self.height_data is None:
            return  # nothing loaded yet

        active_shape = self._active_data.data.shape

        for attr in ("phase_data", "height_data", "intensity_data"):
            img = getattr(self, attr, None)
            if img is not None and img.data.shape != active_shape:
                raise ProcessingError(
                    f"_assert_data_structure_consistency: {attr} shape "
                    f"{img.data.shape} != active data shape {active_shape}."
                )

        if self.structure is None:
            return

        if self.structure.data.shape != active_shape:
            raise ProcessingError(
                "_assert_data_structure_consistency: structure data shape "
                f"{self.structure.data.shape} != active data shape {active_shape}."
            )

        for seg in self.structure.segments:
            if seg.data.shape != active_shape:
                raise ProcessingError(
                    f"_assert_data_structure_consistency: segment id={seg.id} "
                    f"shape {seg.data.shape} != active data shape {active_shape}."
                )

    def detect_structure(
        self,
        method: Literal["threshold", "intensity", "adaptive"] = "threshold",
        **kwargs,
    ) -> "StructureProcessor":
        """
        Detect structure in phase data.

        Args:
            method: Detection method to use
            **kwargs: Additional parameters for the detection method

        Returns:
            Self for method chaining

        Raises:
            ProcessingError: If detection fails
            ValidationError: If required data is missing
        """
        try:
            # todo- structure detection would always be with intensity image - if available
            #   does the method has to be change if intensity_data is none?
            if self.phase_data is None and self.height_data is None:
                raise ValidationError("Data must be loaded before detecting structure!")

            # Call detection function
            intensity = self.intensity_data if method == "intensity" else None
            data = None
            # todo: in future recheck for changes in regard to a boolean variable switching from phase to height and
            #   vice versa

            answer = None
            if self.phase_data is None:
                if self.height_data is not None:
                    data = self.height_data
                    answer = "h"
            elif self.height_data is None:
                if self.phase_data is not None:
                    data = self.phase_data
                    answer = "p"
            elif self.phase_data is not None and self.height_data is not None:
                while answer != "p" and answer != "h":
                    answer = input(
                        "Which data should be used for structure detection? \n\t p -> phase data \n\t h -> height data"
                    )
                if answer == "p":
                    data = self.phase_data
                elif answer == "h":
                    data = self.height_data

            # todo check how method is being handled
            self.structure = detect_structure(
                data, method=method, intensity_image=intensity, **kwargs
            )

            new_data = self.structure.data.data
            # print(f"data shape: {data.shape}")
            # print(f"data shape: {new_data.shape}")
            if answer == "h":
                self.update_data(height_data=new_data)
            elif answer == "p":
                self.update_data(phase_data=new_data)
            else:
                raise Warning("No data changed!")

            self.logger.info(f"Detected structure using {method} method")
            return self

        except Exception as e:
            if isinstance(e, (ProcessingError, ValidationError)):
                raise
            raise ProcessingError(f"Structure detection failed: {str(e)}")

    def realign_structure(self) -> "StructureProcessor":
        """
        De-rotate and center the structure, transforming the raw data and the
        structure/segment masks together so they stay consistent.

        The affine transform is taken from ``structure.alignment_transform``
        (rotate the structure to be axis-aligned, then move its center onto the
        image center) and applied through the central geometry gateway
        ``apply_affine_transform``. This is the key difference from the older
        behaviour, where only ``structure.data`` was rotated while the raw
        ``phase_data`` / ``height_data`` were left untouched - leaving the two
        representations pointing at different geometries and breaking any
        downstream step that reads both (plots, profiles, measurements).

        The output canvas keeps the current shape, matching the previous
        realignment contract. If a rotated structure would be clipped, align
        the processor via ``align_with`` (which sizes an enlarged canvas)
        instead.

        Returns:
            Self for method chaining.

        Raises:
            ValidationError: If no structure has been detected.
        """
        if self.structure is None:
            raise ValidationError("Structure must be detected before realignment.")

        # 2x3 affine: rotate to axis-aligned + translate center -> image center.
        matrix = self.structure.alignment_transform
        self.apply_affine_transform(matrix, output_shape=self.shape)
        self.logger.info(
            "Structure realigned: raw data, structure and segments transformed "
            "together via apply_affine_transform."
        )
        return self

    def segment_structure(
        self,
        method: Literal["grid", "kmeans", "watershed"] = "grid",
        num_segments: Tuple[int, int] = (2, 2),
        **kwargs,
    ) -> "StructureProcessor":
        """
        NOTE Hough lines maybe interessting for automatic structure separation using intensity image
        Segment the detected structure.

        Args:
            method: Segmentation method to use
            num_segments: Number of segments as (rows, cols)
            **kwargs: Additional parameters for the segmentation method

        Returns:
            Self for method chaining

        Raises:
            SegmentationError: If segmentation fails
            ValidationError: If structure is not detected
        """
        try:
            if self.structure is None:
                try:
                    self.detect_structure()
                except Exception as e:
                    raise ValidationError(f"Structure was not detected: {e}")

            # Call segmentation function
            # todo check methods watershed and kmeans - what data is the most suitable for this?
            self._segments = segment_structure(
                self.structure, method=method, num_segments=num_segments, **kwargs
            )

            # Update structure with segments
            self.structure.segments = self._segments

            self.logger.info(
                f"Segmented structure into {len(self._segments)} segments using {method} method"
            )
            return self

        except Exception as e:
            if isinstance(e, (SegmentationError, ValidationError)):
                raise
            raise SegmentationError(f"Structure segmentation failed: {str(e)}")

    def align_with(
        self,
        target: "Union[StructureProcessor, List[StructureProcessor]]",
        method: "Literal['edge']" = "edge",
        scaling: "Literal['upsampling', 'downsampling']" = "downsampling",
        zero_background: bool = True,
        refine: "Union[bool, str]" = "none",  # <-- NEW: 'ecc' | 'phase' | 'none' | bool
        detect_structure_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> "Tuple[Union[StructureProcessor, List[StructureProcessor]], Dict]":
        """
        Align this processor with one or more targets.

        Each structure is rotated to be axis-aligned and its center is moved to
        the output-canvas center; all raw channels (phase/height/intensity) and
        the structure mask are warped with the same affine matrix, so they stay
        consistent. Structures are auto-detected if not present yet. The mask is
        propagated through the warp (and through later padding), never
        re-detected, so the original detection settings are preserved.

        Parameters
        ----------
        target : StructureProcessor | list[StructureProcessor]
            One target or a list of targets.
        method : {'edge'}
            Alignment method. Only 'edge' is implemented; any other value
            raises NotImplementedError (instead of being silently ignored).
        scaling : {'upsampling', 'downsampling'}
            Which image is rescaled to match pixel sizes.
        zero_background : bool
            If True, normalise the background to 0 before warping so that
            newly introduced border pixels (borderValue=0) stay consistent.
        detect_structure_kwargs : dict | None
            kwargs for detect_structure(), used only when a structure must be
            auto-detected before alignment.
        **kwargs
            Forwarded to align_processor_pair / align_processors_many.

        Returns
        -------
        (result, transform_dict)
        """
        try:
            if method != "edge":
                raise NotImplementedError(
                    f"align_with: method='{method}' is not implemented; "
                    "only 'edge' is currently supported."
                )

            detect_kw = detect_structure_kwargs or {}

            # Step 1: ensure every processor has a detected structure.
            # if self.structure is None:
            #     self.logger.info("align_with: auto-detecting structure (self).")
            #     self.detect_structure(**detect_kw)

            targets = target if isinstance(target, list) else [target]
            for t in targets:
                if not isinstance(t, StructureProcessor):
                    raise ValidationError(
                        f"target must be a StructureProcessor, got {type(t)}"
                    )
                if t.structure is None:
                    self.logger.info(
                        f"align_with: auto-detecting structure (target {id(t)})."
                    )
                    t.detect_structure(**detect_kw)

            # Step 2: zero the background so warp border pixels (=0) are clean.
            if zero_background:
                self._zero_background_inplace()
                for t in targets:
                    t._zero_background_inplace()

            # Step 3: run the alignment.
            # Coarse-to-fine: resolve the content-registration refiner that runs
            # on top of the geometric pre-alignment. 'none' restores the legacy
            # centroid-only behaviour.
            from .processing.registration import get_refiner

            kwargs.setdefault("refiner", get_refiner(refine))
            if isinstance(target, list):
                result, transform_dict = align_processors_many(
                    self, target, scaling=scaling, **kwargs
                )
            else:
                _, transform_dict = align_processor_pair(
                    self, target, scaling=scaling, **kwargs
                )
                result = self

            # Step 4: no re-detection. The mask (and any segments) are carried
            # through the warp (INTER_NEAREST) and through padding, preserving
            # the original detection settings. See apply_affine_transform (the
            # geometry gateway) and sync_raw_data_canvas in alignment.py.

            self.logger.info(f"align_with: done (method='{method}').")
            return result, transform_dict

        except Exception as e:
            if isinstance(e, (AlignmentError, ValidationError, NotImplementedError)):
                raise
            raise AlignmentError(f"align_with failed: {str(e)}") from e

    def _resync_structure_to_raw(self, *, validate: bool = True) -> None:
        """
        Rebuild the structure and its segments from the current active raw array.

        This is the shared re-sync primitive for *value* operations (background
        removal, offset shifts, unit conversions): after a raw channel has been
        modified in place, the structure's ``data`` and each segment's ``data``
        are repointed at the current active raw array while their masks are
        preserved. Because such operations do not change geometry, the masks -
        and therefore the cached ``center`` / ``bounds`` / ``corner_points`` -
        stay valid, so the existing ``Structure`` object is kept (only its
        ``.data`` view is updated) rather than rebuilt.

        Geometric operations must NOT use this method: they change the mask
        shape and must go through :meth:`apply_affine_transform`, which rebuilds
        the structure so the geometry caches are reset.

        Args:
            validate: If True (default), run
                :meth:`_assert_data_structure_consistency` afterwards so any
                residual raw/structure divergence becomes an explicit error.

        Notes:
            No-op if no structure has been detected yet.
        """
        if self.structure is None:
            return

        new_active = self._active_data.data
        self.structure.data = ma.masked_array(new_active, mask=self.structure.data.mask)
        for seg in self.structure.segments:
            seg.data = ma.masked_array(new_active, mask=seg.data.mask)

        if validate:
            self._assert_data_structure_consistency()

    def _zero_background_inplace(self) -> None:
        """
        Subtract the mean background value from all raw channels in-place.

        Goal: background ~ 0 so that warpAffine with borderValue=0 produces
        consistent borders. Acts on phase_data, height_data and intensity_data
        together so all channels stay consistent. The structure (and its
        segments) are re-synced afterwards so structure.data never references a
        stale, pre-shift array. Requires a detected structure.

        Raises
        ------
        ValidationError
            If no structure has been detected yet.
        """
        if self.structure is None:
            raise ValidationError(
                "_zero_background_inplace: structure must be detected first."
            )

        # Mask convention: True = masked = background.
        bg_mask = self.structure.data.mask
        if not np.any(~bg_mask):  # no structure pixels -> nothing to normalise
            self.logger.warning(
                "_zero_background_inplace: mask has no structure pixels; skipping."
            )
            return

        # Read background from raw data (not structure.data) to avoid mismatches
        # if structure.data was cropped to a different shape.
        raw = self._active_data.data
        if raw.shape != bg_mask.shape:
            self.logger.warning(
                f"_zero_background_inplace: shape mismatch between raw data "
                f"({raw.shape}) and structure mask ({bg_mask.shape}); skipping."
            )
            return

        bg_pixels = raw[bg_mask]
        if len(bg_pixels) == 0:
            self.logger.warning("_zero_background_inplace: no background pixels found.")
            return

        bg_offset = float(np.mean(bg_pixels))
        self.logger.debug(f"_zero_background_inplace: bg_offset = {bg_offset:.6f}")

        # ------------------------------------------------------------------------------------------- todo WRONG
        # phase image and height have to have indivual offsets - only selected active data should be shifted
        # Shift all available channels by the same offset.
        for attr in ("phase_data", "height_data", "intensity_data"):
            img: Optional[ImageData] = getattr(self, attr, None)
            if img is not None:
                img.update_data(data=img.data - bg_offset)

        # Re-sync the structure (and segments) to the freshly shifted raw data,
        # keeping the existing masks. Shared with adjust_background so the
        # value-correction resync lives in a single place.
        self._resync_structure_to_raw()

    def convert_to_height(
        self,
        n_resin: Optional[float] = None,
        wavelength: Optional[float] = None,
        convert_structure=False,
    ) -> "StructureProcessor":
        """
        Convert phase data to height.

        Args:
            n_resin: Refractive index of the resin (default: from constructor)
            wavelength: Light wavelength in meters (default: from constructor)

        Returns:
            Self for method chaining

        Raises:
            PhaseRetrievalError: If conversion fails
            ValidationError: If required data is missing
        """
        try:
            # todo - not only structure has to be converted but complete data (phase data)
            # Check if we have structure data or phase data
            if convert_structure and self.structure is not None:
                # Convert structure data
                self.structure = convert_phase_to_height(
                    self.structure,
                    n_resin=n_resin or self.n_resin,
                    wavelength=wavelength or self.wavelength,
                )
                self.logger.info("Converted structure phase data to height")

            else:
                if self.phase_data is not None:
                    # Convert image data
                    self.height_data = convert_phase_to_height(
                        self.phase_data,
                        n_resin=n_resin or self.n_resin,
                        wavelength=wavelength or self.wavelength,
                    )
                    self.logger.info("Converted phase data to height")

                else:
                    raise ValidationError(
                        "Phase data or structure must be available for conversion"
                    )

            return self

        except Exception as e:
            if isinstance(e, (ValidationError, ProcessingError)):
                raise
            raise ProcessingError(f"Phase to height conversion failed: {str(e)}")

    def convert_to_phase(
        self, n_resin: Optional[float] = None, wavelength: Optional[float] = None
    ) -> "StructureProcessor":
        """
        Convert height data to phase.

        Args:
            n_resin: Refractive index of the resin (default: from constructor)
            wavelength: Light wavelength in meters (default: from constructor)

        Returns:
            Self for method chaining

        Raises:
            PhaseRetrievalError: If conversion fails
            ValidationError: If required data is missing
        """
        try:
            # Check if we have structure data or phase data
            if self.structure is not None:
                # Convert structure data
                self.structure = convert_height_to_phase(
                    self.structure,
                    n_resin=n_resin or self.n_resin,
                    wavelength=wavelength or self.wavelength,
                )
                self.logger.info("Converted structure height data to phase")

            # todo - converting phase data - phase data from height_2_phase is complete bullshit
            elif self.height_data is not None:
                # Convert image data
                self.phase_data = convert_height_to_phase(
                    self.phase_data,
                    n_resin=n_resin or self.n_resin,
                    wavelength=wavelength or self.wavelength,
                )
                self.logger.info("Converted height data to phase")

            else:
                raise ValidationError(
                    "Height data or structure must be available for conversion"
                )

            return self

        except Exception as e:
            if isinstance(e, (ValidationError, ProcessingError)):
                raise
            raise ProcessingError(f"Height to phase conversion failed: {str(e)}")

    # todo - is this really necessary?
    #   structure data is not the right approach to do ?!
    def unwrap_phase(
        self,
        method: Literal["path_following", "quality_guided"] = "path_following",
        **kwargs,
    ) -> "StructureProcessor":
        """
        Unwrap phase data to remove 2π discontinuities.

        Args:
            method: Unwrapping method to use
            **kwargs: Additional parameters for the method

        Returns:
            Self for method chaining

        Raises:
            PhaseRetrievalError: If unwrapping fails
            ValidationError: If required data is missing
        """
        try:
            # Check if we have structure data or phase data
            if self.structure is not None:
                # Unwrap structure data
                self.structure = unwrap_phase(self.structure, method=method, **kwargs)
                self.logger.info(
                    f"Unwrapped structure phase data using {method} method"
                )

            elif self.phase_data is not None:
                # Unwrap image data
                self.phase_data = unwrap_phase(self.phase_data, method=method, **kwargs)
                self.logger.info(f"Unwrapped phase data using {method} method")

            else:
                raise ValidationError(
                    "Phase data or structure must be available for unwrapping"
                )

            return self

        except Exception as e:
            if isinstance(e, (ValidationError, ProcessingError)):
                raise
            raise ProcessingError(f"Phase unwrapping failed: {str(e)}")

    def phase_retrieval(
        self,
        target_data: Union[StructureProcessor, Structure, List[float]],
        convert_to_height: bool = False,
        **kwargs,
    ) -> "StructureProcessor":
        try:
            if self.structure is None:
                raise ValidationError(
                    "Structure must be detected and segmented before phase retrieval"
                )
            if not self.structure.segments:
                raise ValidationError(
                    "Structure must be segmented before phase retrieval"
                )

            if isinstance(target_data, StructureProcessor):
                if target_data.structure is None:
                    raise ValidationError(
                        "Target processor must have a detected structure"
                    )
                target = target_data.structure
            elif isinstance(target_data, Structure):
                target = target_data
            else:
                target = target_data

            self.structure = full_phase_retrieval(
                self.structure,
                target,
                n_resin=kwargs.pop("n_resin", self.n_resin),
                wavelength=kwargs.pop("wavelength", self.wavelength),
                convert_to_height=convert_to_height,
                **kwargs,
            )

            # Korrektur zurück in den aktiven Rohkanal propagieren - sonst bleiben
            # processor.data / processor.phase_data / processor.height_data (und
            # damit Plots, save_data(), ...) auf den unkorrigierten Werten stehen.
            # Spiegelt das Muster aus adjust_background().
            corrected = self.structure.data.data

            if convert_to_height:
                # Repräsentation wechselt von Phase zu Höhe.
                if self.height_data is not None:
                    self.height_data.update_data(data=corrected)
                else:
                    self.height_data = ImageData(
                        data=corrected, pixel_size=self.structure.pixel_size
                    )
                # phase_data passt nicht mehr zur korrigierten (Höhen-)Struktur -
                # verwerfen, damit _active_data nicht stillschweigend auf die
                # veraltete Phase zurückfällt.
                self.phase_data = None
            else:
                self._active_data.update_data(data=corrected)

            self.logger.info(
                f"Performed phase retrieval{' and converted to height' if convert_to_height else ''}"
            )
            return self

        except Exception as e:
            if isinstance(e, (ValidationError, ProcessingError)):
                raise
            raise ProcessingError(f"Phase retrieval failed: {str(e)}")

    # def phase_retrieval_errorbehaftet(
    #     self,
    #     target_data: Union[StructureProcessor, Structure, List[float]],
    #     convert_to_height: bool = False,
    #     **kwargs,
    # ) -> "StructureProcessor":
    #     """
    #     Perform full phase retrieval using segmented structure.

    #     Args:
    #         target_data: Target phase values or structure
    #         convert_to_height: Whether to convert result to height
    #         **kwargs: Additional parameters for retrieval

    #     Returns:
    #         Self for method chaining

    #     Raises:
    #         PhaseRetrievalError: If retrieval fails
    #         ValidationError: If required data is missing
    #     """
    #     try:
    #         if self.structure is None:
    #             raise ValidationError(
    #                 "Structure must be detected and segmented before phase retrieval"
    #             )

    #         if not self.structure.segments:
    #             raise ValidationError(
    #                 "Structure must be segmented before phase retrieval"
    #             )

    #         # Get target data
    #         if isinstance(target_data, StructureProcessor):
    #             if target_data.structure is None:
    #                 raise ValidationError(
    #                     "Target processor must have a detected structure"
    #                 )
    #             target = target_data.structure
    #         elif isinstance(target_data, Structure):
    #             target = target_data
    #         else:
    #             # Assume list of values
    #             target = target_data

    #         # todo - convert to height is a boolean with default = true, but i want to retrieve the PHASE not height
    #         #   check what the convertion means in the functionality
    #         # Perform phase retrieval
    #         self.structure = full_phase_retrieval(
    #             self.structure,
    #             target,
    #             n_resin=kwargs.pop("n_resin", self.n_resin),
    #             wavelength=kwargs.pop("wavelength", self.wavelength),
    #             convert_to_height=convert_to_height,
    #             **kwargs,
    #         )

    #         self.logger.info(
    #             f"Performed phase retrieval{' and converted to height' if convert_to_height else ''}"
    #         )
    #         return self

    #     except Exception as e:
    #         if isinstance(e, (ValidationError, ProcessingError)):
    #             raise
    #         raise ProcessingError(f"Phase retrieval failed: {str(e)}")

    def adjust_background(
        self, method: Literal["subtract", "plane_fit"] = "subtract", **kwargs
    ) -> "StructureProcessor":
        """
        Remove the background from the active measurement channel.

        Background removal is a *value* correction (a constant offset for
        ``'subtract'`` or a fitted tilted plane for ``'plane_fit'``), in
        contrast to the geometric transforms routed through
        :meth:`apply_affine_transform`. The correction is applied to the active
        raw channel (phase or height) and the structure - together with its
        segments - is rebuilt from that single corrected array, so
        ``structure.data`` and ``_active_data`` never diverge in value. This
        mirrors the resync pattern used by :meth:`_zero_background_inplace`.

        Only the active channel is corrected. The intensity channel is a
        different physical quantity, so subtracting a phase/height background
        from it would be meaningless; it is intentionally left untouched.

        Args:
            method: Background model to remove. ``'subtract'`` removes the mean
                background level; ``'plane_fit'`` fits and removes a plane.
            **kwargs: Additional parameters forwarded to the underlying
                ``adjust_background`` routine. An explicit ``background`` (a
                :class:`Background`) may be passed; otherwise it is derived from
                the current structure mask.

        Returns:
            Self for method chaining.

        Raises:
            ValidationError: If no structure has been detected.
            ProcessingError: If adjustment fails or leaves the data inconsistent.
        """
        try:
            if self.structure is None:
                raise ValidationError(
                    "Structure must be detected before background adjustment"
                )

            # Build the Background from the current structure mask unless the
            # caller supplied one explicitly. Mask convention: True = masked =
            # background, so bg_data leaves the background pixels unmasked.
            background = kwargs.pop("background", None)
            if background is None:
                bg_mask = self.structure.data.mask
                bg_data = ma.masked_array(self.structure.data.data, mask=~bg_mask)
                background = Background(
                    data=bg_data, pixel_size=self.structure.pixel_size
                )

            # Compute the background-removed structure. The underlying routine
            # subtracts the correction from the *entire* canvas array (not just
            # the masked-in region), so ``adjusted.data.data`` is exactly the
            # active raw array with the background removed.
            adjusted = adjust_background(
                self.structure, background, method=method, **kwargs
            )

            # Propagate the correction to the active raw channel, then re-sync
            # the structure and segments through the shared value-correction
            # primitive (same path as _zero_background_inplace) so raw data and
            # structure/segment values stay identical. The masks are unchanged
            # by a value correction, so the existing Structure object is kept.
            self._active_data.update_data(data=adjusted.data.data)
            self._resync_structure_to_raw()

            self.logger.info(f"Adjusted background using {method} method")
            return self

        except Exception as e:
            if isinstance(e, (ValidationError, ProcessingError)):
                raise
            raise ProcessingError(f"Background adjustment failed: {str(e)}")

    def plot_height(self):
        if self.height_data is not None:
            plt.imshow(self.height_data.data.data)
        else:
            raise ValueError("No height data!")

    def plot_image(self, phase_rad=True, **kwargs):
        """
        Plot the phase or structure data as a 2D image.

        Args:
            **kwargs: Additional parameters for the plotting function

        Returns:
            Matplotlib figure object

        Raises:
            VisualizationError: If plotting fails
            ValidationError: If required data is missing
        """
        try:
            from .visualization.plotter_2d import plot_image

            if phase_rad:
                if self.structure is not None:
                    return plot_image(
                        self.structure.data.data, pixel_size=self.pixel_size, **kwargs
                    )
                elif self.phase_data is not None:
                    return plot_image(
                        self.phase_data, pixel_size=self.pixel_size, **kwargs
                    )
                else:
                    raise ValidationError("No data available for plotting")
            else:
                if self.structure is not None:
                    data = self.structure.data.data / (2 * np.pi)
                elif self.phase_data is not None:
                    data = self.phase_data / (2 * np.pi)
                else:
                    raise ValidationError("No data available for plotting")
                return plot_image(data, pixel_size=self.pixel_size, **kwargs)

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ProcessingError(f"Image plotting failed: {str(e)}")

    def plot_comparison(
        self,
        others: Union["StructureProcessor", List["StructureProcessor"]],
        **kwargs,
    ) -> Tuple["plt.Figure", Dict[str, list]]:
        """
        Compare this processor's raw data against 1-2 other processors.

        Plots a heatmap for each processor (self plus `others`) side by
        side, along with a shared profile comparison below. This assumes
        all processors are already aligned to a common canvas (same size,
        orientation and scale) - see `align_with` / `align_processors_many`.

        Always uses the raw, unmasked channel data of each processor (i.e.
        `processor.data`, which resolves to phase or height data), not the
        masked structure region.

        Args:
            others: A single StructureProcessor or a list of 1-2
                StructureProcessor instances to compare against `self`.
                The total number of processors (self + others) must be
                2 or 3.
            **kwargs: Additional parameters forwarded to
                `plot_structure_comparison`, e.g. `titles`, `cbar_labels`,
                `profile_labels`, `profile_axis`, `profile_location`,
                `clip_percentile`, `vmin`, `vmax`, `same_y_lim`,
                `align_zero`, `save_path`, `show_plot`. See that
                function's docstring for the full list of options.

        Returns:
            Tuple of (Figure, axes_dict) - see `plot_structure_comparison`.

        Raises:
            ValidationError: If the number of processors is not 2 or 3.
            VisualizationError: If plotting fails.
        """
        try:
            from .visualization.plotter_2d import plot_structure_comparison

            if isinstance(others, StructureProcessor):
                others = [others]

            processors = [self] + list(others)

            if len(processors) not in (2, 3):
                raise ValidationError(
                    f"plot_comparison supports 2 or 3 processors in total "
                    f"(self + 1-2 others), got {len(processors)}."
                )

            data_list = [p.structure.data.data for p in processors]
            pixel_size_list = [p.pixel_size for p in processors]

            default_titles = [
                (
                    p.description
                    if isinstance(p.description, str) and p.description
                    else f"Processor {i + 1}"
                )
                for i, p in enumerate(processors)
            ]
            kwargs.setdefault("titles", default_titles)

            fig, axes = plot_structure_comparison(
                data_list=data_list,
                pixel_size_list=pixel_size_list,
                **kwargs,
            )

            self.logger.info(
                f"Created comparison plot for {len(processors)} processors"
            )
            return fig, axes

        except Exception as e:
            if isinstance(e, (ValidationError, VisualizationError)):
                raise
            raise ProcessingError(f"Comparison plotting failed: {str(e)}")

    def plot_structure(self, **kwargs):
        """
        Plot the detected structure with optional highlighting.

        Args:
            **kwargs: Additional parameters for the plotting function

        Returns:
            Matplotlib figure object

        Raises:
            VisualizationError: If plotting fails
            ValidationError: If structure is not detected
        """
        try:
            from .visualization.plotter_2d import plot_structure

            if self.structure is None:
                raise ValidationError("Structure must be detected before plotting")

            return plot_structure(self.structure, **kwargs)

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ProcessingError(f"Structure plotting failed: {str(e)}")

    def plot_segments(self, **kwargs):
        """
        Plot the segmented structure.

        Args:
            **kwargs: Additional parameters for the plotting function

        Returns:
            Matplotlib figure object

        Raises:
            VisualizationError: If plotting fails
            ValidationError: If structure is not segmented
        """
        try:
            from .visualization.plotter_2d import plot_segments

            if self.structure is None or not self.structure.segments:
                raise ValidationError(
                    "Structure must be detected and segmented before plotting"
                )

            return plot_segments(self.structure, **kwargs)

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ProcessingError(f"Segment plotting failed: {str(e)}")

    def plot_3d(self, **kwargs):
        """
        Plot the phase or structure data as a 3D surface.

        Args:
            **kwargs: Additional parameters for the plotting function

        Returns:
            Matplotlib figure object

        Raises:
            VisualizationError: If plotting fails
            ValidationError: If required data is missing
        """
        try:
            from .visualization.plotter_3d import plot_surface

            if self.structure is not None:
                return plot_surface(self.structure.data.data, **kwargs)
            elif self.phase_data is not None:
                return plot_surface(self.phase_data, **kwargs)
            else:
                raise ValidationError("No data available for plotting")

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ProcessingError(f"3D plotting failed: {str(e)}")

    def plot_height_map(self, **kwargs):
        """
        Plot a height map with contour lines.

        Args:
            **kwargs: Additional parameters for the plotting function

        Returns:
            Matplotlib figure object

        Raises:
            VisualizationError: If plotting fails
            ValidationError: If required data is missing
        """
        try:
            from .visualization.plotter_3d import plot_height_map

            if self.structure is not None:
                return plot_height_map(self.structure.data.data, **kwargs)
            elif self.phase_data is not None:
                return plot_height_map(self.phase_data, **kwargs)
            else:
                raise ValidationError("No data available for plotting")

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ProcessingError(f"Height map plotting failed: {str(e)}")

    def plot_debug(self, suptitle=None, save_path=None):
        if self.structure is None:
            self.detect_structure()
        plot_structure_properties(
            self.structure, suptitle=suptitle, save_path=save_path
        )

    def save_data(
        self,
        filepath: Union[str, Path],
        data_type: Literal["phase", "height", "structure"] = "phase",
        format: str = "npy",
        **kwargs,
    ) -> str:
        """
        Save phase or structure data to file.

        Args:
            filepath: Path to save the file
            data_type: Type of data to save
            format: File format
            **kwargs: Additional parameters for saving

        Returns:
            Path to saved file

        Raises:
            ProcessingError: If saving fails
            ValidationError: If required data is missing
        """
        try:
            filepath = Path(filepath)

            if data_type == "phase":
                if self.phase_data is None:
                    raise ValidationError("No phase data available for saving")
                self.phase_data.save(filepath, format=format)

                # Save pixel size info
                pixel_info_path = filepath.with_name(f"{filepath.stem}_pixel_info.txt")
                with open(pixel_info_path, "w") as f:
                    f.write(f"pixel_size = {self.pixel_size}\n")

            elif data_type == "height":
                if self.height_data is None:
                    raise ValidationError("No height data available for saving")
                self.height_data.save(filepath, format=format)

                # Save pixel size info
                pixel_info_path = filepath.with_name(f"{filepath.stem}_pixel_info.txt")
                with open(pixel_info_path, "w") as f:
                    f.write(f"pixel_size = {self.pixel_size}\n")

            elif data_type == "structure":
                if self.structure is None:
                    raise ValidationError("No structure data available for saving")

                # Save as numpy array for now
                if format == "npy":
                    np.save(filepath, self.structure.data.filled(np.nan))
                elif format == "csv":
                    np.savetxt(
                        filepath, self.structure.data.filled(np.nan), delimiter=","
                    )
                else:
                    raise ValidationError(
                        f"Unsupported format for structure data: {format}"
                    )

                # Save pixel size info
                pixel_info_path = filepath.with_name(f"{filepath.stem}_pixel_info.txt")
                with open(pixel_info_path, "w") as f:
                    f.write(f"pixel_size = {self.structure.pixel_size}\n")

            else:
                raise ValidationError(f"Unsupported data type: {data_type}")

            self.logger.info(f"Saved {data_type} data to {filepath}")
            return str(filepath)

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ProcessingError(f"Data saving failed: {str(e)}")

    @classmethod
    def from_file(
        cls,
        phase_path: Union[str, Path],
        intensity_path: Optional[Union[str, Path]] = None,
        pixel_size: Optional[Tuple[float, float]] = None,
        **kwargs,
    ) -> "StructureProcessor":
        """
        Create StructureProcessor from files.

        Args:
            phase_path: Path to phase data file
            intensity_path: Path to intensity data file (optional)
            pixel_size: Pixel size in physical units (optional)
            **kwargs: Additional parameters for processor

        Returns:
            New StructureProcessor instance

        Raises:
            DataLoadError: If loading fails
        """
        try:
            # Create processor without data
            processor = cls(**kwargs)

            # Load phase data
            processor.load_phase_data(phase_path, pixel_size)

            # Load intensity data if provided
            if intensity_path is not None:
                processor.load_intensity_data(intensity_path, pixel_size)

            return processor

        except Exception as e:
            if isinstance(e, DataLoadError):
                raise
            raise DataLoadError(f"Failed to create processor from files: {str(e)}")
