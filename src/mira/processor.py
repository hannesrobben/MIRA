"""
Main StructureProcessor class for the image processing library.

This module provides the primary entry point for the library, offering a
unified interface for all operations on image data, structures, and segments.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from sqlite3 import DataError
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy import ma
from scipy.ndimage import binary_dilation
from scipy.ndimage import shift as nd_shift

from .core.image import ImageData
from .core.structure import Background, Segment, Structure
from .processing.alignment import (
    align_processor_pair,
    align_processors_many,
)
from .processing.detection import (
    detect_structure,
)
from .processing.phase_retrieval import (
    adjust_background,
    convert_height_to_phase,
    convert_phase_to_height,
    full_phase_retrieval,
    unwrap_phase,
)
from .processing.segmentation import (
    segment_structure,
)
from .utils.exceptions import (
    AlignmentError,
    DataLoadError,
    ProcessingError,
    SegmentationError,
    ValidationError,
    VisualizationError,
)
from .processing.calibration import (
    DeltaNResult,
    calculate_delta_n as fit_delta_n,
)
from .analysis.comparison import RegressionResult, fit_segment_regression
from .utils.helpers import get_logger

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

# OpenCV and matplotlib are imported where they are used, so that
# `import mira` loads neither of them (MI-05). This constant mirrors
# cv2.INTER_LINEAR for the default argument of apply_affine_transform;
# tests/test_import_hygiene.py checks that both values agree.
_CV2_INTER_LINEAR = 1


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

    # File format version for save_snapshot()/load_snapshot(). Bump this
    # whenever the on-disk payload layout changes in a way older readers
    # could not handle, so a mismatch fails with a clear error instead of a
    # cryptic KeyError deep inside load_snapshot().
    SNAPSHOT_FORMAT_VERSION = 1

    def __init__(
        self,
        phase_data: ImageData | np.ndarray | Path | str | None = None,
        height_data: ImageData | np.ndarray | Path | str | None = None,
        intensity_data: ImageData | np.ndarray | Path | str | None = None,
        pixel_size: tuple[float, float] | None = None,
        n_resin: float = 1.52,
        wavelength: float = 0.6749e-6,
        log_level: int = logging.INFO,
        # NOTE: previously these used dataclasses.field(default=...), which
        # has no special meaning as a plain-method default (this class is
        # not a @dataclass) - it left the literal Field object itself as the
        # default value whenever a caller omitted the argument, silently
        # breaking downstream code that expects an actual bool/str value
        # (e.g. json.dumps in save_snapshot's metadata). Plain defaults fix
        # this without changing behavior for any caller that already passed
        # these explicitly.
        adjusting_background: bool = True,
        data_upsidedown: bool = False,
        description: str | dict | None = None,
        segmentation_method: Literal["manual", "automatic"] = "manual",
        structure_detection_method: Literal["threshold", "intensity"] = "threshold",
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

    @classmethod
    def from_product(cls, product, *, n_resin: float, **kwargs) -> "StructureProcessor":
        """Create a processor from a DHM product (MI-10, decision M2).

        Thin wrapper around :func:`mira.products.to_structure_processor`, the
        one place that converts ``(y, x)`` µm to ``(x, y)`` metres.

        Parameters
        ----------
        product : mira.products.TileProduct or mira.products.LayerProduct
            The product; a tile gives the wrapped phase, a layer the
            unwrapped one.
        n_resin : float
            Refractive index for phase → height; required (no silent
            default, G7).
        **kwargs
            Passed on to :func:`mira.products.to_structure_processor`.

        Returns
        -------
        StructureProcessor
        """
        from .products import to_structure_processor

        return to_structure_processor(product, n_resin=n_resin, **kwargs)

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
        from matplotlib import patches

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
        data: ImageData | np.ndarray | Path | str,
        pixel_size: tuple[float, float] | None = None,
        data_upsidedown=False,
    ) -> StructureProcessor:
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
                    raise DataLoadError(f"Failed to load phase data from file: {e!s}")

            else:
                raise ValidationError(f"Unsupported data type: {type(data)}")

            self.logger.info(f"Loaded phase data with shape {self.phase_data.shape}")
            return self

        except Exception as e:
            if isinstance(e, (DataLoadError, ValidationError)):
                raise
            raise DataLoadError(f"Failed to load phase data: {e!s}")

    def load_height_data(
        self,
        data: ImageData | np.ndarray | Path | str,
        pixel_size: tuple[float, float] | None = None,
        data_upsidedown=False,
    ) -> StructureProcessor:
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
                    raise DataLoadError(f"Failed to load height data from file: {e!s}")

            else:
                raise ValidationError(f"Unsupported data type: {type(data)}")

            self.logger.info(f"Loaded height data with shape {self.height_data.shape}")
            return self

        except Exception as e:
            if isinstance(e, (DataLoadError, ValidationError)):
                raise
            raise DataLoadError(f"Failed to load phase data: {e!s}")

    def load_intensity_data(
        self,
        data: ImageData | np.ndarray | Path | str,
        pixel_size: tuple[float, float] | None = None,
        data_upsidedown=False,
    ) -> StructureProcessor:
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
                        f"Failed to load intensity data from file: {e!s}"
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
            raise DataLoadError(f"Failed to load intensity data: {e!s}")

    def update_data(self, height_data=None, phase_data=None, intensity_data=None):
        if phase_data is None and height_data is None and intensity_data is None:
            raise DataError("No data given!")

        if phase_data is not None:
            self.phase_data.update_data(data=phase_data)
        if height_data is not None:
            self.height_data.update_data(data=height_data)
        if intensity_data is not None:
            self.intensity_data.update_data(data=intensity_data)

    def get_mean_background_value(self, margin: int = 5) -> float | None:
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
    ) -> float | None:
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

    def update_pixel_size(self, pixel_size: tuple[float, float]) -> None:
        """Propagate a new pixel_size to all loaded ImageData objects."""
        for attr in ("phase_data", "height_data", "intensity_data"):
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.pixel_size = pixel_size

    def center_structure(self, matrix: np.ndarray | None = None) -> np.ndarray:
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
        output_shape: tuple[int, int] | None = None,
        *,
        data_interpolation: int = _CV2_INTER_LINEAR,
        border_value: float | None = None,
        validate: bool = True,
    ) -> StructureProcessor:
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
        import cv2

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
            img: ImageData | None = getattr(self, attr, None)
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
        new_segments: list[Segment] = []
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
        method: Literal["threshold", "intensity", "adaptive"] | None = None,
        **kwargs,
    ) -> StructureProcessor:
        """
        Detect structure in phase data.

        Args:
            method: Detection method to use. Defaults to the processor's
                configured ``structure_detection_method`` (set at construction)
                when not given explicitly, so implicit auto-detect calls
                (e.g. from ``segment_structure`` or ``align_with``) honour the
                same detection method the processor was configured with,
                instead of silently falling back to "threshold".
            **kwargs: Additional parameters for the detection method

        Returns:
            Self for method chaining

        Raises:
            ProcessingError: If detection fails
            ValidationError: If required data is missing
        """
        try:
            if method is None:
                method = self.structure_detection_method

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
            raise ProcessingError(f"Structure detection failed: {e!s}")

    def realign_structure(self) -> StructureProcessor:
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
        num_segments: tuple[int, int] = (2, 2),
        **kwargs,
    ) -> StructureProcessor:
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
            raise SegmentationError(f"Structure segmentation failed: {e!s}")

    def align_with(
        self,
        target: StructureProcessor | list[StructureProcessor],
        method: Literal["edge"] = "edge",
        scaling: Literal["upsampling", "downsampling"] = "downsampling",
        zero_background: bool = True,
        refine: bool | str = "none",  # <-- NEW: 'ecc' | 'phase' | 'none' | bool
        detect_structure_kwargs: dict | None = None,
        **kwargs,
    ) -> tuple[StructureProcessor | list[StructureProcessor], dict]:
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
            raise AlignmentError(f"align_with failed: {e!s}") from e

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
            img: ImageData | None = getattr(self, attr, None)
            if img is not None:
                img.update_data(data=img.data - bg_offset)

        # Re-sync the structure (and segments) to the freshly shifted raw data,
        # keeping the existing masks. Shared with adjust_background so the
        # value-correction resync lives in a single place.
        self._resync_structure_to_raw()

    def convert_to_height(
        self,
        n_resin: float | None = None,
        n_air: float = 1.0,
        wavelength: float | None = None,
        convert_structure=False,
    ) -> StructureProcessor:
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
                    n_air=n_air,
                    wavelength=wavelength or self.wavelength,
                )
                self.logger.info("Converted structure phase data to height")

            else:
                if self.phase_data is not None:
                    # Convert image data
                    self.height_data = convert_phase_to_height(
                        self.phase_data,
                        n_air=n_air,
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
            raise ProcessingError(f"Phase to height conversion failed: {e!s}")

    def convert_to_phase(
        self, n_resin: float | None = None, wavelength: float | None = None
    ) -> StructureProcessor:
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
            raise ProcessingError(f"Height to phase conversion failed: {e!s}")

    def calculate_delta_n(
        self,
        reference: "StructureProcessor",
        method: str = "segment_mean",
        offset_pixels: int = 10,
        include_background: bool = True,
        adjust_background: bool = True,
        **kwargs,
    ) -> DeltaNResult:
        """
        Fit delta_n (refractive index difference) by comparing this
        processor's phase data against `reference`'s height data,
        segment by segment.

        Call this on the processor holding phase data (e.g. in-situ DHM);
        pass the processor holding an independent height measurement
        (e.g. LSM, Sensofar) as `reference`. Both must already be
        segmented with the same number of segments in the same physical
        layout - matching the segmentation is the caller's responsibility,
        since automatic segmentation is not yet reliable enough to do
        this safely on its own.

        Args:
            reference:           StructureProcessor holding height data,
                                 segmented identically to self.
            method:              Registered delta_n fitting method (see
                                 `mira.processing.calibration.
                                 available_delta_n_methods()`).
            offset_pixels:       Pixels excluded from every segment/
                                 background edge before computing
                                 statistics (see
                                 `processing.calibration.calculate_delta_n`
                                 for details).
            include_background:  If True, include the background as an
                                 additional data point in the fit.
            adjust_background:   If True, background-adjust both
                                 processors' active channel (via
                                 `adjust_background()`) before fitting, so
                                 the background-relative values used
                                 internally are meaningful. Set to False
                                 if you already adjusted the background
                                 yourself.
            **kwargs:            Forwarded to the selected fitting method.

        Returns:
            DeltaNResult. The fitted scalar is also stored on
            `self.delta_n` for convenience.

        Raises:
            ValidationError: If self has no phase data, reference has no
                height data, either is unsegmented, or their segment
                counts differ.
            ProcessingError: If the fit itself fails.
        """
        try:
            if self.phase_data is None:
                raise ValidationError(
                    "calculate_delta_n must be called on a processor with "
                    "phase_data loaded."
                )
            if reference.height_data is None:
                raise ValidationError(
                    "reference processor must have height_data loaded."
                )
            if self.structure is None or not self.structure.segments:
                raise ValidationError(
                    "self must be segmented (segment_structure()) before "
                    "fitting delta_n."
                )
            if reference.structure is None or not reference.structure.segments:
                raise ValidationError(
                    "reference must be segmented (segment_structure()) "
                    "before fitting delta_n."
                )

            if adjust_background:
                self.adjust_background()
                reference.adjust_background()

            result = fit_delta_n(
                phase_structure=self.structure,
                height_structure=reference.structure,
                wavelength=self.wavelength,
                method=method,
                offset_pixels=offset_pixels,
                include_background=include_background,
                **kwargs,
            )

            self.delta_n = result.delta_n
            self.structure.add_statistic(f"delta_n_{method}", result)
            self.logger.info(
                f"Fitted delta_n = {result.delta_n:.6g} using '{method}' "
                f"(R^2 = {result.r_squared:.4f}, "
                f"n_segments={result.n_segments})"
            )
            return result

        except Exception as e:
            if isinstance(e, (ValidationError, ProcessingError)):
                raise
            raise ProcessingError(f"delta_n fitting failed: {e!s}")

    def fit_segment_regression(
        self,
        reference: "StructureProcessor",
        offset_pixels: int = 10,
        include_background: bool = True,
        relative: bool = True,
        statistic_name: str = "segment_regression",
    ) -> RegressionResult:
        """
        Fit a linear regression of this processor's segment means against
        those of `reference`, and store it in `structure.statistics`.

        Wraps `mira.analysis.comparison.fit_segment_regression` the way
        `calculate_delta_n` wraps the delta_n fit: `reference` provides the
        x values (independent variable, e.g. a Sensofar or LSM height),
        this processor the y values (e.g. the in-situ DHM channel). Each
        segment (and optionally the background) is one data point. The
        active channel of each processor is used as it is; unlike
        `calculate_delta_n`, no background adjustment is applied, because
        `relative=True` already uses background-relative means.

        Parameters
        ----------
        reference : StructureProcessor
            Processor segmented identically to this one (same number of
            segments in the same layout).
        offset_pixels : int, optional
            Pixels excluded from every segment and background edge.
        include_background : bool, optional
            If True, the background is an additional data point.
        relative : bool, optional
            If True, use background-relative segment means.
        statistic_name : str, optional
            Key under which the result is stored with
            `Structure.add_statistic`; an existing entry is overwritten.

        Returns
        -------
        RegressionResult
            Slope, intercept, R^2 and the points used.

        Raises
        ------
        ValidationError
            If either processor is not segmented or the segment counts
            differ.
        ProcessingError
            If the fit itself fails.
        """
        try:
            if self.structure is None or not self.structure.segments:
                raise ValidationError(
                    "self must be segmented (segment_structure()) before "
                    "fitting the segment regression."
                )
            if reference.structure is None or not reference.structure.segments:
                raise ValidationError(
                    "reference must be segmented (segment_structure()) before "
                    "fitting the segment regression."
                )

            result = fit_segment_regression(
                structure_x=reference.structure,
                structure_y=self.structure,
                offset_pixels=offset_pixels,
                include_background=include_background,
                relative=relative,
            )
            self.structure.add_statistic(statistic_name, result)
            self.logger.info(
                f"Segment regression '{statistic_name}': slope = "
                f"{result.slope:.6g}, intercept = {result.intercept:.6g}, "
                f"R^2 = {result.r_squared:.4f}, n = {result.n_points}"
            )
            return result

        except Exception as e:
            if isinstance(e, (ValidationError, ProcessingError)):
                raise
            raise ProcessingError(f"Segment regression failed: {e!s}")

    # todo - is this really necessary?
    #   structure data is not the right approach to do ?!
    def unwrap_phase(
        self,
        method: Literal["path_following", "quality_guided"] = "path_following",
        **kwargs,
    ) -> StructureProcessor:
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
            raise ProcessingError(f"Phase unwrapping failed: {e!s}")

    def phase_retrieval(
        self,
        target_data: StructureProcessor | Structure | list[float],
        convert_to_height: bool = False,
        **kwargs,
    ) -> StructureProcessor:
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

            # Propagate the correction back into the active raw channel. Otherwise
            # processor.data / processor.phase_data / processor.height_data (and
            # with them plots, save_data(), ...) keep the uncorrected values.
            # Mirrors the pattern of adjust_background().
            corrected = self.structure.data.data

            if convert_to_height:
                # The representation changes from phase to height.
                if self.height_data is not None:
                    self.height_data.update_data(data=corrected)
                else:
                    self.height_data = ImageData(
                        data=corrected, pixel_size=self.structure.pixel_size
                    )
                # phase_data no longer matches the corrected (height) structure.
                # Discard it so that _active_data does not silently fall back to
                # the outdated phase.
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
            raise ProcessingError(f"Phase retrieval failed: {e!s}")

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
    ) -> StructureProcessor:
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
            raise ProcessingError(f"Background adjustment failed: {e!s}")

    def plot_height(self):
        import matplotlib.pyplot as plt

        if self.height_data is not None:
            plt.imshow(self.height_data.data.data)
        else:
            raise ValueError("No height data!")

    def plot_image(self, plot_in_rad=True, **kwargs):
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

            if plot_in_rad:
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
            raise ProcessingError(f"Image plotting failed: {e!s}")

    def plot_image_with_profile(
        self,
        data_type: Literal["height", "phase", "intensity"] = "height",
        profile_axis: str = "x",
        profile_location: str | int = "mid",
        phase_rad: bool = True,
        **kwargs,
    ):
        """
        Plot a heatmap of one raw data channel together with an independent
        profile plot, with a reference line on the heatmap marking where the
        profile was extracted.

        Always uses the raw, unmasked channel (`height_data`, `phase_data`,
        or `intensity_data`), not the masked `structure` region - consistent
        with `plot_comparison`.

        Args:
            data_type:         Which channel to plot: "height", "phase", or
                            "intensity".
            profile_axis:       "x" for a horizontal profile (fixed row), "y"
                            for a vertical profile (fixed column).
            profile_location:   Row/column index, or "mid" for the center.
            phase_rad:           Only relevant when `data_type="phase"`. If
                            True (default), phase is plotted in radians.
                            If False, phase is divided by 2*pi and plotted
                            in cycles/multiples of 2*pi instead - applied
                            identically to the heatmap and the profile, so
                            the reference line still lines up correctly.
                            Ignored for "height"/"intensity".
            **kwargs:            Forwarded to
                            `plotter_2d.plot_image_with_profile`, e.g.
                            `cmap`, `fontsize`, `clip_percentile`,
                            `colorbar_label`, `save_path_heatmap`,
                            `save_path_profile`, `show_plot`. See that
                            function's docstring for the full list.

        Returns:
            `HeatmapProfileResult` with both figures/axes and the extracted
            profile data.

        Raises:
            ValidationError: If `data_type` is invalid or the requested
                channel has not been loaded.
            VisualizationError: If plotting fails.
        """
        try:
            from .visualization.plotter_2d import (
                plot_image_with_profile as _plot_image_with_profile,
            )

            channel_map = {
                "height": self.height_data,
                "phase": self.phase_data,
                "intensity": self.intensity_data,
            }
            if data_type not in channel_map:
                raise ValidationError(
                    f"Invalid data_type {data_type!r}. Choose from {list(channel_map)}."
                )

            image_data = channel_map[data_type]
            if image_data is None:
                raise ValidationError(f"No {data_type} data has been loaded.")

            # Phase-specific unit handling. Kept here rather than in
            # plotter_2d.py, which stays deliberately unaware of what a
            # "phase" is - it only ever plots generic arrays. Dividing here
            # and passing a plain ndarray onward keeps that separation
            # intact; pixel_size is forwarded explicitly since the plotting
            # function can only auto-resolve it from an ImageData instance.
            plot_target = image_data
            if data_type == "phase":
                if phase_rad:
                    kwargs.setdefault("colorbar_label", "Phase [rad]")
                else:
                    plot_target = image_data.data / (2 * np.pi)
                    kwargs.setdefault("pixel_size", image_data.pixel_size)
                    kwargs.setdefault("colorbar_label", "Phase [\u00d7 2\u03c0]")

            return _plot_image_with_profile(
                plot_target,
                profile_axis=profile_axis,
                profile_location=profile_location,
                **kwargs,
            )

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ProcessingError(f"Image/profile plotting failed: {e!s}")

    def plot_comparison(
        self,
        others: StructureProcessor | list[StructureProcessor],
        **kwargs,
    ) -> tuple[plt.Figure, dict[str, list]]:
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
            raise ProcessingError(f"Comparison plotting failed: {e!s}")

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
            raise ProcessingError(f"Structure plotting failed: {e!s}")

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
            raise ProcessingError(f"Segment plotting failed: {e!s}")

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
            raise ProcessingError(f"3D plotting failed: {e!s}")

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
            raise ProcessingError(f"Height map plotting failed: {e!s}")

    def plot_debug(self, suptitle=None, save_path=None):
        from .visualization.visualization_debug import plot_structure_properties

        if self.structure is None:
            self.detect_structure()
        plot_structure_properties(
            self.structure, suptitle=suptitle, save_path=save_path
        )

    def save_data(
        self,
        filepath: str | Path,
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
            raise ProcessingError(f"Data saving failed: {e!s}")

    # ------------------------------------------------------------------ #
    # Full-state snapshot (save/load)                                     #
    # ------------------------------------------------------------------ #
    # Unlike Structure.save_compressed (a self-contained snapshot of one   #
    # structure, including its own copy of the data), a processor         #
    # snapshot captures every currently loaded raw channel (phase/height/ #
    # intensity) in its CURRENT form - i.e. after any alignment,          #
    # background adjustment, or other in-place processing - plus the      #
    # structure mask and segment masks needed to reconstruct              #
    # self.structure WITHOUT re-running detect_structure() (which would   #
    # silently discard custom detection parameters). The structure's mask #
    # is stored via Structure._mask_payload() rather than                 #
    # Structure.save_compressed(), so the underlying data array is        #
    # written exactly once - as the active raw channel - never duplicated #
    # inside the structure's own payload.                                 #
    # ------------------------------------------------------------------ #

    def _snapshot_metadata(self) -> dict:
        """
        Collect the processing parameters and description needed to
        reproduce this processor's *behaviour* - as opposed to its raster
        geometry, which lives in the raw arrays and masks instead.

        Kept as a small, plain dict (JSON-serializable) rather than being
        folded into the main array payload, so it can optionally be written
        out on its own as a human-readable ".json" sidecar file (see
        ``save_snapshot``'s ``save_metadata_sidecar`` argument) - useful for
        a quick "what was this measurement?" look without loading the full
        (potentially large) array data.

        Returns:
            Dict of JSON-serializable processing metadata.
        """
        return {
            "format_version": self.SNAPSHOT_FORMAT_VERSION,
            "n_resin": self.n_resin,
            "wavelength": self.wavelength,
            "description": self.description,
            "segmentation_method": self.segmentation_method,
            "structure_detection_method": self.structure_detection_method,
            "data_upsidedown": self.data_upsidedown,
            "adjusting_background": self.adjusting_background,
        }

    def save_snapshot(
        self,
        file_path: str | Path,
        include_intensity: bool = True,
        save_metadata_sidecar: bool = False,
        dtype: np.dtype = np.float32,
    ) -> Path:
        """
        Save the full processor state to a single compressed .npz file.

        Captures every currently loaded raw channel (phase / height /
        intensity) in its current, possibly already-processed form,
        together with the structure mask and segment masks (if a structure
        has been detected). See the module-level comment above this method
        for how this differs from ``Structure.save_compressed``.

        Args:
            file_path: Destination path. A ".npz" suffix is appended if
                missing. Parent directories are created as needed.
            include_intensity: If True (default) and intensity data is
                loaded, include it in the snapshot. Set False to skip it,
                e.g. once it is no longer needed and file size matters.
            save_metadata_sidecar: If True, additionally write the
                processing metadata (n_resin, wavelength, description,
                detection/segmentation method, ...) as a plain ".json" file
                next to the .npz (same stem, ".json" suffix), so it can be
                inspected without loading the full array data.
            dtype: Floating-point dtype used to store the raw channel
                arrays. Defaults to float32, which halves file size
                relative to float64 with no meaningful precision loss for
                phase/height measurement data (their actual measurement
                noise floor is far above float32 precision).

        Returns:
            The resolved Path the .npz file was written to.

        Raises:
            ProcessingError: If the processor's raw and structure data are
                found to be inconsistent (checked before writing, so a
                mismatched state is never silently persisted).
        """
        file_path = Path(file_path)
        if file_path.suffix != ".npz":
            file_path = file_path.with_suffix(".npz")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Fail loudly before writing rather than silently persisting a
        # raw/structure mismatch that would only surface later, on load.
        self._assert_data_structure_consistency()

        payload: dict[str, np.ndarray] = {
            "format_version": np.array(self.SNAPSHOT_FORMAT_VERSION)
        }

        for attr, prefix in (
            ("phase_data", "phase"),
            ("height_data", "height"),
            ("intensity_data", "intensity"),
        ):
            if attr == "intensity_data" and not include_intensity:
                continue
            img: ImageData | None = getattr(self, attr, None)
            if img is not None:
                payload[f"{prefix}_data"] = img.data.astype(dtype)
                payload[f"{prefix}_pixel_size"] = np.array(
                    img.pixel_size, dtype=np.float64
                )

        if self.structure is not None:
            payload.update(self.structure._mask_payload())

        metadata = self._snapshot_metadata()
        # Stored as a 0-d array so it round-trips cleanly through npz; str()
        # around the loaded value undoes the numpy string wrapping on load.
        payload["metadata_json"] = np.array(json.dumps(metadata))

        # Atomic write: build the file under a temp name, then rename, so a
        # crash mid-write never leaves a corrupt snapshot at file_path.
        # np.savez_compressed silently appends ".npz" unless the filename
        # already ends with ".npz" - so the temp name must end in ".npz"
        # too, or the rename below would fail to find its source file.
        tmp_path = file_path.with_name(file_path.stem + ".tmp.npz")
        np.savez_compressed(tmp_path, **payload)
        tmp_path.replace(file_path)

        if save_metadata_sidecar:
            sidecar_path = file_path.with_suffix(".json")
            with open(sidecar_path, "w") as f:
                json.dump(metadata, f, indent=2)

        self.logger.info(f"Saved processor snapshot to {file_path}")
        return file_path

    @classmethod
    def load_snapshot(cls, file_path: str | Path) -> StructureProcessor:
        """
        Reconstruct a StructureProcessor previously written by
        ``save_snapshot``.

        Every raw channel present in the file is restored through the
        normal ``load_phase_data`` / ``load_height_data`` /
        ``load_intensity_data`` entry points (each wrapped in ``ImageData``,
        preserving its own pixel size), exactly as if it had just been
        loaded from a fresh measurement. If a structure mask was saved,
        ``self.structure`` is rebuilt directly from that mask combined with
        the freshly loaded active channel - via
        ``Structure._from_mask_payload`` - instead of calling
        ``detect_structure()`` again, so any custom detection parameters
        the original structure was found with (e.g. a manually chosen
        percentile threshold) are preserved exactly rather than silently
        replaced by defaults.

        Args:
            file_path: Path to a .npz file written by ``save_snapshot``.

        Returns:
            A new, fully reconstructed StructureProcessor.

        Raises:
            DataLoadError: If the file is missing required keys or was
                written by an incompatible format version.
        """
        file_path = Path(file_path)
        with np.load(file_path, allow_pickle=False) as npz:
            if "format_version" not in npz.files or "metadata_json" not in npz.files:
                raise DataLoadError(
                    f"{file_path}: not a valid StructureProcessor snapshot "
                    "(missing 'format_version' or 'metadata_json')."
                )
            version = int(npz["format_version"])
            if version != cls.SNAPSHOT_FORMAT_VERSION:
                raise DataLoadError(
                    f"{file_path}: unsupported snapshot format version "
                    f"{version} (expected {cls.SNAPSHOT_FORMAT_VERSION})."
                )
            payload = {key: npz[key] for key in npz.files}

        metadata = json.loads(str(payload["metadata_json"]))

        # Bypass __init__: it mandates phase_data or height_data be passed
        # up front and can prompt interactively (detect_structure's "p"/"h"
        # question), neither of which fits reconstructing from a snapshot.
        # Every field __init__ would normally set is assigned explicitly
        # below instead - keep this list in sync if __init__ gains fields.
        processor = cls.__new__(cls)
        processor.logger = get_logger("StructureProcessor")
        processor.n_resin = metadata["n_resin"]
        processor.delta_n = None
        processor.wavelength = metadata["wavelength"]
        processor.phase_data = None
        processor.height_data = None
        processor.intensity_data = None
        processor.structure = None
        processor._segments = []
        processor.data_upsidedown = metadata["data_upsidedown"]
        processor.adjusting_background = metadata["adjusting_background"]
        processor.description = metadata["description"]
        processor.segmentation_method = metadata["segmentation_method"]
        processor.structure_detection_method = metadata["structure_detection_method"]

        for prefix, loader in (
            ("phase", processor.load_phase_data),
            ("height", processor.load_height_data),
            ("intensity", processor.load_intensity_data),
        ):
            key = f"{prefix}_data"
            if key in payload:
                pixel_size = tuple(payload[f"{prefix}_pixel_size"].tolist())
                loader(ImageData(data=payload[key], pixel_size=pixel_size))

        if "structure_mask" in payload:
            raw = processor._active_data.data
            processor.structure = Structure._from_mask_payload(payload, raw)

        processor.logger.info(f"Loaded processor snapshot from {file_path}")
        return processor

    @classmethod
    def from_file(
        cls,
        phase_path: str | Path,
        intensity_path: str | Path | None = None,
        pixel_size: tuple[float, float] | None = None,
        **kwargs,
    ) -> StructureProcessor:
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
            raise DataLoadError(f"Failed to create processor from files: {e!s}")
