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
from scipy.ndimage import binary_dilation
from scipy.ndimage import shift as nd_shift
from pathlib import Path
from typing import Union, Optional, Tuple, List, Dict, Literal
import logging

from .core.image import ImageData
from .core.structure import Structure, Background
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

        # Load data if provided
        if phase_data is not None:
            self.load_phase_data(phase_data, pixel_size)
            if intensity_data is not None:
                self.load_intensity_data(intensity_data, pixel_size)
        elif height_data is not None:
            self.load_height_data(height_data, pixel_size)  # todo make function
        else:
            raise DataLoadError("No data given. Provide phase or height data.")

        self.adjusting_background = adjusting_background
        self.data_upsidedown = data_upsidedown
        self.description = description
        self.segmentation_method = segmentation_method
        self.structure_detection_method = structure_detection_method

    @property
    def _active_data(self) -> ImageData:
        # todo future: insert flag or something similar to know what active data is
        if self.phase_data is None and self.height_data is None:
            raise DataLoadError("No data given. Provide phase or height data.")
        if self.phase_data is None and self.height_data is not None:
            return self.height_data
        elif self.height_data is None and self.phase_data is not None:
            return self.phase_data
        else:  # height and phase data is given - for now phase data is
            return self.phase_data

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

            elif isinstance(data, np.ndarray):
                if pixel_size is None:
                    raise ValidationError(
                        "Pixel size must be provided for numpy array input"
                    )
                self.phase_data = ImageData(data, pixel_size)

            elif isinstance(data, (str, Path)):
                # Load from file
                try:
                    # todo check loading from file - because pixel size to be 1.0|1.0 is sus
                    self.phase_data = ImageData.from_file(
                        data, pixel_size or (1.0, 1.0)
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

            elif isinstance(data, np.ndarray):
                if pixel_size is None:
                    raise ValidationError(
                        "Pixel size must be provided for numpy array input"
                    )
                self.height_data = ImageData(data, pixel_size)

            elif isinstance(data, (str, Path)):
                # Load from file
                try:
                    self.height_data = ImageData.from_file(
                        data, pixel_size or (1.0, 1.0)
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

            elif isinstance(data, np.ndarray):
                if pixel_size is None and self.phase_data is not None:
                    # Use pixel size from phase data if available
                    pixel_size = self.phase_data.pixel_size
                elif pixel_size is None:
                    raise ValidationError(
                        "Pixel size must be provided for numpy array input"
                    )

                self.intensity_data = ImageData(data, pixel_size)

            elif isinstance(data, (str, Path)):
                # Load from file
                if pixel_size is None and self.phase_data is not None:
                    # Use pixel size from phase data if available
                    pixel_size = self.phase_data.pixel_size

                try:
                    self.intensity_data = ImageData.from_file(
                        data, pixel_size or (1.0, 1.0)
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
        # Shape-Guard — verhindert den ursprünglichen Fehler
        if raw.shape != bg_mask.shape:
            self.logger.warning(
                f"get_mean_background_value: Shape-Mismatch "
                f"(raw={raw.shape}, mask={bg_mask.shape}), gebe None zurück."
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
        Realign the structure by rotating and centering it using the
        alignment transform computed from the structure's corner points.

        Returns:
            Self for method chaining.

        Raises:
            ValidationError: If no structure has been detected.
        """
        if self.structure is None:
            raise ValidationError("Structure must be detected before realignment.")

        self.structure = self.structure.realign()
        self.logger.info("Structure realigned using alignment transform.")
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

    '''
    def align_with(
        self,
        target: "Union[StructureProcessor, List[StructureProcessor]]",
        method: "Literal['edge']" = "edge",
        scaling: "Literal['upsampling', 'downsampling']" = "downsampling",
        **kwargs,
    ) -> "Tuple[Union[StructureProcessor, List[StructureProcessor]], Dict]":
        """
        Richte self mit einem oder mehreren Targets aus.

        Parameters
        ----------
        target : StructureProcessor | list[StructureProcessor]
            Ein Zielobjekt oder eine Liste davon.
        method : {'edge'}
            Alignment-Methode.
        scaling : {'upsampling', 'downsampling'}
            Welches Bild skaliert wird (nach Bildgröße entschieden).
        **kwargs
            Weitere Argumente, z.B. ``visualize_line_detection=False``.

        Returns
        -------
        result, transform_dict
            **Einzelnes Target**: result = self (in-place mutiert),
            transform_dict = Info des Paares.

            **Mehrere Targets**: result = [self, *aligned_targets],
            transform_dict[i] = Info für (self, targets[i]),
            transform_dict['canvas_shape'] = finale gemeinsame Bildgröße.

        Examples
        --------
        >>> dhm_devel, t = dhm_devel.align_with(sensofar)

        >>> all_procs, t = dhm_devel.align_with([dhm_insitu, sensofar])
        >>> dhm_devel, dhm_insitu, sensofar = all_procs
        """
        try:
            if self.structure is None:
                self.detect_structure()

            if isinstance(target, list):
                return align_processors_many(self, target, scaling=scaling, **kwargs)
            else:
                if not isinstance(target, StructureProcessor):
                    raise ValidationError(
                        f"target muss StructureProcessor sein, erhalten: {type(target)}"
                    )
                if target.structure is None:
                    target.detect_structure()

                _, transform_dict = align_processor_pair(
                    self, target, scaling=scaling, **kwargs
                )

                self.detect_structure()
                target.detect_structure()

                self.logger.info(
                    f"align_with (1-zu-1): ausgerichtet mit method='{method}'"
                )
                return self, transform_dict

        except Exception as e:
            if isinstance(e, (AlignmentError, ValidationError)):
                raise
            raise AlignmentError(f"align_with fehlgeschlagen: {str(e)}")
    '''

    def align_with(
        self,
        target: "Union[StructureProcessor, List[StructureProcessor]]",
        method: "Literal['edge']" = "edge",
        scaling: "Literal['upsampling', 'downsampling']" = "downsampling",
        zero_background: bool = True,
        detect_structure_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> "Tuple[Union[StructureProcessor, List[StructureProcessor]], Dict]":
        """
        Richte self mit einem oder mehreren Targets aus.

        Vor dem Alignment wird optional der Hintergrund auf 0 normiert
        (``zero_background=True``), sodass beim Warp sauber Nullen am Rand
        angehängt werden können.  Die Struktur wird automatisch detektiert,
        falls noch nicht geschehen.

        Parameters
        ----------
        target : StructureProcessor | list[StructureProcessor]
            Ein Zielobjekt oder eine Liste davon.
        method : {'edge'}
            Alignment-Methode.
        scaling : {'upsampling', 'downsampling'}
            Welches Bild skaliert wird.
        zero_background : bool
            Wenn True: Hintergrund wird vor dem Warp auf 0 normiert und
            danach wieder auf dem transformierten Bild detektiert.
            Wenn False: Verhalten wie bisher (nicht empfohlen bei Randwerten ≠ 0).
        detect_structure_kwargs : dict | None
            Keyword-Argumente für ``detect_structure()`` (z.B. method, threshold).
        **kwargs
            Weitere Argumente für ``align_processor_pair`` / ``align_processors_many``.

        Returns
        -------
        result, transform_dict
        """
        try:
            detect_kw = detect_structure_kwargs or {}

            # ── Schritt 1: Struktur sicherstellen ─────────────────────────────
            if self.structure is None:
                self.logger.info(
                    "align_with: Struktur wird automatisch detektiert (self)."
                )
                self.detect_structure(**detect_kw)

            targets = target if isinstance(target, list) else [target]
            for t in targets:
                if not isinstance(t, StructureProcessor):
                    raise ValidationError(
                        f"target muss StructureProcessor sein, erhalten: {type(t)}"
                    )
                if t.structure is None:
                    self.logger.info(
                        f"align_with: Struktur wird automatisch detektiert (target {id(t)})."
                    )
                    t.detect_structure(**detect_kw)

            # ── Schritt 2: Hintergrund auf 0 normieren ────────────────────────
            # Analogie: Münze auf den Tisch legen, der schon auf 0 kalibriert ist.
            # Dann können beim Verschieben Nullen am Rand angehängt werden —
            # ohne dass ein fremder Grauton "eingeschmuggelt" wird.
            if zero_background:
                self._zero_background_inplace()
                for t in targets:
                    t._zero_background_inplace()

            # ── Schritt 3: Alignment ──────────────────────────────────────────
            if isinstance(target, list):
                result, transform_dict = align_processors_many(
                    self, target, scaling=scaling, **kwargs
                )
            else:
                _, transform_dict = align_processor_pair(
                    self, target, scaling=scaling, **kwargs
                )
                result = self

            # ── Schritt 4: Struktur neu detektieren (auf transformierten Daten) ──
            # Nach dem Warp hat das Bild eine neue Shape → alte Maske ist ungültig.
            self.detect_structure(**detect_kw)
            for t in targets:
                t.detect_structure(**detect_kw)

            self.logger.info(f"align_with: abgeschlossen mit method='{method}'.")
            return result, transform_dict

        except Exception as e:
            if isinstance(e, (AlignmentError, ValidationError)):
                raise
            raise AlignmentError(f"align_with fehlgeschlagen: {str(e)}")

    def _zero_background_inplace(self) -> None:
        """
        Subtrahiert den mittleren Hintergrundwert von allen Rohdaten *in-place*.

        Das Ziel: Hintergrund ≈ 0, sodass warpAffine mit borderValue=0 konsistente
        Ränder erzeugt.  Die Struktur muss vorher detektiert sein.

        Wirkt auf phase_data, height_data und intensity_data gleichzeitig,
        damit alle Kanäle konsistent bleiben.

        Raises
        ------
        ValidationError
            Wenn noch keine Struktur detektiert wurde.
        """
        if self.structure is None:
            raise ValidationError(
                "_zero_background_inplace: Struktur muss vorher detektiert sein."
            )

        # Hintergrundpixel: Maske True = maskiert = Hintergrund
        bg_mask = self.structure.data.mask  # True außerhalb der Struktur
        structure_present = np.any(~bg_mask)  # Sicherheitsprüfung

        if not structure_present:
            self.logger.warning(
                "_zero_background_inplace: Maske enthält keine Strukturpixel — "
                "Hintergrundnormierung wird übersprungen."
            )
            return

        # Hintergrundpixel aus den Rohdaten lesen (nicht aus structure.data,
        # um Shape-Mismatches zu vermeiden, falls structure.data bereits
        # auf eine andere Shape gecroppt wurde).
        raw = self._active_data.data
        if raw.shape != bg_mask.shape:
            self.logger.warning(
                f"_zero_background_inplace: Shape-Mismatch zwischen Rohdaten "
                f"({raw.shape}) und Strukturmaske ({bg_mask.shape}). "
                "Hintergrundnormierung wird übersprungen."
            )
            return

        bg_pixels = raw[bg_mask]
        if len(bg_pixels) == 0:
            self.logger.warning(
                "_zero_background_inplace: Keine Hintergrundpixel gefunden."
            )
            return

        bg_offset = float(np.mean(bg_pixels))
        self.logger.debug(f"_zero_background_inplace: bg_offset = {bg_offset:.6f}")

        # Alle verfügbaren Kanäle gleichmäßig verschieben
        for attr in ("phase_data", "height_data", "intensity_data"):
            img: Optional[ImageData] = getattr(self, attr, None)
            if img is not None:
                img.update_data(data=img.data - bg_offset)

    def convert_to_height(
        self, n_resin: Optional[float] = None, wavelength: Optional[float] = None
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
            if self.structure is not None:
                # Convert structure data
                self.structure = convert_phase_to_height(
                    self.structure,
                    n_resin=n_resin or self.n_resin,
                    wavelength=wavelength or self.wavelength,
                )
                self.logger.info("Converted structure phase data to height")

            elif self.phase_data is not None:
                # Convert image data
                self.phase_data = convert_phase_to_height(
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
            elif self.phase_data is not None:
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
        convert_to_height: bool = True,
        **kwargs,
    ) -> "StructureProcessor":
        """
        Perform full phase retrieval using segmented structure.

        Args:
            target_data: Target phase values or structure
            convert_to_height: Whether to convert result to height
            **kwargs: Additional parameters for retrieval

        Returns:
            Self for method chaining

        Raises:
            PhaseRetrievalError: If retrieval fails
            ValidationError: If required data is missing
        """
        try:
            if self.structure is None:
                raise ValidationError(
                    "Structure must be detected and segmented before phase retrieval"
                )

            if not self.structure.segments:
                raise ValidationError(
                    "Structure must be segmented before phase retrieval"
                )

            # Get target data
            if isinstance(target_data, StructureProcessor):
                if target_data.structure is None:
                    raise ValidationError(
                        "Target processor must have a detected structure"
                    )
                target = target_data.structure
            elif isinstance(target_data, Structure):
                target = target_data
            else:
                # Assume list of values
                target = target_data

            # todo - convert to height is a boolean with default = true, but i want to retrieve the PHASE not height
            #   check what the convertion means in the functionality
            # Perform phase retrieval
            self.structure = full_phase_retrieval(
                self.structure,
                target,
                n_resin=kwargs.pop("n_resin", self.n_resin),
                wavelength=kwargs.pop("wavelength", self.wavelength),
                convert_to_height=convert_to_height,
                **kwargs,
            )

            self.logger.info(
                f"Performed phase retrieval{' and converted to height' if convert_to_height else ''}"
            )
            return self

        except Exception as e:
            if isinstance(e, (ValidationError, ProcessingError)):
                raise
            raise ProcessingError(f"Phase retrieval failed: {str(e)}")

    def adjust_background(
        self, method: Literal["subtract", "plane_fit"] = "subtract", **kwargs
    ) -> "StructureProcessor":
        """
        Adjust structure data by removing background.

        Args:
            method: Adjustment method to use
            **kwargs: Additional parameters for the method

        Returns:
            Self for method chaining

        Raises:
            ProcessingError: If adjustment fails
            ValidationError: If required data is missing
        """
        try:
            if self.structure is None:
                raise ValidationError(
                    "Structure must be detected before background adjustment"
                )

            # Create background if not explicitly provided
            background = kwargs.pop("background", None)

            if background is None:
                # Extract background from structure mask
                bg_mask = self.structure.data.mask
                bg_data = ma.masked_array(self.structure.data.data, mask=~bg_mask)

                background = Background(
                    data=bg_data, pixel_size=self.structure.pixel_size
                )

            # Adjust background
            self.structure = adjust_background(
                self.structure, background, method=method, **kwargs
            )

            self.logger.info(f"Adjusted background using {method} method")
            return self

        except Exception as e:
            if isinstance(e, (ValidationError, ProcessingError)):
                raise
            raise ProcessingError(f"Background adjustment failed: {str(e)}")

    def plot_image(self, **kwargs):
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

            if self.structure is not None:
                return plot_image(
                    self.structure.data.data, pixel_size=self.pixel_size, **kwargs
                )
            elif self.phase_data is not None:
                return plot_image(self.phase_data, pixel_size=self.pixel_size, **kwargs)
            else:
                raise ValidationError("No data available for plotting")

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

            data_list = [p.data for p in processors]
            pixel_size_list = [p.pixel_size for p in processors]

            default_titles = [
                p.description if isinstance(p.description, str) and p.description
                else f"Processor {i + 1}"
                for i, p in enumerate(processors)
            ]
            kwargs.setdefault("titles", default_titles)

            fig, axes = plot_structure_comparison(
                data_list=data_list,
                pixel_size_list=pixel_size_list,
                **kwargs,
            )

            self.logger.info(f"Created comparison plot for {len(processors)} processors")
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

    def plot_debug(self, suptitle=None):
        if self.structure is None:
            self.detect_structure()
        plot_structure_properties(self.structure, suptitle=suptitle)

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


"""
    def align_with_old(
        self,
        target: StructureProcessor | list[StructureProcessor],
        method: Literal["edge", "mask", "feature"] = "edge",
        scaling: Literal["upsampling", "downsampling"] = "downsampling",
        **kwargs,
    ) -> "StructureProcessor":
        '''
        Align this processor's structure with a target.

        Args:
            target: Target processor or structure to align with
            method: Alignment method to use
            scaling: Scaling approach for pixel size differences
            **kwargs: Additional parameters for the alignment method

        Returns:
            Self for method chaining

        Raises:
            AlignmentError: If alignment fails
            ValidationError: If structure is not detected
        '''
        try:
            if self.structure is None:
                try:
                    self.detect_structure()
                except Exception:
                    raise ValidationError("Structure detection failed: {e}")

            # Get target structure
            if isinstance(target, StructureProcessor):
                if target.structure is None:
                    try:
                        target.detect_structure()
                    except Exception:
                        raise ValidationError("Target structure detection failed: {e}")
                target_structure = target.structure
            else:
                target_structure = target

            # Call alignment function
            # todo (new)    --- should be structure processor and not only a structure type
            #   - up/ downsampling here!
            #   - only method and the resizing/padding/alignment in the other method
            #   - transform dictionary always as a return
            #   - saving the structures
            #       - how to do with the chaining and different structure type
            # ------------------------------------------------------------------
            # todo - return transform - important for intensity image
            #   not structure has to be given as input but the phase-/height data?!
            #   investigate method in respect to self.structure information available

            self.structure = align_structures(
                self.structure,
                target_structure,
                method=method,
                scaling=scaling,
                return_transform=False,
                **kwargs,
            )

            self.logger.info(f"Aligned structure using {method} method")
            return self

        except Exception as e:
            if isinstance(e, (AlignmentError, ValidationError)):
                raise
            raise AlignmentError(f"Structure alignment failed: {str(e)}")


"""
