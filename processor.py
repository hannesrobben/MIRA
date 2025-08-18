"""
Main StructureProcessor class for the image processing library.

This module provides the primary entry point for the library, offering a
unified interface for all operations on image data, structures, and segments.
"""
from dataclasses import field

import numpy as np
import numpy.ma as ma
from pathlib import Path
from typing import Union, Optional, Tuple, List, Dict, Literal, Any, Type, overload
import logging

from .core.image import ImageData
from .core.structure import Structure, Segment, Background

from .processing.detection import (
    detect_structure, detect_edges, detect_corners,
    threshold_detection, intensity_detection, adaptive_detection
)

from .processing.segmentation import (
    segment_structure, segment_by_grid, segment_by_kmeans,
    segment_by_watershed, create_labeled_mask
)

from .processing.alignment import (
    align_structures, calculate_rotation, normalize_pixel_size,
    pad_to_same_size, center_structures
)

from .processing.phase_retrieval import (
    convert_phase_to_height, convert_height_to_phase,
    unwrap_phase, full_phase_retrieval, adjust_background
)

from .utils.helpers import get_logger
from .utils.exceptions import (
    DataLoadError, ValidationError, ProcessingError,
    StructureError, SegmentationError, AlignmentError
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
    adjusting_background: bool = field(default=True)
    data_upsidedown: bool = field(default=False)
    description: [str, dict] = field(default=None)
    segmentation_method: Literal["manual", "automatic"] = field(default="manual")
    structure_detection_method: Literal["threshold", "intensity"] = field(
        default="threshold")

    # Todo - Was ist mit height?
    def __init__(self,
                 phase_data: Optional[Union[ImageData, np.ndarray, Path, str]] = None,
                 intensity_data: Optional[Union[ImageData, np.ndarray, Path, str]] = None,
                 pixel_size: Optional[Tuple[float, float]] = None,
                 n_resin: float = 1.52,
                 wavelength: float = 0.6749e-6,
                 log_level: int = logging.INFO):
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
        self.logger.debug("Initializing StructureProcessor")

        # Store optical parameters
        self.n_resin = n_resin
        self.delta_n = None  # Difference in refractive index - for easier use in the future (printed data for fitting)
        self.wavelength = wavelength

        # Initialize image data attributes
        # Todo - include height data attribute (for use with offline measurement methods like LSM, CM)
        #   include height data and change the methods (phase -> height and height -> phase) accordingly
        #   check the interconnectivity between data (phase, intensity, height) and the structure class
        self.phase_data = None
        self.intensity_data = None
        self.structure = None
        self._segments = []

        # Load data if provided
        if phase_data is not None:
            self.load_phase_data(phase_data, pixel_size)

        if intensity_data is not None:
            self.load_intensity_data(intensity_data, pixel_size)

    def load_phase_data(self,
                        data: Union[ImageData, np.ndarray, Path, str],
                        pixel_size: Optional[Tuple[float, float]] = None) -> 'StructureProcessor':
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
                    raise ValidationError("Pixel size must be provided for numpy array input")
                self.phase_data = ImageData(data, pixel_size)

            elif isinstance(data, (str, Path)):
                # Load from file
                try:
                    # todo check loading from file - because pixel size to be 1.0|1.0 is sus
                    self.phase_data = ImageData.from_file(data, pixel_size or (1.0, 1.0))
                except Exception as e:
                    raise DataLoadError(f"Failed to load phase data from file: {str(e)}")

            else:
                raise ValidationError(f"Unsupported data type: {type(data)}")

            self.logger.info(f"Loaded phase data with shape {self.phase_data.shape}")
            return self

        except Exception as e:
            if isinstance(e, (DataLoadError, ValidationError)):
                raise
            raise DataLoadError(f"Failed to load phase data: {str(e)}")

    def load_intensity_data(self,
                            data: Union[ImageData, np.ndarray, Path, str],
                            pixel_size: Optional[Tuple[float, float]] = None) -> 'StructureProcessor':
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
                    raise ValidationError("Pixel size must be provided for numpy array input")

                self.intensity_data = ImageData(data, pixel_size)

            elif isinstance(data, (str, Path)):
                # Load from file
                if pixel_size is None and self.phase_data is not None:
                    # Use pixel size from phase data if available
                    pixel_size = self.phase_data.pixel_size

                try:
                    self.intensity_data = ImageData.from_file(data, pixel_size or (1.0, 1.0))
                except Exception as e:
                    raise DataLoadError(f"Failed to load intensity data from file: {str(e)}")

            else:
                raise ValidationError(f"Unsupported data type: {type(data)}")

            self.logger.info(f"Loaded intensity data with shape {self.intensity_data.shape}")
            return self

        except Exception as e:
            if isinstance(e, (DataLoadError, ValidationError)):
                raise
            raise DataLoadError(f"Failed to load intensity data: {str(e)}")

    def detect_structure(self,
                         method: Literal['threshold', 'intensity', 'adaptive'] = 'threshold',
                         **kwargs) -> 'StructureProcessor':
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
            # todo- structure detection whould always be with intensity image - if available
            #   does the method has to be change if intensity_data is none?
            if self.phase_data is None:
                raise ValidationError("Phase data must be loaded before detecting structure")

            # Call detection function
            intensity = self.intensity_data if method == 'intensity' else None

            # todo check how method is being handled
            self.structure = detect_structure(
                self.phase_data,
                method=method,
                intensity_image=intensity,
                **kwargs
            )

            self.logger.info(f"Detected structure using {method} method")
            return self

        except Exception as e:
            if isinstance(e, (ProcessingError, ValidationError)):
                raise
            raise ProcessingError(f"Structure detection failed: {str(e)}")

    def segment_structure(self,
                          method: Literal['grid', 'kmeans', 'watershed'] = 'grid',
                          num_segments: Tuple[int, int] = (2, 2),
                          **kwargs) -> 'StructureProcessor':
        """
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
                raise ValidationError("Structure must be detected before segmentation")

            # Call segmentation function
            # todo check methods watershed and kmeans - what data is the most suitable for this?
            self._segments = segment_structure(
                self.structure,
                method=method,
                num_segments=num_segments,
                **kwargs
            )

            # Update structure with segments
            self.structure.segments = self._segments

            self.logger.info(f"Segmented structure into {len(self._segments)} segments using {method} method")
            return self

        except Exception as e:
            if isinstance(e, (SegmentationError, ValidationError)):
                raise
            raise SegmentationError(f"Structure segmentation failed: {str(e)}")

    def align_with(self,
                   target: StructureProcessor | list[StructureProcessor],
                   method: Literal['edge', 'mask', 'feature'] = 'edge',
                   scaling: Literal['upsampling', 'downsampling', 'none'] = 'downsampling',
                   **kwargs) -> 'StructureProcessor':
        """
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
        """
        # todo: check why target [StructureProcessor] is not a class
        #   what does "none" at scaling mean? - delete it?
        try:
            if self.structure is None:
                raise ValidationError("Structure must be detected before alignment")

            # Get target structure
            if isinstance(target, StructureProcessor):
                if target.structure is None:
                    raise ValidationError("Target processor must have a detected structure")
                target_structure = target.structure
            else:
                target_structure = target

            # Call alignment function
            # todo - return transform - important for intensity image
            #   not structure has to be given as input but the phase-/height data?!
            #   investigate method in respect to self.structure information available
            self.structure = align_structures(
                self.structure,
                target_structure,
                method=method,
                scaling=scaling,
                return_transform=False,
                **kwargs
            )

            self.logger.info(f"Aligned structure using {method} method")
            return self

        except Exception as e:
            if isinstance(e, (AlignmentError, ValidationError)):
                raise
            raise AlignmentError(f"Structure alignment failed: {str(e)}")

    def convert_to_height(self,
                          n_resin: Optional[float] = None,
                          wavelength: Optional[float] = None) -> 'StructureProcessor':
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
                    wavelength=wavelength or self.wavelength
                )
                self.logger.info("Converted structure phase data to height")

            elif self.phase_data is not None:
                # Convert image data
                self.phase_data = convert_phase_to_height(
                    self.phase_data,
                    n_resin=n_resin or self.n_resin,
                    wavelength=wavelength or self.wavelength
                )
                self.logger.info("Converted phase data to height")

            else:
                raise ValidationError("Phase data or structure must be available for conversion")

            return self

        except Exception as e:
            if isinstance(e, (ValidationError, ProcessingError)):
                raise
            raise ProcessingError(f"Phase to height conversion failed: {str(e)}")

    def convert_to_phase(self,
                         n_resin: Optional[float] = None,
                         wavelength: Optional[float] = None) -> 'StructureProcessor':
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
                    wavelength=wavelength or self.wavelength
                )
                self.logger.info("Converted structure height data to phase")

            # todo - converting phase data - phase data from height_2_phase is complete bullshit
            elif self.phase_data is not None:
                # Convert image data
                self.phase_data = convert_height_to_phase(
                    self.phase_data,
                    n_resin=n_resin or self.n_resin,
                    wavelength=wavelength or self.wavelength
                )
                self.logger.info("Converted height data to phase")

            else:
                raise ValidationError("Height data or structure must be available for conversion")

            return self

        except Exception as e:
            if isinstance(e, (ValidationError, ProcessingError)):
                raise
            raise ProcessingError(f"Height to phase conversion failed: {str(e)}")

    # todo - is this really necessary?
    #   structure data is not the right approach to do ?!
    def unwrap_phase(self,
                     method: Literal['path_following', 'quality_guided'] = 'path_following',
                     **kwargs) -> 'StructureProcessor':
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
                self.structure = unwrap_phase(
                    self.structure,
                    method=method,
                    **kwargs
                )
                self.logger.info(f"Unwrapped structure phase data using {method} method")

            elif self.phase_data is not None:
                # Unwrap image data
                self.phase_data = unwrap_phase(
                    self.phase_data,
                    method=method,
                    **kwargs
                )
                self.logger.info(f"Unwrapped phase data using {method} method")

            else:
                raise ValidationError("Phase data or structure must be available for unwrapping")

            return self

        except Exception as e:
            if isinstance(e, (ValidationError, ProcessingError)):
                raise
            raise ProcessingError(f"Phase unwrapping failed: {str(e)}")

    def phase_retrieval(self,
                        target_data: Union[StructureProcessor, Structure, List[float]],
                        convert_to_height: bool = True,
                        **kwargs) -> 'StructureProcessor':
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
                raise ValidationError("Structure must be detected and segmented before phase retrieval")

            if not self.structure.segments:
                raise ValidationError("Structure must be segmented before phase retrieval")

            # Get target data
            if isinstance(target_data, StructureProcessor):
                if target_data.structure is None:
                    raise ValidationError("Target processor must have a detected structure")
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
                n_resin=kwargs.pop('n_resin', self.n_resin),
                wavelength=kwargs.pop('wavelength', self.wavelength),
                convert_to_height=convert_to_height,
                **kwargs
            )

            self.logger.info(f"Performed phase retrieval{' and converted to height' if convert_to_height else ''}")
            return self

        except Exception as e:
            if isinstance(e, (ValidationError, ProcessingError)):
                raise
            raise ProcessingError(f"Phase retrieval failed: {str(e)}")

    def adjust_background(self,
                          method: Literal['subtract', 'plane_fit'] = 'subtract',
                          **kwargs) -> 'StructureProcessor':
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
                raise ValidationError("Structure must be detected before background adjustment")

            # Create background if not explicitly provided
            background = kwargs.pop('background', None)

            if background is None:
                # Extract background from structure mask
                bg_mask = self.structure.data.mask
                bg_data = ma.masked_array(
                    self.structure.data.data,
                    mask=~bg_mask
                )

                background = Background(
                    data=bg_data,
                    pixel_size=self.structure.pixel_size
                )

            # Adjust background
            self.structure = adjust_background(
                self.structure,
                background,
                method=method,
                **kwargs
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
                return plot_image(self.structure.data, **kwargs)
            elif self.phase_data is not None:
                return plot_image(self.phase_data, **kwargs)
            else:
                raise ValidationError("No data available for plotting")

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ProcessingError(f"Image plotting failed: {str(e)}")

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
                raise ValidationError("Structure must be detected and segmented before plotting")

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
                return plot_surface(self.structure.data, **kwargs)
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
                return plot_height_map(self.structure, **kwargs)
            elif self.phase_data is not None:
                return plot_height_map(self.phase_data, **kwargs)
            else:
                raise ValidationError("No data available for plotting")

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ProcessingError(f"Height map plotting failed: {str(e)}")

    def save_data(self,
                  filepath: Union[str, Path],
                  data_type: Literal['phase', 'structure'] = 'phase',
                  format: str = 'npy',
                  **kwargs) -> str:
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

            if data_type == 'phase':
                if self.phase_data is None:
                    raise ValidationError("No phase data available for saving")
                self.phase_data.save(filepath, format=format)

            elif data_type == 'structure':
                if self.structure is None:
                    raise ValidationError("No structure data available for saving")

                # Save as numpy array for now
                if format == 'npy':
                    np.save(filepath, self.structure.data.filled(np.nan))
                elif format == 'csv':
                    np.savetxt(filepath, self.structure.data.filled(np.nan), delimiter=',')
                else:
                    raise ValidationError(f"Unsupported format for structure data: {format}")

                # Save pixel size info
                pixel_info_path = filepath.with_name(f"{filepath.stem}_pixel_info.txt")
                with open(pixel_info_path, 'w') as f:
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
    def from_file(cls,
                  phase_path: Union[str, Path],
                  intensity_path: Optional[Union[str, Path]] = None,
                  pixel_size: Optional[Tuple[float, float]] = None,
                  **kwargs) -> 'StructureProcessor':
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
