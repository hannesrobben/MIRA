"""
Phase retrieval module.

This module provides functions for phase retrieval and conversion between
phase data and height data in holographic microscopy.
"""

import numpy as np
import numpy.ma as ma
from typing import Union, Optional, Tuple, List, Dict, Literal, Any

from ..core.image import ImageData
from ..core.structure import Structure, Segment, Background
from ..utils.exceptions import ProcessingError, PhaseRetrievalError, ValidationError

_LENGTH_UNIT_TO_METERS: Dict[str, float] = {
    "m": 1.0,
    "mm": 1e-3,
    "um": 1e-6,
    "µm": 1e-6,  # Alias, falls jemand das Mu-Zeichen benutzt
    "nm": 1e-9,
}


def _scale_height(
    height_data: Union[ImageData, np.ndarray, Structure], factor: float
) -> Union[ImageData, np.ndarray, Structure]:
    """Skaliert Höhendaten um einen Faktor, unabhängig vom Eingabetyp."""
    if factor == 1.0:
        return height_data

    if isinstance(height_data, Structure):
        scaled = Structure(
            data=ma.masked_array(
                height_data.data.data * factor, mask=height_data.data.mask
            ),
            pixel_size=height_data.pixel_size,
        )
        for segment in height_data.segments:
            scaled.segments.append(
                Segment(
                    data=ma.masked_array(
                        segment.data.data * factor, mask=segment.data.mask
                    ),
                    pixel_size=segment.pixel_size,
                    id=segment.id,
                )
            )
        return scaled

    elif isinstance(height_data, ImageData):
        return ImageData(
            data=height_data.data * factor, pixel_size=height_data.pixel_size
        )

    else:
        return np.asarray(height_data, dtype=float) * factor


def convert_phase_to_height(
    phase_data: Union[ImageData, np.ndarray, Structure],
    n_resin: float = 1.52,
    n_air: float = 1.0,
    wavelength: float = 0.6749e-6,
    **kwargs,
) -> Union[np.ndarray, Structure]:
    """
    Convert phase data to height profile.

    Based on the theoretical investigation by Nguyen et al. (2016)
    Thanh Nguyen, George Nehmetallah, Christopher Raub, Scott Mathews, and Rola Aylo, "Accurate quantitative phase
    digital holographic microscopy with single- and multiple-wavelength telecentric and nontelecentric
    configurations,"
    Appl. Opt. 55, 5666-5683 (2016)
    http://dx.doi.org/10.1364/AO.55.005666

    Args:
        phase_data: Phase data to convert
        n_resin: Refractive index of the resin
        n_air: Refractive index of air
        wavelength: Light wavelength in meters
        **kwargs: Additional parameters (ignored)

    Returns:
        Height profile with same type as input

    Raises:
        PhaseRetrievalError: If conversion fails
        ValidationError: If inputs are invalid
    """
    # todo - include scale for height as well
    try:
        # Validate parameters
        if n_resin <= n_air:
            raise ValidationError("Refractive index of resin must be greater than air")
        if wavelength <= 0:
            raise ValidationError("Wavelength must be positive")

        # Calculate refractive index difference
        delta_n = n_resin - n_air

        # Convert based on input type
        if isinstance(phase_data, Structure):
            # Convert structure data
            height_data = wavelength * phase_data.data.data / (2 * np.pi * delta_n)

            # Create new structure with height data
            height_structure = Structure(
                data=ma.masked_array(height_data, mask=phase_data.data.mask),
                pixel_size=phase_data.pixel_size,
            )

            # Transfer segments if any
            if phase_data.segments:
                for segment in phase_data.segments:
                    # Convert segment data
                    segment_height_data = (
                        wavelength * segment.data.data / (2 * np.pi * delta_n)
                    )

                    # Create new segment with height data
                    height_segment = Segment(
                        data=ma.masked_array(
                            segment_height_data, mask=segment.data.mask
                        ),
                        pixel_size=segment.pixel_size,
                        id=segment.id,
                    )

                    height_structure.segments.append(height_segment)

            return height_structure

        elif isinstance(phase_data, ImageData):
            # Convert image data
            height_data = wavelength * phase_data.data / (2 * np.pi * delta_n)

            # Create new ImageData with height data
            return ImageData(data=height_data, pixel_size=phase_data.pixel_size)

        else:
            # Convert numpy array
            return wavelength * phase_data / (2 * np.pi * delta_n)

    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise PhaseRetrievalError(f"Phase to height conversion failed: {str(e)}")


# def convert_height_to_phase(
#     height_data: Union[ImageData, np.ndarray, Structure],
#     n_resin: float = 1.52,
#     n_air: float = 1.0,
#     wavelength: float = 0.6749e-6,
#     **kwargs,
# ) -> Union[np.ndarray, Structure]:

#     delta_n = n_resin - n_air
#     # ... ab hier bleibt der Rest der Funktion exakt wie vorher
# try:
#     # Validate parameters
#     if n_resin <= n_air:
#         raise ValidationError("Refractive index of resin must be greater than air")
#     if wavelength <= 0:
#         raise ValidationError("Wavelength must be positive")


def convert_height_to_phase(
    height_data: Union[ImageData, np.ndarray, Structure],
    n_resin: float = 1.52,
    n_air: float = 1.0,
    wavelength: float = 0.6749e-6,
    height_unit: str = "m",
    **kwargs,
) -> Union[np.ndarray, Structure]:
    """
    Convert height profile to phase data.

    Args:
        height_data: Height profile to convert
        n_resin: Refractive index of the resin
        n_air: Refractive index of air
        wavelength: Light wavelength in meters
        **kwargs: Additional parameters (ignored)

    Returns:
        Phase data with same type as input

    Raises:
        PhaseRetrievalError: If conversion fails
        ValidationError: If inputs are invalid
    """

    try:
        if n_resin <= n_air:
            raise ValidationError("Refractive index of resin must be greater than air")
        if wavelength <= 0:
            raise ValidationError("Wavelength must be positive")
        if height_unit not in _LENGTH_UNIT_TO_METERS:
            raise ValidationError(
                f"Unknown height_unit: {height_unit}. Supported: {list(_LENGTH_UNIT_TO_METERS)}"
            )

        # height normalised to meter
        height_data = _scale_height(height_data, _LENGTH_UNIT_TO_METERS[height_unit])

        # Calculate refractive index difference
        delta_n = n_resin - n_air

        # Convert based on input type
        if isinstance(height_data, Structure):
            # Convert structure data
            phase_data = 2 * np.pi * height_data.data.data * delta_n / wavelength

            # Create new structure with phase data
            phase_structure = Structure(
                data=ma.masked_array(phase_data, mask=height_data.data.mask),
                pixel_size=height_data.pixel_size,
            )

            # Transfer segments if any
            if height_data.segments:
                for segment in height_data.segments:
                    # Convert segment data
                    segment_phase_data = (
                        2 * np.pi * segment.data.data * delta_n / wavelength
                    )

                    # Create new segment with phase data
                    phase_segment = Segment(
                        data=ma.masked_array(
                            segment_phase_data, mask=segment.data.mask
                        ),
                        pixel_size=segment.pixel_size,
                        id=segment.id,
                    )

                    phase_structure.segments.append(phase_segment)

            return phase_structure

        elif isinstance(height_data, ImageData):
            # Convert image data
            phase_data = 2 * np.pi * height_data.data * delta_n / wavelength

            # Create new ImageData with phase data
            return ImageData(data=phase_data, pixel_size=height_data.pixel_size)

        else:
            # Convert numpy array
            return 2 * np.pi * height_data * delta_n / wavelength

    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise PhaseRetrievalError(f"Height to phase conversion failed: {str(e)}")


def unwrap_phase(
    phase_data: Union[ImageData, np.ndarray, Structure],
    method: Literal["path_following", "quality_guided"] = "path_following",
    **kwargs,
) -> Union[np.ndarray, Structure]:
    """
    Unwrap phase data to remove 2π discontinuities.

    Args:
        phase_data: Wrapped phase data
        method: Unwrapping method
        **kwargs: Additional parameters for the method

    Returns:
        Unwrapped phase data with same type as input

    Raises:
        PhaseRetrievalError: If unwrapping fails
        ValidationError: If inputs are invalid
    """
    try:
        # Try to import unwrapping functions
        try:
            from skimage.restoration import unwrap_phase as skimage_unwrap
        except ImportError:
            raise PhaseRetrievalError("Phase unwrapping requires scikit-image")

        # Handle different input types
        if isinstance(phase_data, Structure):
            # Get data and mask
            wrapped_data = phase_data.data.filled(0)
            mask = phase_data.data.mask

            # Apply unwrapping
            if method == "path_following":
                unwrapped_data = skimage_unwrap(wrapped_data, **kwargs)
            elif method == "quality_guided":
                # This method is just a placeholder - scikit-image doesn't have it
                # In a real implementation, you might have multiple algorithms
                raise ValidationError("Quality-guided unwrapping not implemented")
            else:
                raise ValidationError(f"Unknown unwrapping method: {method}")

            # Create new structure with unwrapped data
            unwrapped_structure = Structure(
                data=ma.masked_array(unwrapped_data, mask=mask),
                pixel_size=phase_data.pixel_size,
            )

            # Transfer segments if any
            if phase_data.segments:
                for segment in phase_data.segments:
                    # Unwrap segment data
                    wrapped_segment = segment.data.filled(0)
                    unwrapped_segment = skimage_unwrap(wrapped_segment, **kwargs)

                    # Create new segment with unwrapped data
                    new_segment = Segment(
                        data=ma.masked_array(unwrapped_segment, mask=segment.data.mask),
                        pixel_size=segment.pixel_size,
                        id=segment.id,
                    )

                    unwrapped_structure.segments.append(new_segment)

            return unwrapped_structure

        elif isinstance(phase_data, ImageData):
            # Unwrap image data
            unwrapped_data = skimage_unwrap(phase_data.data, **kwargs)

            # Create new ImageData with unwrapped data
            return ImageData(data=unwrapped_data, pixel_size=phase_data.pixel_size)

        else:
            # Unwrap numpy array
            return skimage_unwrap(phase_data, **kwargs)

    except Exception as e:
        if isinstance(e, (PhaseRetrievalError, ValidationError)):
            raise
        raise PhaseRetrievalError(f"Phase unwrapping failed: {str(e)}")


def full_phase_retrieval(
    phase_data: Structure,
    target_data: Union[Structure, List[float]],
    target_data_type: Literal["phase", "height"] = "phase",
    background: Optional[Background] = None,
    n_resin: float = 1.52,
    n_air: float = 1.0,
    wavelength: float = 0.6749e-6,
    convert_to_height: bool = True,
    **kwargs,
) -> Structure:
    """
    Perform full phase retrieval and correction.

    This function adjusts the phase of each segment to match target values,
    then optionally converts to height.

    Args:
        phase_data: Structure with phase data to correct
        target_data: Target values (Structure or list of values for each segment)
        target_data_type: Whether target_data contains 'phase' or 'height' values.
            If 'height', target_data is converted to phase before retrieval
            using n_resin, n_air and wavelength.
        background: Optional background for correction
        n_resin: Refractive index of the resin
        n_air: Refractive index of air
        wavelength: Light wavelength in meters
        convert_to_height: Whether to convert result to height
        **kwargs: Additional parameters (ignored)

    Returns:
        Structure with corrected data

    Raises:
        PhaseRetrievalError: If retrieval fails
        ValidationError: If inputs are invalid
    """
    try:
        # Check if structure has segments
        if not phase_data.segments:
            raise ValidationError("Structure must have segments for phase retrieval")

        # Validate and apply target_data_type
        if target_data_type not in ("phase", "height"):
            raise ValidationError(f"Unknown target_data_type: {target_data_type}")

        if target_data_type == "height":
            # Convert target data to phase before doing anything else,
            # so the rest of the function can keep assuming phase values.

            # todo - change height_unit to a variable accessible from outside
            if isinstance(target_data, Structure):
                target_data = convert_height_to_phase(
                    target_data,
                    n_resin=n_resin,
                    n_air=n_air,
                    wavelength=wavelength,
                    height_unit="um",
                )
            else:
                target_data = convert_height_to_phase(
                    np.asarray(target_data, dtype=float),
                    n_resin=n_resin,
                    n_air=n_air,
                    wavelength=wavelength,
                    height_unit="um",
                )

        # Handle target data
        if isinstance(target_data, Structure):
            # Check if target has segments
            if not target_data.segments:
                raise ValidationError("Target structure must have segments")

            # Check if segment counts match
            if len(target_data.segments) != len(phase_data.segments):
                raise ValidationError(
                    f"Segment count mismatch: {len(phase_data.segments)} vs {len(target_data.segments)}"
                )

            # Extract target values
            target_values = [segment.data.mean() for segment in target_data.segments]
        else:
            # Use provided list/array of target values
            target_values = target_data

            # Check if value count matches segment count
            if len(target_values) != len(phase_data.segments):
                raise ValidationError(
                    f"Value count mismatch: {len(phase_data.segments)} vs {len(target_values)}"
                )

        # Adjust background if provided
        if background is not None:
            adjusted_data = adjust_background(phase_data, background)
        else:
            adjusted_data = phase_data

        # Create a copy to modify
        result = Structure(
            data=adjusted_data.data.copy(), pixel_size=adjusted_data.pixel_size
        )

        # Copy segments
        for segment in adjusted_data.segments:
            result.segments.append(
                Segment(
                    data=segment.data.copy(),
                    pixel_size=segment.pixel_size,
                    id=segment.id,
                )
            )

        # Process each segment
        for i, segment in enumerate(result.segments):
            # Get current and target values
            current_value = segment.data.mean()
            target_value = target_values[i]

            # Calculate phase difference (accounting for 2π wrapping)
            phase_diff = (
                target_value
                + ((current_value - target_value + np.pi) % (2 * np.pi))
                - np.pi
            )

            # Apply correction
            segment_mask = ~segment.data.mask
            result.data.data[segment_mask] += phase_diff - current_value

            # Update segment data
            segment.data.data[~segment.data.mask] += phase_diff - current_value

        # Convert to height if requested
        if convert_to_height:
            result = convert_phase_to_height(
                result, n_resin=n_resin, wavelength=wavelength
            )

        return result

    except Exception as e:
        if isinstance(e, (PhaseRetrievalError, ValidationError)):
            raise
        raise PhaseRetrievalError(f"Full phase retrieval failed: {str(e)}")


def full_phase_retrieval_old(
    phase_data: Structure,
    target_data: Union[Structure, List[float]],
    background: Optional[Background] = None,
    n_resin: float = 1.52,
    wavelength: float = 0.6749e-6,
    convert_to_height: bool = True,
    **kwargs,
) -> Structure:
    """
    Perform full phase retrieval and correction.

    This function adjusts the phase of each segment to match target values,
    then optionally converts to height.

    Args:
        phase_data: Structure with phase data to correct
        target_data: Target phase values (Structure or list of values for each segment)
        background: Optional background for correction
        n_resin: Refractive index of the resin
        wavelength: Light wavelength in meters
        convert_to_height: Whether to convert result to height
        **kwargs: Additional parameters (ignored)

    Returns:
        Structure with corrected data

    Raises:
        PhaseRetrievalError: If retrieval fails
        ValidationError: If inputs are invalid
    """
    try:
        # Check if structure has segments
        if not phase_data.segments:
            raise ValidationError("Structure must have segments for phase retrieval")

        # Handle target data
        if isinstance(target_data, Structure):
            # Check if target has segments
            if not target_data.segments:
                raise ValidationError("Target structure must have segments")

            # Check if segment counts match
            if len(target_data.segments) != len(phase_data.segments):
                raise ValidationError(
                    f"Segment count mismatch: {len(phase_data.segments)} vs {len(target_data.segments)}"
                )

            # Extract target values
            target_values = [segment.data.mean() for segment in target_data.segments]
        else:
            # Use provided list of target values
            target_values = target_data

            # Check if value count matches segment count
            if len(target_values) != len(phase_data.segments):
                raise ValidationError(
                    f"Value count mismatch: {len(phase_data.segments)} vs {len(target_values)}"
                )

        # Adjust background if provided
        if background is not None:
            adjusted_data = adjust_background(phase_data, background)
        else:
            adjusted_data = phase_data

        # Create a copy to modify
        result = Structure(
            data=adjusted_data.data.copy(), pixel_size=adjusted_data.pixel_size
        )

        # Copy segments
        for segment in adjusted_data.segments:
            result.segments.append(
                Segment(
                    data=segment.data.copy(),
                    pixel_size=segment.pixel_size,
                    id=segment.id,
                )
            )

        # Process each segment
        for i, segment in enumerate(result.segments):
            # Get current and target values
            current_value = segment.data.mean()
            target_value = target_values[i]

            # Calculate phase difference (accounting for 2π wrapping)
            phase_diff = (
                target_value
                + ((current_value - target_value + np.pi) % (2 * np.pi))
                - np.pi
            )

            # Apply correction
            segment_mask = ~segment.data.mask
            result.data.data[segment_mask] += phase_diff - current_value

            # Update segment data
            segment.data.data[~segment.data.mask] += phase_diff - current_value

        # Convert to height if requested
        if convert_to_height:
            result = convert_phase_to_height(
                result, n_resin=n_resin, wavelength=wavelength
            )

        return result

    except Exception as e:
        if isinstance(e, (PhaseRetrievalError, ValidationError)):
            raise
        raise PhaseRetrievalError(f"Full phase retrieval failed: {str(e)}")


def adjust_background(
    structure: Structure,
    background: Background,
    method: Literal["subtract", "plane_fit"] = "subtract",
    **kwargs,
) -> Structure:
    """
    Adjust structure data by subtracting or fitting background.

    Args:
        structure: Structure to adjust
        background: Background data
        method: Adjustment method
        **kwargs: Additional parameters for the method

    Returns:
        Adjusted structure

    Raises:
        PhaseRetrievalError: If adjustment fails
        ValidationError: If inputs are invalid
    """
    try:
        # Create a copy of the structure
        result = Structure(data=structure.data.copy(), pixel_size=structure.pixel_size)

        # Copy segments if any
        if structure.segments:
            for segment in structure.segments:
                result.segments.append(
                    Segment(
                        data=segment.data.copy(),
                        pixel_size=segment.pixel_size,
                        id=segment.id,
                    )
                )

        # Apply adjustment method
        if method == "subtract":
            # Get background mean value
            bg_level = background.mean_value

            # Subtract background level
            result.data = ma.masked_array(
                result.data.data - bg_level, mask=result.data.mask
            )

            # Adjust segments if any
            for segment in result.segments:
                segment.data = ma.masked_array(
                    segment.data.data - bg_level, mask=segment.data.mask
                )

        elif method == "plane_fit":
            # Fit a plane to background data
            try:
                # Get background points
                bg_mask = ~background.data.mask
                bg_points = np.where(bg_mask)
                bg_values = background.data.data[bg_mask]

                # Fit plane (z = ax + by + c)
                from sklearn.linear_model import LinearRegression

                X = np.column_stack([bg_points[1], bg_points[0]])  # x, y coordinates
                model = LinearRegression().fit(X, bg_values)

                # Create a grid of coordinates
                y, x = np.indices(structure.data.shape)
                X_grid = np.column_stack([x.ravel(), y.ravel()])

                # Predict background plane
                bg_plane = model.predict(X_grid).reshape(structure.data.shape)

                # Subtract fitted plane
                result.data = ma.masked_array(
                    result.data.data - bg_plane, mask=result.data.mask
                )

                # Adjust segments if any
                for segment in result.segments:
                    segment_mask = ~segment.data.mask
                    segment.data.data[segment_mask] -= bg_plane[segment_mask]

            except ImportError:
                raise PhaseRetrievalError("Plane fitting requires scikit-learn")

        else:
            raise ValidationError(f"Unknown background adjustment method: {method}")

        return result

    except Exception as e:
        if isinstance(e, (PhaseRetrievalError, ValidationError)):
            raise
        raise PhaseRetrievalError(f"Background adjustment failed: {str(e)}")
