"""
Phase retrieval module.

This module provides functions for phase retrieval and conversion between
phase data and height data in holographic microscopy.

Units contract (applies to all functions in this module)
----------------------------------------------------------
- Phase data is always in **radians**, unwrapped (no 2π ambiguity). This is
  never configurable - a phase value is dimensionless by definition, so there
  is nothing to convert.
- Height data has an explicit, configurable unit via the ``height_unit``
  parameter (default: ``"m"``, matching ``wavelength`` which is always given
  in meters). Supported units are the keys of ``_LENGTH_UNIT_TO_METERS``.
- Every function that consumes or produces height data exposes the same
  ``height_unit`` parameter with the same default, so call sites that never
  touch it keep behaving exactly as before (meters in, meters out).
"""

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy import ma

from ..core.image import ImageData
from ..core.structure import Background, Segment, Structure
from ..utils.exceptions import PhaseRetrievalError, ValidationError

_LENGTH_UNIT_TO_METERS: dict[str, float] = {
    "m": 1.0,
    "mm": 1e-3,
    "um": 1e-6,
    "µm": 1e-6,  # Alias, in case someone uses the mu character
    "nm": 1e-9,
}


def _validate_optical_params(n_resin: float, n_air: float, wavelength: float) -> None:
    """Shared validation for every function that works with refractive indices/wavelength."""
    if n_resin <= n_air:
        raise ValidationError("Refractive index of resin must be greater than air")
    if wavelength <= 0:
        raise ValidationError("Wavelength must be positive")


def _validate_height_unit(height_unit: str) -> float:
    """Validate height_unit and return its scaling factor (-> meters)."""
    if height_unit not in _LENGTH_UNIT_TO_METERS:
        raise ValidationError(
            f"Unknown height_unit: {height_unit}. Supported: {list(_LENGTH_UNIT_TO_METERS)}"
        )
    return _LENGTH_UNIT_TO_METERS[height_unit]


def _copy_structure_metadata(source: Structure, target: Structure) -> None:
    """
    Copy `info` and `statistics` from `source` onto `target`, in place.

    Every function in this module that derives a new `Structure` from an
    existing one (phase<->height conversion, unwrapping, background
    adjustment, ...) only ever changes the *data* - identifying metadata
    (`doe_id`, `print_id`, ...) and previously computed results
    (`DeltaNResult`, ...) describe the same physical structure regardless,
    and callers reasonably expect them to survive.

    Before this helper existed, every one of those call sites built its
    result via `Structure(data=..., pixel_size=...)` directly, which left
    `info`/`statistics` at their dataclass defaults (`{}`) - silently
    dropping anything `set_info()`/`add_statistic()` had attached
    beforehand. This is the single place that copy now happens, so future
    derivation sites only need to remember to call this helper once,
    instead of re-deriving the copy logic (and the same bug) themselves.

    Plain `dict(...)` (shallow copy) is used deliberately: `target` gets
    its own top-level `info`/`statistics` dict, so later calling
    `target.set_info(...)` or `target.add_statistic(...)` can never
    accidentally mutate `source`'s metadata too.
    """
    target.info = dict(source.info)
    target.statistics = dict(source.statistics)


def _apply_elementwise(
    data: ImageData | np.ndarray | Structure,
    func: Callable[[np.ndarray], np.ndarray],
) -> ImageData | np.ndarray | Structure:
    """
    Apply `func` elementwise to the raw data, regardless of input type, and
    rebuild the same type (incl. mask, pixel_size, segments, info,
    statistics) around the result.

    This is the single place that distinguishes the three supported input
    types (Structure / ImageData / np.ndarray) - both scaling
    (_scale_height) and the actual phase<->height conversions use this
    function, so new special cases only ever need to be handled once.
    """
    if isinstance(data, Structure):
        new_data = func(data.data.data)
        result = Structure(
            data=ma.masked_array(new_data, mask=data.data.mask),
            pixel_size=data.pixel_size,
        )
        for segment in data.segments:
            result.segments.append(
                Segment(
                    data=ma.masked_array(
                        func(segment.data.data), mask=segment.data.mask
                    ),
                    pixel_size=segment.pixel_size,
                    id=segment.id,
                )
            )
        _copy_structure_metadata(source=data, target=result)
        return result

    elif isinstance(data, ImageData):
        return ImageData(data=func(data.data), pixel_size=data.pixel_size)

    else:
        return func(np.asarray(data, dtype=float))


def _scale_height(
    height_data: ImageData | np.ndarray | Structure, factor: float
) -> ImageData | np.ndarray | Structure:
    """Scale height data by a factor, regardless of input type."""
    if factor == 1.0:
        return height_data
    return _apply_elementwise(height_data, lambda arr: arr * factor)


def convert_height_unit(
    height_data: ImageData | np.ndarray | Structure,
    from_unit: str,
    to_unit: str = "m",
) -> ImageData | np.ndarray | Structure:
    """
    Rescale height data from `from_unit` to `to_unit`, preserving type,
    mask, pixel_size, segments, info, and statistics.

    Every function in this module assumes height data is already in
    meters unless a `height_unit` argument says otherwise - this is the
    general-purpose tool for the one situation that contract does not
    cover by itself: raw values loaded straight from an instrument (e.g.
    a Sensofar profilometer snapshot) that are not in meters to begin
    with. Call this once, right after loading, rather than relying on a
    downstream comparison factor to compensate - a compensating factor
    only fixes the one place it was tuned for and produces a new,
    equal-and-opposite mismatch anywhere else the same data is used
    (e.g. any of the `mira.analysis` modules, which all assume meters
    with no comparison factor to correct for it).

    Args:
        height_data: Height data to rescale.
        from_unit:   Unit `height_data` is currently expressed in.
        to_unit:     Unit to convert to. Defaults to "m", the unit every
                     other function in this module assumes.

    Returns:
        Height data of the same type as the input, expressed in `to_unit`.
        Returns the *same object* unchanged if `from_unit == to_unit`
        (no unnecessary copy).

    Raises:
        ValidationError: If either unit is not one of "m"/"mm"/"um"/"µm"/"nm".
    """
    from_factor = _validate_height_unit(from_unit)
    to_factor = _validate_height_unit(to_unit)
    return _scale_height(height_data, from_factor / to_factor)


def convert_phase_to_height(
    phase_data: ImageData | np.ndarray | Structure,
    n_resin: float = 1.52,
    n_air: float = 1.0,
    wavelength: float = 0.6749e-6,
    height_unit: str = "m",
    **kwargs,
) -> np.ndarray | Structure:
    """
    Convert phase data to height profile.

    Based on the theoretical investigation by Nguyen et al. (2016)
    Thanh Nguyen, George Nehmetallah, Christopher Raub, Scott Mathews, and Rola Aylo, "Accurate quantitative phase
    digital holographic microscopy with single- and multiple-wavelength telecentric and nontelecentric
    configurations,"
    Appl. Opt. 55, 5666-5683 (2016)
    http://dx.doi.org/10.1364/AO.55.005666

    Args:
        phase_data: Phase data to convert. Must be unwrapped phase in radians.
        n_resin: Refractive index of the resin
        n_air: Refractive index of air
        wavelength: Light wavelength in meters
        height_unit: Unit of the *returned* height data. Defaults to "m",
            i.e. the historical behaviour (raw wavelength/(2*pi*delta_n) result,
            no rescaling) is preserved when this argument is omitted.
            Supported: "m", "mm", "um"/"µm", "nm".
        **kwargs: Additional parameters (ignored)

    Returns:
        Height profile with same type as input, expressed in `height_unit`.

    Raises:
        PhaseRetrievalError: If conversion fails
        ValidationError: If inputs are invalid
    """
    try:
        _validate_optical_params(n_resin, n_air, wavelength)
        unit_factor = _validate_height_unit(height_unit)

        # Calculate refractive index difference
        delta_n = n_resin - n_air

        # Core formula always yields meters (wavelength is given in meters).
        # Converting to the requested output unit means dividing by the
        # unit's meter-factor (e.g. um -> m factor is 1e-6, so to express a
        # value that's already in meters *as* um we divide by 1e-6).
        height_data = _apply_elementwise(
            phase_data, lambda arr: wavelength * arr / (2 * np.pi * delta_n)
        )

        if unit_factor != 1.0:
            height_data = _scale_height(height_data, 1.0 / unit_factor)

        return height_data

    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise PhaseRetrievalError(f"Phase to height conversion failed: {e!s}")


def convert_height_to_phase(
    height_data: ImageData | np.ndarray | Structure,
    n_resin: float = 1.52,
    n_air: float = 1.0,
    wavelength: float = 0.6749e-6,
    height_unit: str = "m",
    **kwargs,
) -> np.ndarray | Structure:
    """
    Convert height profile to phase data.

    Args:
        height_data: Height profile to convert, expressed in `height_unit`.
        n_resin: Refractive index of the resin
        n_air: Refractive index of air
        wavelength: Light wavelength in meters
        height_unit: Unit that `height_data` is given in. Defaults to "m",
            i.e. the historical behaviour (no rescaling) is preserved when
            this argument is omitted. Supported: "m", "mm", "um"/"µm", "nm".
        **kwargs: Additional parameters (ignored)

    Returns:
        Phase data (radians) with same type as input

    Raises:
        PhaseRetrievalError: If conversion fails
        ValidationError: If inputs are invalid
    """
    try:
        _validate_optical_params(n_resin, n_air, wavelength)
        unit_factor = _validate_height_unit(height_unit)

        # Normalise input height to meters first, so the core formula below
        # always operates on meters regardless of what unit the caller used.
        height_data_m = _scale_height(height_data, unit_factor)

        delta_n = n_resin - n_air

        phase_data = _apply_elementwise(
            height_data_m, lambda arr: 2 * np.pi * arr * delta_n / wavelength
        )

        return phase_data

    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise PhaseRetrievalError(f"Height to phase conversion failed: {e!s}")


def unwrap_phase(
    phase_data: ImageData | np.ndarray | Structure,
    method: Literal["path_following", "quality_guided"] = "path_following",
    **kwargs,
) -> np.ndarray | Structure:
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

            _copy_structure_metadata(source=phase_data, target=unwrapped_structure)
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
        raise PhaseRetrievalError(f"Phase unwrapping failed: {e!s}")


def full_phase_retrieval(
    phase_data: Structure,
    target_data: Structure | list[float],
    target_data_type: Literal["phase", "height"] = "phase",
    target_height_unit: str | None = None,  # previously: str = "m"
    background: Background | None = None,
    n_resin: float = 1.52,
    n_air: float = 1.0,
    wavelength: float = 0.6749e-6,
    convert_to_height: bool = True,
    height_unit: str = "m",
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
            using n_resin, n_air, wavelength and `target_height_unit`.
        target_height_unit: Unit that `target_data` is given in, REQUIRED
            (no default) when target_data_type == "height". Must be one of
            "m", "mm", "um"/"µm", "nm". There is intentionally no fallback
            value: silently assuming the wrong unit produces a phase result
            that is off by a large, easy-to-miss factor (e.g. 1e6 for a
            µm/m mix-up) without raising any error.
        background: Optional background for correction
        n_resin: Refractive index of the resin
        n_air: Refractive index of air
        wavelength: Light wavelength in meters
        convert_to_height: Whether to convert result to height
        height_unit: Unit of the returned height data, only used when
            convert_to_height is True. Defaults to "m".
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
            # No silent unit default here: unlike phase (unitless by
            # definition), height data is only meaningful together with an
            # explicit unit. Falling back to a default (e.g. "m") previously
            # let differently-scaled data (e.g. values actually in µm) be
            # silently misinterpreted as meters, producing phase values off
            # by a factor of 1e6 with no error anywhere in the pipeline.
            if target_height_unit is None:
                raise ValidationError(
                    "target_height_unit must be given explicitly when "
                    "target_data_type='height' (e.g. 'um', 'mm', 'm'). "
                    "There is no safe default, since silently assuming the "
                    "wrong unit produces plausible-looking but wrong phase "
                    "values with no error."
                )
            # Convert target data to phase before doing anything else,
            # so the rest of the function can keep assuming phase values.
            if isinstance(target_data, Structure):
                target_data = convert_height_to_phase(
                    target_data,
                    n_resin=n_resin,
                    n_air=n_air,
                    wavelength=wavelength,
                    height_unit=target_height_unit,
                )
            else:
                target_data = convert_height_to_phase(
                    np.asarray(target_data, dtype=float),
                    n_resin=n_resin,
                    n_air=n_air,
                    wavelength=wavelength,
                    height_unit=target_height_unit,
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

        _copy_structure_metadata(source=adjusted_data, target=result)

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
                result,
                n_resin=n_resin,
                n_air=n_air,
                wavelength=wavelength,
                height_unit=height_unit,
            )

        return result

    except Exception as e:
        if isinstance(e, (PhaseRetrievalError, ValidationError)):
            raise
        raise PhaseRetrievalError(f"Full phase retrieval failed: {e!s}")


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

        _copy_structure_metadata(source=structure, target=result)

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
        raise PhaseRetrievalError(f"Background adjustment failed: {e!s}")
