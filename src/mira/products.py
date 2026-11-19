"""Product data model for DHM products: ``TileProduct`` and ``LayerProduct``.

Decision M2 (2026-11-18): immutable, backend-neutral (NumPy / CuPy) types for
the products that the DHMConnect service delivers.

``TileProduct``
    One capture (one stage position): the compensated complex field
    (complex64) on its grid, the actual stage position, z with its
    propagation index, wavelength, quality, ids, material snapshot and
    provenance. On the wire it is the contract ``ProductSet`` 1.0, which wraps
    an OffAxisHolo ``ReconstructionResult`` unchanged.
``LayerProduct``
    The stitched / fused mosaic of one structure and layer: unwrapped phase
    and ``intensity_rel`` (float32), an optional complex field, a validity
    mask, the grid and one record per tile (transform, piston / tilt, anchor,
    residual, discarded border band, quality). Contract ``LayerProduct`` 1.0
    (= ``GroupProduct``).

Conventions (contracts 1.0, ``DHMConnect/docs/CONTRACTS.md``): units in the
names, image axis 0 = y, in-plane vectors and pixel sizes ``(y, x)`` in µm,
stage positions as ``{"x", "y", "z"}`` in µm, ``grid.origin_um`` is the stage
position of the centre of pixel (0, 0). The conversion to the ``(x, y)``
metres of :class:`mira.StructureProcessor` happens in exactly one place,
:func:`to_structure_processor`.

Messages are a JSON-able header plus named binary buffers, the same shape as
``offaxisholo.realtime.ReconstructionResult.to_message``: ``header["buffers"]``
describes every buffer (name, dtype, shape, byte order) in order, and the
buffer list holds ``(name, ndarray)`` pairs. Received buffers may also be raw
bytes. Nothing is pickled.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any

import numpy as np

from .utils.exceptions import ValidationError

__all__ = [
    "PRODUCT_SET_SCHEMA_VERSION",
    "LAYER_PRODUCT_SCHEMA_VERSION",
    "Grid",
    "TileRecord",
    "TileProduct",
    "LayerProduct",
    "to_structure_processor",
]

PRODUCT_SET_SCHEMA_VERSION = "1.0"
LAYER_PRODUCT_SCHEMA_VERSION = "1.0"
_RECONSTRUCTION_SCHEMA_VERSION = "1.0"

_WIRE_DTYPES = {
    "phase_rad": np.dtype("<f4"),
    "intensity_rel": np.dtype("<f4"),
    "field": np.dtype("<c8"),
    "valid": np.dtype("u1"),
}
_DTYPE_NAMES = {
    np.dtype("<f4"): "float32",
    np.dtype("<c8"): "complex64",
    np.dtype("u1"): "uint8",
}


# ----------------------------------------------------------------- helpers
def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _major(version: Any) -> str:
    return str(version).split(".", 1)[0]


def _to_jsonable(value: Any) -> Any:
    """Return ``value`` with plain JSON types only; NaN / inf become None."""
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Value of type {type(value).__name__} is not JSON-able.")


def _freeze(value: Any) -> Any:
    """Return a read-only view of nested dicts / lists (for immutability)."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    """Inverse of :func:`_freeze`: plain dicts and lists."""
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


def _xp(array: Any):
    """Return the array module (NumPy or CuPy) of ``array``."""
    module = type(array).__module__.split(".", 1)[0]
    if module == "cupy":  # pragma: no cover - no CuPy in the test environment
        import cupy

        return cupy
    return np


def _to_numpy(array: Any) -> np.ndarray:
    """Copy a CuPy array to the host; NumPy arrays pass through."""
    if isinstance(array, np.ndarray):
        return array
    if hasattr(array, "get"):  # pragma: no cover - CuPy only
        return array.get()
    return np.asarray(array)


def _as_array(array: Any, dtype: np.dtype, name: str) -> Any:
    """Copy to ``dtype`` on the array's own backend and make it read-only."""
    xp = np if isinstance(array, (list, tuple)) else _xp(array)
    out = xp.asarray(array).astype(dtype, copy=True)
    if out.ndim != 2:
        raise ValidationError(f"{name} must be 2-D (y, x); got shape {out.shape}.")
    if isinstance(out, np.ndarray):
        out.flags.writeable = False
    return out


def _pair(value: Any, name: str) -> tuple[float, float]:
    try:
        y, x = (float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a (y, x) pair; got {value!r}.") from exc
    return y, x


def _stage(value: Any, name: str) -> Mapping[str, float] | None:
    if value is None:
        return None
    try:
        return MappingProxyType({k: float(value[k]) for k in ("x", "y", "z")})
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(
            f"{name} must be a stage position {{x, y, z}} in µm; got {value!r}."
        ) from exc


def _descriptor(name: str, data: np.ndarray) -> dict:
    return {
        "name": name,
        "dtype": _DTYPE_NAMES[data.dtype],
        "shape": list(data.shape),
        "byte_order": "little",
    }


def _wire(name: str, array: Any) -> np.ndarray:
    data = _to_numpy(array)
    if name == "valid":
        data = data.astype(bool)
    return np.ascontiguousarray(data, dtype=_WIRE_DTYPES[name])


def _decode_buffers(descriptors: Sequence[Mapping], buffers: Sequence) -> dict:
    """Turn received buffers into arrays, checked against their descriptors."""
    buffers = list(buffers)
    if len(buffers) != len(descriptors):
        raise ValidationError(
            f"{len(descriptors)} buffers described, {len(buffers)} received."
        )
    arrays = {}
    for descriptor, payload in zip(descriptors, buffers, strict=True):
        name = descriptor["name"]
        if isinstance(payload, tuple):
            given, payload = payload
            if given != name:
                raise ValidationError(
                    f"Buffer {given!r} received where {name!r} expected."
                )
        if name not in _WIRE_DTYPES:
            # Unknown buffers are ignored (forward compatibility).
            continue
        if descriptor.get("byte_order", "little") != "little":
            raise ValidationError(f"Buffer {name!r}: only little-endian is supported.")
        dtype = np.dtype(descriptor["dtype"]).newbyteorder("<")
        if dtype != _WIRE_DTYPES[name].newbyteorder("<"):
            raise ValidationError(f"Buffer {name!r} has dtype {descriptor['dtype']}.")
        shape = tuple(int(v) for v in descriptor["shape"])
        if isinstance(payload, np.ndarray):
            array = payload
        else:
            array = np.frombuffer(payload, dtype=dtype)
            if array.size != math.prod(shape):
                raise ValidationError(
                    f"Buffer {name!r}: {array.size} values for shape {shape}."
                )
            array = array.reshape(shape)
        if tuple(array.shape) != shape:
            raise ValidationError(f"Buffer {name!r}: shape {array.shape} != {shape}.")
        arrays[name] = array
    return arrays


def _check_message(header: Mapping, message_type: str, supported: str) -> None:
    if header.get("message_type") != message_type:
        raise ValidationError(
            f"Not a {message_type} message: {header.get('message_type')!r}."
        )
    version = header.get("schema_version", "")
    if _major(version) != _major(supported):
        raise ValidationError(
            f"Incompatible {message_type} schema_version {version!r} "
            f"(supported: {_major(supported)}.x)."
        )


def _field_from(phase: Any, intensity: Any) -> Any:
    """Complex field from wrapped phase and ``intensity_rel`` = |field|²."""
    xp = _xp(phase)
    amplitude = xp.sqrt(xp.clip(xp.asarray(intensity, dtype=xp.float32), 0, None))
    return (amplitude * xp.exp(1j * xp.asarray(phase, dtype=xp.float32))).astype(
        xp.complex64
    )


# -------------------------------------------------------------------- grid
@dataclass(frozen=True)
class Grid:
    """Pixel grid of a product in stage coordinates.

    Parameters
    ----------
    origin_um : tuple of float
        Stage position ``(y, x)`` in µm of the centre of pixel (0, 0).
    pixel_um : tuple of float
        Pixel size ``(y, x)`` in µm in the object plane; both positive.
    shape : tuple of int
        Number of pixels ``(ny, nx)``.

    Notes
    -----
    Until the stage-to-image calibration (MI-11, nanofactorysystem T104)
    exists the grid is axis-aligned with the stage: rotation 0 and the nominal
    pixel size (decision G10 SA2).
    """

    origin_um: tuple[float, float]
    pixel_um: tuple[float, float]
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin_um", _pair(self.origin_um, "origin_um"))
        pixel = _pair(self.pixel_um, "pixel_um")
        if not all(math.isfinite(p) and p > 0 for p in pixel):
            raise ValidationError(f"pixel_um must be positive and finite; got {pixel}.")
        object.__setattr__(self, "pixel_um", pixel)
        try:
            ny, nx = (int(v) for v in self.shape)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"shape must be (ny, nx); got {self.shape!r}."
            ) from exc
        if ny < 1 or nx < 1:
            raise ValidationError(f"shape must be positive; got {(ny, nx)}.")
        object.__setattr__(self, "shape", (ny, nx))

    @classmethod
    def centred(
        cls,
        centre_um: Mapping[str, float],
        pixel_um: Sequence[float],
        shape: Sequence[int],
    ) -> Grid:
        """Grid whose image centre lies at a stage position.

        Parameters
        ----------
        centre_um : mapping
            Stage position ``{"x", "y", ...}`` in µm of the image centre (for a
            tile: the actual stage position of the capture).
        pixel_um : sequence of float
            Pixel size ``(y, x)`` in µm.
        shape : sequence of int
            ``(ny, nx)``.

        Returns
        -------
        Grid
        """
        py, px = _pair(pixel_um, "pixel_um")
        ny, nx = (int(v) for v in shape)
        origin = (
            float(centre_um["y"]) - 0.5 * (ny - 1) * py,
            float(centre_um["x"]) - 0.5 * (nx - 1) * px,
        )
        return cls(origin_um=origin, pixel_um=(py, px), shape=(ny, nx))

    @property
    def extent_um(self) -> tuple[float, float]:
        """Size ``(y, x)`` in µm covered by the pixels."""
        return (self.shape[0] * self.pixel_um[0], self.shape[1] * self.pixel_um[1])

    @property
    def centre_um(self) -> tuple[float, float]:
        """Stage position ``(y, x)`` in µm of the image centre."""
        return (
            self.origin_um[0] + 0.5 * (self.shape[0] - 1) * self.pixel_um[0],
            self.origin_um[1] + 0.5 * (self.shape[1] - 1) * self.pixel_um[1],
        )

    def to_dict(self) -> dict:
        """Return the contract ``grid`` object."""
        return {
            "origin_um": list(self.origin_um),
            "pixel_um": list(self.pixel_um),
            "shape": list(self.shape),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> Grid:
        """Build a grid from the contract ``grid`` object."""
        try:
            return cls(data["origin_um"], data["pixel_um"], data["shape"])
        except KeyError as exc:
            raise ValidationError(f"grid lacks {exc.args[0]!r}.") from exc


# ------------------------------------------------------------ tile records
_TILE_RECORD_KEYS = (
    "product_id",
    "tile_index",
    "planned_um",
    "actual_um",
    "registration",
    "piston_rad",
    "tilt_rad_per_um",
    "anchor",
    "residual_rad",
    "border_band_um",
    "reference_source",
    "reference_fraction",
    "unwrap",
    "flagged",
)


@dataclass(frozen=True)
class TileRecord:
    """Stitching record of one member tile of a :class:`LayerProduct`.

    Parameters
    ----------
    product_id : str
        Id of the member ``ProductSet`` (tile product in the DHM-PC archive).
    tile_index : int
        Index of the tile in the planned position list.
    planned_um, actual_um : mapping or None
        Planned and actual stage position ``{x, y, z}`` in µm.
    registration : mapping
        Registration result, e.g. ``shift_um`` ``(y, x)``, ``method``,
        ``residual_um``.
    piston_rad : float or None
        Piston removed from the tile.
    tilt_rad_per_um : tuple of float or None
        Linear phase ``(y, x)`` removed from the tile, rad per µm.
    anchor : str or None
        How the tile is anchored (e.g. ``"ring_substrate"``, ``"overlap"``).
    residual_rad : float or None
        Residual of the global adjustment.
    border_band_um : float or None
        Width of the discarded border band.
    reference_source : str or None
        D13 key: ``mask_a_priori``, ``mask_a_posteriori``, ``border_ring`` or
        ``whole_field``.
    reference_fraction : float or None
        Share of the tile used as reference region.
    unwrap : mapping
        Unwrapping record (``method``, ``k_offset``, ``consistent``,
        ``residues``).
    flagged : bool
        Whether the tile failed a consistency check.
    extra : mapping
        Unknown keys of the message, kept for forward compatibility.
    """

    product_id: str
    tile_index: int
    planned_um: Mapping[str, float] | None = None
    actual_um: Mapping[str, float] | None = None
    registration: Mapping[str, Any] = dc_field(default_factory=dict)
    piston_rad: float | None = None
    tilt_rad_per_um: tuple[float, float] | None = None
    anchor: str | None = None
    residual_rad: float | None = None
    border_band_um: float | None = None
    reference_source: str | None = None
    reference_fraction: float | None = None
    unwrap: Mapping[str, Any] = dc_field(default_factory=dict)
    flagged: bool = False
    extra: Mapping[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tile_index", int(self.tile_index))
        object.__setattr__(self, "planned_um", _stage(self.planned_um, "planned_um"))
        object.__setattr__(self, "actual_um", _stage(self.actual_um, "actual_um"))
        if self.tilt_rad_per_um is not None:
            object.__setattr__(
                self, "tilt_rad_per_um", _pair(self.tilt_rad_per_um, "tilt_rad_per_um")
            )
        for name in ("registration", "unwrap", "extra"):
            object.__setattr__(self, name, _freeze(dict(getattr(self, name) or {})))
        object.__setattr__(self, "flagged", bool(self.flagged))

    def to_dict(self) -> dict:
        """Return the contract tile object."""
        data = {
            "product_id": self.product_id,
            "tile_index": self.tile_index,
            "planned_um": dict(self.planned_um) if self.planned_um else None,
            "actual_um": dict(self.actual_um) if self.actual_um else None,
            "registration": _thaw(self.registration),
            "piston_rad": self.piston_rad,
            "tilt_rad_per_um": (
                list(self.tilt_rad_per_um) if self.tilt_rad_per_um is not None else None
            ),
            "anchor": self.anchor,
            "residual_rad": self.residual_rad,
            "border_band_um": self.border_band_um,
            "reference_source": self.reference_source,
            "reference_fraction": self.reference_fraction,
            "unwrap": _thaw(self.unwrap),
            "flagged": self.flagged,
        }
        data.update(_thaw(self.extra))
        return _to_jsonable(data)

    @classmethod
    def from_dict(cls, data: Mapping) -> TileRecord:
        """Build a record from the contract tile object."""
        known = {k: data[k] for k in _TILE_RECORD_KEYS if k in data}
        extra = {k: v for k, v in data.items() if k not in _TILE_RECORD_KEYS}
        if "product_id" not in known or "tile_index" not in known:
            raise ValidationError("A tile record needs product_id and tile_index.")
        return cls(**known, extra=extra)


# ------------------------------------------------------------ tile product
@dataclass(frozen=True)
class TileProduct:
    """One capture at one stage position (contract ``ProductSet`` 1.0).

    Parameters
    ----------
    field : array (complex64)
        Compensated complex field, shape ``grid.shape``; NumPy or CuPy.
        ``abs(field)**2`` is ``intensity_rel``, ``angle(field)`` the wrapped
        phase.
    grid : Grid
        Pixel grid in stage coordinates.
    z_um : float
        Propagation distance of the field.
    propagation_index : float
        Index the propagation used (z is meaningless without it, G4).
    wavelength_um : float
        Vacuum wavelength.
    product_id, context_id : str
        Ids of the ``ProductSet`` and of its ``CaptureContext``.
    position_actual_um : mapping
        Actual stage position ``{x, y, z}`` in µm of the capture.
    position_commanded_um : mapping or None, optional
        Commanded stage position.
    structure_id, group_id : str or None, optional
    layer_index, tile_index : int or None, optional
    n_structure, n_surround : float or None, optional
        Indices from the capture context (never defaulted, G7).
    material : mapping or None, optional
        Material snapshot of the capture context.
    parameter_set_id, config_hash : str or None, optional
        Parameter set of the reconstruction (DL-A3 a).
    background_id : str or None, optional
    quality : mapping, optional
        Quality block of the reconstruction (``reference_source``, …).
    job_id : str or None, optional
    capture : mapping, optional
        The full ``CaptureContext`` as received (provenance).
    reconstruction : mapping, optional
        The ``ReconstructionResult`` header without buffers (frames, timings,
        versions; provenance).
    extra : mapping, optional
        Remaining ``ProductSet`` keys (``raw_ref``, ``recipe_hash``,
        ``service_versions``, ``created_utc``, unknown keys).

    Notes
    -----
    Instances are immutable: the dataclass is frozen, NumPy arrays are
    read-only and metadata mappings are read-only views. Use
    :func:`dataclasses.replace` to derive a changed copy.
    """

    field: Any
    grid: Grid
    z_um: float
    propagation_index: float
    wavelength_um: float
    product_id: str
    context_id: str | None
    position_actual_um: Mapping[str, float]
    position_commanded_um: Mapping[str, float] | None = None
    structure_id: str | None = None
    group_id: str | None = None
    layer_index: int | None = None
    tile_index: int | None = None
    n_structure: float | None = None
    n_surround: float | None = None
    material: Mapping[str, Any] | None = None
    parameter_set_id: str | None = None
    config_hash: str | None = None
    background_id: str | None = None
    quality: Mapping[str, Any] = dc_field(default_factory=dict)
    job_id: str | None = None
    capture: Mapping[str, Any] = dc_field(default_factory=dict)
    reconstruction: Mapping[str, Any] = dc_field(default_factory=dict)
    extra: Mapping[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        values = _as_array(self.field, np.dtype(np.complex64), "field")
        object.__setattr__(self, "field", values)
        if not isinstance(self.grid, Grid):
            raise ValidationError("grid must be a mira.products.Grid.")
        if tuple(values.shape) != self.grid.shape:
            raise ValidationError(
                f"field shape {tuple(values.shape)} does not match grid {self.grid.shape}."
            )
        for name in ("z_um", "propagation_index", "wavelength_um"):
            value = getattr(self, name)
            if value is None or not math.isfinite(float(value)):
                raise ValidationError(f"{name} is required (no default, G4 / G7).")
            object.__setattr__(self, name, float(value))
        if self.propagation_index <= 0 or self.wavelength_um <= 0:
            raise ValidationError(
                "propagation_index and wavelength_um must be positive."
            )
        object.__setattr__(
            self,
            "position_actual_um",
            _stage(self.position_actual_um, "position_actual_um"),
        )
        object.__setattr__(
            self,
            "position_commanded_um",
            _stage(self.position_commanded_um, "position_commanded_um"),
        )
        for name in ("material",):
            value = getattr(self, name)
            object.__setattr__(
                self, name, None if value is None else _freeze(dict(value))
            )
        for name in ("quality", "capture", "reconstruction", "extra"):
            object.__setattr__(self, name, _freeze(dict(getattr(self, name) or {})))

    # ---------------------------------------------------------- channels
    @property
    def phase_rad(self) -> Any:
        """Wrapped phase in rad, float32, same backend as :attr:`field`."""
        xp = _xp(self.field)
        return xp.angle(self.field).astype(xp.float32)

    @property
    def intensity_rel(self) -> Any:
        """``|field|²`` = I_object / I_background, float32 (1.0 = substrate)."""
        xp = _xp(self.field)
        return (self.field.real**2 + self.field.imag**2).astype(xp.float32)

    # ------------------------------------------------------- constructors
    @classmethod
    def from_arrays(
        cls,
        *,
        phase_rad: Any = None,
        intensity_rel: Any = None,
        field: Any = None,
        **kwargs: Any,
    ) -> TileProduct:
        """Build a tile from a complex field or from phase + ``intensity_rel``.

        Parameters
        ----------
        phase_rad, intensity_rel : array, optional
            Wrapped phase and relative intensity; used when ``field`` is None.
        field : array, optional
            Complex field; takes precedence.
        **kwargs
            Remaining :class:`TileProduct` parameters.

        Returns
        -------
        TileProduct
        """
        if field is None:
            if phase_rad is None or intensity_rel is None:
                raise ValidationError(
                    "Give either field or both phase_rad and intensity_rel."
                )
            field = _field_from(phase_rad, intensity_rel)
        return cls(field=field, **kwargs)

    @classmethod
    def from_message(cls, header: Mapping, buffers: Sequence) -> TileProduct:
        """Build a tile from a ``ProductSet`` 1.0 message.

        Parameters
        ----------
        header : mapping
            The ``ProductSet`` header; its nested ``reconstruction`` header
            describes the buffers.
        buffers : sequence
            One entry per buffer descriptor of the reconstruction, in order:
            ndarray, bytes-like or ``(name, payload)``.

        Returns
        -------
        TileProduct

        Raises
        ------
        ValidationError
            Wrong message type or major version, buffers that do not match,
            missing physical parameters.
        """
        _check_message(header, "ProductSet", PRODUCT_SET_SCHEMA_VERSION)
        recon = dict(header.get("reconstruction") or {})
        _check_message(recon, "ReconstructionResult", _RECONSTRUCTION_SCHEMA_VERSION)
        capture = dict(header.get("capture") or {})
        arrays = _decode_buffers(recon.get("buffers", []), buffers)
        recon_meta = {k: v for k, v in recon.items() if k != "buffers"}
        return cls._from_parts(
            arrays=arrays,
            recon=recon_meta,
            capture=capture,
            product_id=header.get("product_id"),
            context_id=header.get("context_id"),
            job_id=header.get("job_id"),
            background_id=header.get("background_id"),
            parameter_set_id=header.get("parameter_set_id"),
            config_hash=header.get("config_hash"),
            extra={
                k: v
                for k, v in header.items()
                if k
                not in {
                    "message_type",
                    "schema_version",
                    "product_id",
                    "context_id",
                    "job_id",
                    "reconstruction",
                    "capture",
                    "background_id",
                    "parameter_set_id",
                    "config_hash",
                }
            },
        )

    @classmethod
    def from_reconstruction(
        cls,
        result: Any,
        capture: Mapping,
        *,
        product_id: str,
        context_id: str | None = None,
        job_id: str | None = None,
        background_id: str | None = None,
    ) -> TileProduct:
        """Build a tile from an OffAxisHolo ``ReconstructionResult``.

        OffAxisHolo is not imported: any object with ``to_message()`` that
        returns a ``ReconstructionResult`` 1.0 header and buffers works.

        Parameters
        ----------
        result : offaxisholo.realtime.ReconstructionResult
            The reconstruction of one capture.
        capture : mapping
            Its ``CaptureContext`` 1.0 (dict).
        product_id : str
            Id of the new product.
        context_id, job_id, background_id : str, optional
            Ids recorded with the product.

        Returns
        -------
        TileProduct
        """
        recon, buffers = result.to_message()
        _check_message(recon, "ReconstructionResult", _RECONSTRUCTION_SCHEMA_VERSION)
        arrays = _decode_buffers(recon.get("buffers", []), buffers)
        quality = recon.get("quality") or {}
        return cls._from_parts(
            arrays=arrays,
            recon={k: v for k, v in recon.items() if k != "buffers"},
            capture=dict(capture),
            product_id=product_id,
            context_id=context_id,
            job_id=job_id,
            background_id=background_id or quality.get("background_id"),
            parameter_set_id=recon.get("parameter_set_id"),
            config_hash=recon.get("config_hash"),
            extra={},
        )

    @classmethod
    def _from_parts(
        cls,
        *,
        arrays: Mapping,
        recon: Mapping,
        capture: Mapping,
        extra: Mapping,
        **ids: Any,
    ) -> TileProduct:
        if "field" in arrays:
            field_ = arrays["field"]
        elif "phase_rad" in arrays and "intensity_rel" in arrays:
            field_ = _field_from(arrays["phase_rad"], arrays["intensity_rel"])
        else:
            raise ValidationError(
                "A tile needs the buffer field or both phase_rad and intensity_rel."
            )
        if "position_actual_um" not in capture:
            raise ValidationError("CaptureContext lacks position_actual_um.")
        try:
            pixel_um = recon["pixel_um"]
        except KeyError as exc:
            raise ValidationError("ReconstructionResult lacks pixel_um.") from exc
        grid = Grid.centred(capture["position_actual_um"], pixel_um, field_.shape)
        return cls(
            field=field_,
            grid=grid,
            z_um=recon.get("z_um"),
            propagation_index=recon.get("propagation_index"),
            wavelength_um=recon.get("wavelength_um"),
            position_actual_um=capture["position_actual_um"],
            position_commanded_um=capture.get("position_commanded_um"),
            structure_id=capture.get("structure_id"),
            group_id=capture.get("group_id"),
            layer_index=capture.get("layer_index"),
            tile_index=capture.get("tile_index"),
            n_structure=capture.get("n_structure"),
            n_surround=capture.get("n_surround"),
            material=capture.get("material"),
            quality=recon.get("quality") or {},
            capture=capture,
            reconstruction=recon,
            extra=extra,
            **ids,
        )

    # ------------------------------------------------------------ message
    def to_message(
        self, *, include_field: bool = False
    ) -> tuple[dict, list[tuple[str, np.ndarray]]]:
        """Serialise as a ``ProductSet`` 1.0 message.

        Parameters
        ----------
        include_field : bool, optional
            Also send the complex field (the contract sends it on request).

        Returns
        -------
        header : dict
            JSON-able ``ProductSet`` header; the nested ``reconstruction``
            header describes the buffers.
        buffers : list of (str, numpy.ndarray)
            ``phase_rad``, ``intensity_rel`` and optionally ``field``.
        """
        arrays = [("phase_rad", self.phase_rad), ("intensity_rel", self.intensity_rel)]
        if include_field:
            arrays.append(("field", self.field))
        buffers = [(name, _wire(name, data)) for name, data in arrays]
        recon = _thaw(self.reconstruction)
        recon.update(
            {
                "message_type": "ReconstructionResult",
                "schema_version": recon.get(
                    "schema_version", _RECONSTRUCTION_SCHEMA_VERSION
                ),
                "pixel_um": list(self.grid.pixel_um),
                "z_um": self.z_um,
                "propagation_index": self.propagation_index,
                "wavelength_um": self.wavelength_um,
                "parameter_set_id": self.parameter_set_id,
                "config_hash": self.config_hash,
                "quality": _thaw(self.quality),
                "buffers": [_descriptor(name, data) for name, data in buffers],
            }
        )
        recon.setdefault("created_utc", _utc_now())
        # A product built from arrays has no received CaptureContext; send the
        # context fields it carries so that from_message can rebuild it.
        capture = _thaw(self.capture)
        for key in (
            "position_actual_um",
            "position_commanded_um",
            "structure_id",
            "group_id",
            "layer_index",
            "tile_index",
            "n_structure",
            "n_surround",
            "material",
        ):
            value = getattr(self, key)
            if key not in capture and value is not None:
                capture[key] = _thaw(value)
        header = _thaw(self.extra)
        header.update(
            {
                "message_type": "ProductSet",
                "schema_version": PRODUCT_SET_SCHEMA_VERSION,
                "product_id": self.product_id,
                "context_id": self.context_id,
                "job_id": self.job_id,
                "reconstruction": recon,
                "capture": capture,
                "background_id": self.background_id,
                "parameter_set_id": self.parameter_set_id,
                "config_hash": self.config_hash,
            }
        )
        header.setdefault("created_utc", _utc_now())
        return _to_jsonable(header), buffers


# ----------------------------------------------------------- layer product
_LAYER_KEYS = (
    "product_id",
    "context_id",
    "job_id",
    "group_id",
    "group_kind",
    "experiment_uuid",
    "structure_id",
    "layer_index",
    "member_ids",
    "quality",
    "z_um",
    "propagation_index",
    "n_surround",
    "wavelength_um",
    "background_ids",
    "parameter_set_id",
    "config_hash",
    "versions",
    "timings_s",
    "warnings",
    "created_utc",
)


@dataclass(frozen=True)
class LayerProduct:
    """Stitched / fused mosaic of one structure and layer (``LayerProduct`` 1.0).

    Parameters
    ----------
    phase_rad : array (float32)
        Unwrapped phase in rad on ``grid`` (2π resolved through the overlaps).
    intensity_rel : array (float32)
        I_object / I_background on ``grid``.
    valid : array (bool)
        Validity mask; False outside the covered area or for flagged pixels.
    grid : Grid
        Common mosaic grid in stage coordinates.
    tiles : sequence of TileRecord
        One record per member tile.
    z_um, propagation_index, wavelength_um : float
        Plane of the mosaic and its propagation index; wavelength.
    product_id : str
        Id of the product.
    field : array (complex64) or None, optional
        Complex field of the same mosaic (on request).
    n_surround : float or None, optional
    context_id, job_id, group_id, group_kind, experiment_uuid, structure_id : str or None
    layer_index : int or None
    member_ids : sequence of str, optional
    quality : mapping, optional
        ``n_tiles``, ``n_flagged``, ``phase_noise_rad``, ``unwrap_method``,
        ``coverage_min``, ``recovered``, …
    background_ids : sequence of str, optional
    parameter_set_id, config_hash : str or None, optional
    versions, timings_s : mapping, optional
    warnings : sequence of str, optional
    created_utc : str or None, optional
    extra : mapping, optional
        Unknown header keys (forward compatibility).

    Notes
    -----
    Immutable like :class:`TileProduct`. Arrays may live on NumPy or CuPy;
    messages and HDF5 always hold NumPy data.
    """

    phase_rad: Any
    intensity_rel: Any
    valid: Any
    grid: Grid
    tiles: tuple[TileRecord, ...]
    z_um: float
    propagation_index: float
    wavelength_um: float
    product_id: str
    field: Any = None
    n_surround: float | None = None
    context_id: str | None = None
    job_id: str | None = None
    group_id: str | None = None
    group_kind: str | None = None
    experiment_uuid: str | None = None
    structure_id: str | None = None
    layer_index: int | None = None
    member_ids: tuple[str, ...] = ()
    quality: Mapping[str, Any] = dc_field(default_factory=dict)
    background_ids: tuple[str, ...] = ()
    parameter_set_id: str | None = None
    config_hash: str | None = None
    versions: Mapping[str, Any] = dc_field(default_factory=dict)
    timings_s: Mapping[str, Any] = dc_field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    created_utc: str | None = None
    extra: Mapping[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.grid, Grid):
            raise ValidationError("grid must be a mira.products.Grid.")
        arrays = {
            "phase_rad": _as_array(self.phase_rad, np.dtype(np.float32), "phase_rad"),
            "intensity_rel": _as_array(
                self.intensity_rel, np.dtype(np.float32), "intensity_rel"
            ),
            "valid": _as_array(self.valid, np.dtype(bool), "valid"),
        }
        if self.field is not None:
            arrays["field"] = _as_array(self.field, np.dtype(np.complex64), "field")
        for name, array in arrays.items():
            if tuple(array.shape) != self.grid.shape:
                raise ValidationError(
                    f"{name} shape {tuple(array.shape)} does not match grid "
                    f"{self.grid.shape}."
                )
            object.__setattr__(self, name, array)
        for name in ("z_um", "propagation_index", "wavelength_um"):
            value = getattr(self, name)
            if value is None or not math.isfinite(float(value)):
                raise ValidationError(f"{name} is required (no default, G4 / G7).")
            object.__setattr__(self, name, float(value))
        tiles = tuple(
            t if isinstance(t, TileRecord) else TileRecord.from_dict(t)
            for t in self.tiles
        )
        object.__setattr__(self, "tiles", tiles)
        for name in ("member_ids", "background_ids", "warnings"):
            object.__setattr__(
                self, name, tuple(str(v) for v in getattr(self, name) or ())
            )
        for name in ("quality", "versions", "timings_s", "extra"):
            object.__setattr__(self, name, _freeze(dict(getattr(self, name) or {})))

    # ------------------------------------------------------------ message
    def _header(self, buffers: list[tuple[str, np.ndarray]]) -> dict:
        header = _thaw(self.extra)
        header.update(
            {
                "message_type": "LayerProduct",
                "schema_version": LAYER_PRODUCT_SCHEMA_VERSION,
                "product_id": self.product_id,
                "context_id": self.context_id,
                "job_id": self.job_id,
                "group_id": self.group_id,
                "group_kind": self.group_kind,
                "experiment_uuid": self.experiment_uuid,
                "structure_id": self.structure_id,
                "layer_index": self.layer_index,
                "grid": self.grid.to_dict(),
                "tiles": [tile.to_dict() for tile in self.tiles],
                "member_ids": list(self.member_ids),
                "quality": _thaw(self.quality),
                "z_um": self.z_um,
                "propagation_index": self.propagation_index,
                "n_surround": self.n_surround,
                "wavelength_um": self.wavelength_um,
                "background_ids": list(self.background_ids),
                "parameter_set_id": self.parameter_set_id,
                "config_hash": self.config_hash,
                "versions": _thaw(self.versions),
                "timings_s": _thaw(self.timings_s),
                "warnings": list(self.warnings),
                "buffers": [_descriptor(name, data) for name, data in buffers],
                "created_utc": self.created_utc or _utc_now(),
            }
        )
        return _to_jsonable(header)

    def _buffers(self, include_field: bool) -> list[tuple[str, np.ndarray]]:
        arrays = [
            ("phase_rad", self.phase_rad),
            ("intensity_rel", self.intensity_rel),
            ("valid", self.valid),
        ]
        if include_field and self.field is not None:
            arrays.append(("field", self.field))
        return [(name, _wire(name, data)) for name, data in arrays]

    def to_message(
        self, *, include_field: bool = True
    ) -> tuple[dict, list[tuple[str, np.ndarray]]]:
        """Serialise as a ``LayerProduct`` 1.0 message.

        Parameters
        ----------
        include_field : bool, optional
            Send the complex field if the product has one (default True).

        Returns
        -------
        header : dict
        buffers : list of (str, numpy.ndarray)
            ``phase_rad``, ``intensity_rel``, ``valid`` (uint8) and optionally
            ``field``.
        """
        buffers = self._buffers(include_field)
        return self._header(buffers), buffers

    @classmethod
    def from_message(cls, header: Mapping, buffers: Sequence) -> LayerProduct:
        """Build a product from a ``LayerProduct`` 1.0 message.

        Parameters
        ----------
        header : mapping
            The JSON header.
        buffers : sequence
            One entry per ``header["buffers"]`` descriptor, in order.

        Returns
        -------
        LayerProduct

        Raises
        ------
        ValidationError
            Wrong message type or major version, missing buffers or fields.
        """
        _check_message(header, "LayerProduct", LAYER_PRODUCT_SCHEMA_VERSION)
        arrays = _decode_buffers(header.get("buffers", []), buffers)
        missing = {"phase_rad", "intensity_rel", "valid"} - set(arrays)
        if missing:
            raise ValidationError(f"Missing buffers: {sorted(missing)}.")
        if "grid" not in header:
            raise ValidationError("LayerProduct lacks grid.")
        known = {k: header[k] for k in _LAYER_KEYS if k in header}
        extra = {
            k: v
            for k, v in header.items()
            if k not in _LAYER_KEYS
            and k not in {"message_type", "schema_version", "grid", "tiles", "buffers"}
        }
        return cls(
            phase_rad=arrays["phase_rad"],
            intensity_rel=arrays["intensity_rel"],
            valid=arrays["valid"].astype(bool),
            field=arrays.get("field"),
            grid=Grid.from_dict(header["grid"]),
            tiles=tuple(TileRecord.from_dict(t) for t in header.get("tiles", [])),
            extra=extra,
            **known,
        )

    # --------------------------------------------------------------- HDF5
    def to_h5(self, group: Any, *, include_field: bool = True) -> None:
        """Write the product into an HDF5 group (``h5py.Group``).

        Layout: attribute ``header_json`` holds the contract header; one
        dataset per buffer under its contract name (``valid`` as uint8).

        Parameters
        ----------
        group : h5py.Group
            Empty target group.
        include_field : bool, optional
            Also write the complex field if present.
        """
        header, buffers = self.to_message(include_field=include_field)
        for name, data in buffers:
            group.create_dataset(name, data=data, compression="gzip", shuffle=True)
        group.attrs["header_json"] = json.dumps(header)

    @classmethod
    def from_h5(cls, group: Any) -> LayerProduct:
        """Read a product written by :meth:`to_h5`.

        Only ``h5py`` objects are touched; nanofactorysystem is not imported.

        Parameters
        ----------
        group : h5py.Group
            Group with the attribute ``header_json`` and one dataset per
            buffer.

        Returns
        -------
        LayerProduct
        """
        try:
            header = json.loads(group.attrs["header_json"])
        except KeyError as exc:
            raise ValidationError(
                f"HDF5 group {getattr(group, 'name', group)!r} has no header_json."
            ) from exc
        buffers = []
        for descriptor in header.get("buffers", []):
            name = descriptor["name"]
            if name not in group:
                raise ValidationError(f"HDF5 group lacks the dataset {name!r}.")
            buffers.append((name, np.asarray(group[name][()])))
        return cls.from_message(header, buffers)


# ---------------------------------------------------------------- bridge
def to_structure_processor(
    product: TileProduct | LayerProduct,
    *,
    n_resin: float,
    mask_invalid: bool = True,
    **kwargs: Any,
):
    """Create a :class:`mira.StructureProcessor` from a product.

    This is the **one** place where the contract conventions (``(y, x)`` in
    µm) are converted to the conventions of ``StructureProcessor``
    (``pixel_size`` = ``(x, y)`` in metres, wavelength in metres). The image
    arrays keep their axis order (axis 0 = y).

    Parameters
    ----------
    product : TileProduct or LayerProduct
        For a tile the wrapped phase is used, for a layer the unwrapped one.
    n_resin : float
        Refractive index for the phase → height conversion. Required: there is
        no silent default (G7); take it from the material snapshot.
    mask_invalid : bool, optional
        For a :class:`LayerProduct`, mask the pixels outside ``valid``
        (``numpy.ma.MaskedArray``).
    **kwargs
        Passed on to ``StructureProcessor`` (e.g. ``description``).

    Returns
    -------
    mira.StructureProcessor
    """
    from .processor import StructureProcessor

    if n_resin is None or not math.isfinite(float(n_resin)):
        raise ValidationError("n_resin is required (no silent default, G7).")
    py_um, px_um = product.grid.pixel_um
    pixel_size_xy_m = (px_um * 1e-6, py_um * 1e-6)
    phase = np.asarray(_to_numpy(product.phase_rad), dtype=np.float64)
    intensity = np.asarray(_to_numpy(product.intensity_rel), dtype=np.float64)
    if mask_invalid and isinstance(product, LayerProduct):
        invalid = ~np.asarray(_to_numpy(product.valid), dtype=bool)
        if invalid.any():
            phase = np.ma.masked_array(phase, mask=invalid)
            intensity = np.ma.masked_array(intensity, mask=invalid)
    return StructureProcessor(
        phase_data=phase,
        intensity_data=intensity,
        pixel_size=pixel_size_xy_m,
        n_resin=float(n_resin),
        wavelength=product.wavelength_um * 1e-6,
        **kwargs,
    )
