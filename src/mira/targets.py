"""Layer targets and their transfer onto the product grid (MI-50).

nanofactorysystem rasterises the expected geometry after layer *n*
(``LayerTarget`` 1.0, decision M6) at least as fine as the product. MIRA only
shifts and **downsamples** it onto the pixel grid of a
:class:`~mira.products.LayerProduct`, area-weighted, and never upsamples.

Until the stage-to-image calibration (MI-11) exists both grids are taken as
axis-aligned in stage coordinates (G10 SA2: rotation 0, nominal pixel size);
the detected pose of the structure enters as a shift in µm.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import numpy as np
from scipy import sparse

from .products import (
    Grid,
    _check_message,
    _decode_buffers,
    _descriptor,
    _freeze,
    _thaw,
    _to_jsonable,
    _utc_now,
    _wire,
)
from .utils.exceptions import ValidationError

__all__ = ["LAYER_TARGET_SCHEMA_VERSION", "LayerTarget", "TargetOnGrid", "resample_target"]

LAYER_TARGET_SCHEMA_VERSION = "1.0"
_VALUE_BUFFER = {"height": "height_um", "phase": "phase_rad"}
_TARGET_KEYS = (
    "target_id",
    "experiment_uuid",
    "structure_id",
    "layer_index",
    "source",
    "footprint_um",
    "material",
    "n_structure",
    "n_surround",
    "generator",
    "warnings",
    "created_utc",
)


@dataclass(frozen=True)
class LayerTarget:
    """Expected geometry of a structure after one layer (``LayerTarget`` 1.0).

    Parameters
    ----------
    values : numpy.ndarray
        ``height_um`` or ``phase_rad`` (by ``domain``), float32, on ``grid``.
    mask : numpy.ndarray (bool)
        Exact structure mask on ``grid``.
    grid : Grid
        Raster grid in stage coordinates.
    domain : {"height", "phase"}
        What ``values`` holds.
    target_id : str
    source : {"toolpath_voxel", "footprint", "design_heightmap"}, optional
    footprint_um : sequence, optional
        Footprint polygons, vertices ``(y, x)`` in stage µm.
    material : mapping or None, optional
        Material snapshot.
    n_structure, n_surround : float or None, optional
        Indices for the conversion between the domains (no default, G7).
    experiment_uuid, structure_id : str or None, optional
    layer_index : int or None, optional
    generator : mapping, optional
    warnings : sequence of str, optional
    created_utc : str or None, optional
    extra : mapping, optional
        Unknown header keys.
    """

    values: np.ndarray
    mask: np.ndarray
    grid: Grid
    domain: str
    target_id: str
    source: str | None = None
    footprint_um: Sequence = ()
    material: Mapping[str, Any] | None = None
    n_structure: float | None = None
    n_surround: float | None = None
    experiment_uuid: str | None = None
    structure_id: str | None = None
    layer_index: int | None = None
    generator: Mapping[str, Any] = dc_field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    created_utc: str | None = None
    extra: Mapping[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.domain not in _VALUE_BUFFER:
            raise ValidationError(f"domain must be 'height' or 'phase'; got {self.domain!r}.")
        values = np.array(self.values, dtype=np.float32)
        mask = np.array(self.mask, dtype=bool)
        for name, array in (("values", values), ("mask", mask)):
            if array.shape != self.grid.shape:
                raise ValidationError(
                    f"{name} shape {array.shape} does not match grid {self.grid.shape}."
                )
            array.flags.writeable = False
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "footprint_um", _freeze(list(self.footprint_um or ())))
        if self.material is not None:
            object.__setattr__(self, "material", _freeze(dict(self.material)))
        object.__setattr__(self, "generator", _freeze(dict(self.generator or {})))
        object.__setattr__(self, "warnings", tuple(str(w) for w in self.warnings or ()))
        object.__setattr__(self, "extra", _freeze(dict(self.extra or {})))

    def to_message(self) -> tuple[dict, list[tuple[str, np.ndarray]]]:
        """Serialise as a ``LayerTarget`` 1.0 message (values + ``mask`` uint8)."""
        buffers = [
            (_VALUE_BUFFER[self.domain], np.ascontiguousarray(self.values, dtype="<f4")),
            ("mask", _wire("valid", self.mask)),
        ]
        header = _thaw(self.extra)
        header.update(
            {
                "message_type": "LayerTarget",
                "schema_version": LAYER_TARGET_SCHEMA_VERSION,
                "target_id": self.target_id,
                "experiment_uuid": self.experiment_uuid,
                "structure_id": self.structure_id,
                "layer_index": self.layer_index,
                "domain": self.domain,
                "source": self.source,
                "grid": self.grid.to_dict(),
                "footprint_um": _thaw(self.footprint_um),
                "material": _thaw(self.material) if self.material is not None else None,
                "n_structure": self.n_structure,
                "n_surround": self.n_surround,
                "generator": _thaw(self.generator),
                "warnings": list(self.warnings),
                "buffers": [_descriptor(name, data) for name, data in buffers],
                "created_utc": self.created_utc or _utc_now(),
            }
        )
        return _to_jsonable(header), buffers

    @classmethod
    def from_message(cls, header: Mapping, buffers: Sequence) -> LayerTarget:
        """Build a target from a ``LayerTarget`` 1.0 message.

        Parameters
        ----------
        header : mapping
        buffers : sequence
            One entry per descriptor: ``height_um`` or ``phase_rad`` and
            ``mask``.

        Returns
        -------
        LayerTarget
        """
        _check_message(header, "LayerTarget", LAYER_TARGET_SCHEMA_VERSION)
        domain = header.get("domain")
        if domain not in _VALUE_BUFFER:
            raise ValidationError(f"Unknown target domain {domain!r}.")
        descriptors = header.get("buffers", [])
        # The mask travels as uint8 like "valid"; rename for the decoder.
        renamed = [dict(d, name="valid") if d["name"] == "mask" else d for d in descriptors]
        value_name = _VALUE_BUFFER[domain]
        renamed = [
            dict(d, name="phase_rad") if d["name"] == value_name else d for d in renamed
        ]
        payloads = [p[1] if isinstance(p, tuple) else p for p in buffers]
        arrays = _decode_buffers(renamed, payloads)
        if "phase_rad" not in arrays or "valid" not in arrays:
            raise ValidationError(f"LayerTarget needs the buffers {value_name!r} and 'mask'.")
        known = {k: header[k] for k in _TARGET_KEYS if k in header}
        extra = {
            k: v
            for k, v in header.items()
            if k not in _TARGET_KEYS
            and k not in {"message_type", "schema_version", "domain", "grid", "buffers"}
        }
        return cls(
            values=arrays["phase_rad"],
            mask=arrays["valid"].astype(bool),
            grid=Grid.from_dict(header["grid"]),
            domain=domain,
            extra=extra,
            **known,
        )


@dataclass(frozen=True)
class TargetOnGrid:
    """A target transferred onto a product grid.

    Parameters
    ----------
    values : numpy.ndarray
        Area-weighted mean of the target values per product pixel (float64);
        NaN where the target grid does not reach.
    mask_fraction : numpy.ndarray
        Share of every product pixel covered by the structure mask (0 … 1).
    coverage : numpy.ndarray
        Share of every product pixel covered by the target grid (0 … 1).
    grid : Grid
        The product grid.
    domain : str
        Domain of ``values`` (``"height"`` or ``"phase"``).
    shift_um : tuple of float
        Shift ``(y, x)`` applied to the target (pose of the structure).
    """

    values: np.ndarray
    mask_fraction: np.ndarray
    coverage: np.ndarray
    grid: Grid
    domain: str
    shift_um: tuple[float, float]

    @property
    def mask(self) -> np.ndarray:
        """Structure mask: at least half of the pixel inside the footprint."""
        return self.mask_fraction >= 0.5

    @property
    def substrate(self) -> np.ndarray:
        """Pixels fully covered by the target grid and free of structure."""
        return (self.coverage > 0.999) & (self.mask_fraction <= 0.0)

    @property
    def matrix_um(self) -> list[list[float]]:
        """2×3 affine matrix target → product in µm (``(y, x)`` rows)."""
        return [[1.0, 0.0, self.shift_um[0]], [0.0, 1.0, self.shift_um[1]]]


def _overlap_matrix(
    product_origin: float, product_pixel: float, n_product: int,
    target_origin: float, target_pixel: float, n_target: int,
) -> sparse.csr_matrix:
    """Overlap lengths between product bins (rows) and target bins (columns)."""
    edge0 = product_origin - 0.5 * product_pixel
    lo = target_origin - 0.5 * target_pixel + target_pixel * np.arange(n_target)
    hi = lo + target_pixel
    # Downsampling only: a target bin touches at most two product bins.
    k0 = np.floor((lo - edge0) / product_pixel).astype(np.int64)
    rows, cols, vals = [], [], []
    for k in (k0, k0 + 1):
        b_lo = edge0 + k * product_pixel
        overlap = np.minimum(hi, b_lo + product_pixel) - np.maximum(lo, b_lo)
        keep = (overlap > 0) & (k >= 0) & (k < n_product)
        rows.append(k[keep])
        cols.append(np.nonzero(keep)[0])
        vals.append(overlap[keep])
    return sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_product, n_target),
    )


def resample_target(
    target: LayerTarget, grid: Grid, *, shift_um: tuple[float, float] = (0.0, 0.0)
) -> TargetOnGrid:
    """Transfer a target onto a product grid (shift + area-weighted downsampling).

    Parameters
    ----------
    target : LayerTarget
        The target raster.
    grid : Grid
        Product grid (e.g. ``LayerProduct.grid``).
    shift_um : tuple of float, optional
        Shift ``(y, x)`` in µm applied to the target, e.g. the detected pose
        offset of the structure relative to the stage position.

    Returns
    -------
    TargetOnGrid

    Raises
    ------
    ValidationError
        If the target is coarser than the product on any axis (M6: never
        upsample; the rasteriser must deliver at least the product pitch).
    """
    for axis in (0, 1):
        if target.grid.pixel_um[axis] > grid.pixel_um[axis] * (1 + 1e-9):
            raise ValidationError(
                f"Target pixel {target.grid.pixel_um} is coarser than the product "
                f"pixel {grid.pixel_um}; MIRA never upsamples (M6)."
            )
    if not all(math.isfinite(s) for s in shift_um):
        raise ValidationError("shift_um must be finite.")
    wy = _overlap_matrix(
        grid.origin_um[0], grid.pixel_um[0], grid.shape[0],
        target.grid.origin_um[0] + shift_um[0], target.grid.pixel_um[0], target.grid.shape[0],
    )
    wx = _overlap_matrix(
        grid.origin_um[1], grid.pixel_um[1], grid.shape[1],
        target.grid.origin_um[1] + shift_um[1], target.grid.pixel_um[1], target.grid.shape[1],
    )

    def integrate(array: np.ndarray) -> np.ndarray:
        partial = wy @ np.asarray(array, dtype=np.float64)  # (ny_p, nx_t)
        return np.asarray((wx @ partial.T).T)  # (ny_p, nx_p)

    pixel_area = grid.pixel_um[0] * grid.pixel_um[1]
    covered = integrate(np.ones(target.grid.shape))
    coverage = covered / pixel_area
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(covered > 0, integrate(target.values) / covered, np.nan)
    mask_fraction = integrate(target.mask) / pixel_area
    return TargetOnGrid(
        values=values,
        mask_fraction=np.clip(mask_fraction, 0.0, 1.0),
        coverage=np.clip(coverage, 0.0, 1.0),
        grid=grid,
        domain=target.domain,
        shift_um=(float(shift_um[0]), float(shift_um[1])),
    )
