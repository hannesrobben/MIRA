"""One pipeline for speckle fusion (MI-30) and stitching (MI-40), decision M4.

A :class:`MosaicBuilder` receives the tiles of one structure and layer one by
one (incremental, so the DHMConnect plug-in MI-60 can work while the stage
moves) and returns a :class:`~mira.products.LayerProduct` on :meth:`close`.
Fusion and stitching are two configurations (:meth:`MosaicConfig.fusion`,
:meth:`MosaicConfig.stitching`) of the same steps:

1. **Border band** (G10 SA6): a band of ``border_band_um`` along every tile
   edge is discarded (estimate ``|z| · tan θ_max`` with ``sin θ_max = NA / n``;
   :func:`border_band_estimate_um`).
2. **Registration** (MI-20) of the tile against the mosaic accumulated so far
   — the tiles measured before are ground truth — with the stage position as
   prior; low confidence keeps the stage position and flags the tile.
3. **Resampling** onto the mosaic lattice (sub-pixel shift, cubic spline on
   real and imaginary part).
4. **Piston / tilt** against the mosaic in the overlap (MI-21, the one
   implementation shared by fusion and stitching), weighted by the reference
   region when one exists. The first tile is levelled on its own reference
   region.
5. **Unwrapping per tile** (stitching, ``fast2d`` = scikit-image, the same
   algorithm as OffAxisHolo K11): the integer ``k · 2π`` follows from the
   unwrapped difference to the mosaic, with a **consistency check** (the
   difference must be constant; otherwise the tile is flagged).
6. **Global least-squares adjustment** on :meth:`close`: one piston per tile
   from all pairwise overlaps; tiles with enough reference region (substrate)
   are absolute anchors (ring tiles, SA6), tiles without substrate are
   anchored through the overlaps (SA4: ``reference_source`` /
   ``reference_fraction`` per tile).
7. **Combination**: the complex field is the weighted mean ``Σ w U / Σ w``
   (implicit amplitude weighting, optional quality weight per tile, M3) with a
   count per pixel and ``min_count``. The phase of the product is either the
   unwrapped phase of the fused field (fusion) or the amplitude-weighted mean
   of the per-tile unwrapped phases (stitching); ``combine="median"`` gives
   the per-pixel median of the tile phases for the comparison of MI-30.

The reference region follows the ladder of G10 SA4: the a-priori mask (SA1,
``occupied_um`` polygons in stage µm plus a margin are excluded) is the
default; without polygons MIRA's structure detection on each tile gives an
a-posteriori mask (:func:`detect_reference_region`, MI-43); where the
detection finds no clear structure / substrate contrast the whole field is
used with ``reference_fraction = 0`` and the anchoring goes through the
overlaps.
"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from .noise import reference_noise
from .overlap import fit_piston_tilt, remove_piston_tilt
from .products import Grid, LayerProduct, TileProduct, TileRecord
from .register import corrected_grid, register_pair
from .utils.exceptions import ValidationError

__all__ = [
    "MosaicConfig",
    "MosaicBuilder",
    "mosaic_grid",
    "rasterise_polygons",
    "border_band_estimate_um",
    "build_mosaic",
    "detect_reference_region",
    "ReferenceDetection",
    "mosaic_memory_bytes",
]

_TWO_PI = 2.0 * math.pi


def border_band_estimate_um(z_um: float, na: float, n_medium: float) -> float:
    """Estimate of the border band ``|z| · tan θ_max`` with ``sin θ_max = NA / n``.

    Parameters
    ----------
    z_um : float
        Propagation distance.
    na : float
        Numerical aperture (the effective NA of the sideband filter may be
        smaller).
    n_medium : float
        Index of the propagation medium.

    Returns
    -------
    float
    """
    s = min(na / n_medium, 0.999999)
    return abs(z_um) * s / math.sqrt(1.0 - s * s)


#: Bytes per pixel, measured with ``tracemalloc`` (Office PC, 2026-11-19, MI-42).
_BYTES_LATTICE_STATE = 23  # complex64 + 3 float32 accumulators + 3 bool masks
_BYTES_TILE_STATE = 14  # complex64 field + float32 phase + 2 bool masks
_BYTES_TILE_TRANSIENT = 210  # 64-bit temporaries of add_tile (measured 204)
_BYTES_LATTICE_CLOSE = 90  # temporaries and outputs of close (measured 76-85)


def mosaic_memory_bytes(
    grid_shape: Sequence[int],
    tile_shape: Sequence[int],
    n_tiles: int,
    *,
    combine: str = "mean",
) -> dict:
    """Memory estimate of a :class:`MosaicBuilder` (MI-42).

    The builder keeps its state in 32 bit (complex64 / float32): about
    23 bytes per mosaic pixel plus 14 bytes per pixel of every added tile.
    :meth:`MosaicBuilder.add_tile` needs about 210 bytes per tile pixel of
    64-bit temporaries on top; :meth:`MosaicBuilder.close` about 90 bytes per
    mosaic pixel, plus ``4 · n_tiles`` bytes per mosaic pixel for
    ``combine="median"`` (a float32 stack of all tile phases). Coefficients
    were measured with ``tracemalloc`` on synthetic rasters (Office PC,
    2026-11-19).

    Example: a 300 µm structure at 63x (mosaic ≈ 1725² px, 7 × 7 tiles of
    512² px) needs ≈ 250 MB of state and ≈ 520 MB at the peak; with the
    median ≈ 1.1 GB.

    Parameters
    ----------
    grid_shape : sequence of int
        Shape of the mosaic lattice.
    tile_shape : sequence of int
        Shape of one tile.
    n_tiles : int
        Number of tiles (``k · k`` for a raster).
    combine : {"mean", "median"}, optional

    Returns
    -------
    dict
        ``state_bytes`` (after all tiles are added) and ``peak_bytes``.
    """
    g = int(grid_shape[0]) * int(grid_shape[1])
    t = int(tile_shape[0]) * int(tile_shape[1])
    state = _BYTES_LATTICE_STATE * g + _BYTES_TILE_STATE * t * int(n_tiles)
    close = _BYTES_LATTICE_CLOSE * g
    if combine == "median":
        close += 4 * int(n_tiles) * g
    peak = state + max(_BYTES_TILE_TRANSIENT * t, close)
    return {"state_bytes": int(state), "peak_bytes": int(peak)}


def rasterise_polygons(
    polygons_um: Sequence[Sequence[Sequence[float]]], grid: Grid, margin_um: float = 0.0
) -> np.ndarray:
    """Boolean raster of polygons (stage µm, vertices ``(y, x)``) on a grid.

    Even-odd rule at the pixel centres; ``margin_um`` dilates the result.

    Parameters
    ----------
    polygons_um : sequence of polygons
        Each polygon is a sequence of ``(y, x)`` vertices in stage µm (as
        ``occupied_um`` of the capture context).
    grid : Grid
    margin_um : float, optional

    Returns
    -------
    numpy.ndarray of bool
    """
    y = grid.origin_um[0] + grid.pixel_um[0] * np.arange(grid.shape[0])
    x = grid.origin_um[1] + grid.pixel_um[1] * np.arange(grid.shape[1])
    yy, xx = np.meshgrid(y, x, indexing="ij")
    inside = np.zeros(grid.shape, bool)
    for poly in polygons_um:
        v = np.asarray(poly, float)
        if v.ndim != 2 or v.shape[0] < 3 or v.shape[1] != 2:
            raise ValidationError("Each polygon needs at least 3 (y, x) vertices.")
        hit = np.zeros(grid.shape, bool)
        for (y0, x0), (y1, x1) in zip(v, np.roll(v, -1, axis=0)):
            crosses = (y0 > yy) != (y1 > yy)
            with np.errstate(divide="ignore", invalid="ignore"):
                x_cross = x0 + (yy - y0) * (x1 - x0) / (y1 - y0)
            hit ^= crosses & (xx < x_cross)
        inside |= hit
    if margin_um > 0 and inside.any():
        dist = ndimage.distance_transform_edt(~inside, sampling=grid.pixel_um)
        inside = dist <= margin_um
    return inside


def mosaic_grid(
    centres_um: Sequence[Mapping[str, float] | Sequence[float]],
    tile_shape: Sequence[int],
    pixel_um: Sequence[float],
    margin_um: float = 0.0,
) -> Grid:
    """Common lattice covering all planned tiles.

    Parameters
    ----------
    centres_um : sequence
        Planned (or actual) tile centres as ``{"y", "x"}`` or ``(y, x)``.
    tile_shape : sequence of int
        ``(ny, nx)`` of one tile.
    pixel_um : sequence of float
        Pixel size ``(y, x)`` of the tiles (= mosaic pixel).
    margin_um : float, optional
        Extra margin on every side (pose tolerance).

    Returns
    -------
    Grid
    """
    pts = np.array(
        [
            [c["y"], c["x"]] if isinstance(c, Mapping) else list(c)[:2]
            for c in centres_um
        ],
        float,
    )
    if pts.size == 0:
        raise ValidationError("Need at least one tile centre.")
    py, px = (float(v) for v in pixel_um)
    half = np.array([0.5 * (tile_shape[0] - 1) * py, 0.5 * (tile_shape[1] - 1) * px])
    lo = pts.min(axis=0) - half - margin_um
    hi = pts.max(axis=0) + half + margin_um
    shape = (
        int(math.ceil((hi[0] - lo[0]) / py)) + 1,
        int(math.ceil((hi[1] - lo[1]) / px)) + 1,
    )
    return Grid((float(lo[0]), float(lo[1])), (py, px), shape)


@dataclass(frozen=True)
class MosaicConfig:
    """Parameters of the shared fusion / stitching pipeline.

    Parameters
    ----------
    mode : {"fusion", "stitching"}
        Name of the configuration (recorded in the product).
    border_band_um : float
        Width of the band discarded at every tile edge.
    pose_tolerance_um : float
        Search radius of the registration.
    min_confidence : float
        Registration confidence below which the stage position is kept
        (no default until M7).
    min_isotropy : float
        Registration drops the shift along directions with too little
        gradient information (see :func:`mira.register.register_pair`; no
        default until M7).
    phase_from : {"field", "tiles"}
        Product phase = unwrapped phase of the fused field, or the weighted
        mean of the per-tile unwrapped phases.
    unwrap : {"fast2d", "none"}
        Spatial unwrapping (per tile for ``phase_from="tiles"``, of the fused
        field otherwise).
    combine : {"mean", "median"}
        Per-pixel combination of the tile phases (``median`` only for
        ``phase_from="tiles"`` style comparison; the field stays the mean).
    fit_tilt : bool
        Fit a tilt per tile in addition to the piston.
    min_count : int
        Pixels covered by fewer tiles are invalid.
    reference_margin_um : float
        Margin added around ``occupied_um`` (``pose_tolerance_um`` of the
        context + ``optical_margin_um`` of the recipe).
    anchor_min_fraction : float
        Tiles whose reference region covers at least this share are absolute
        anchors in the global adjustment.
    min_reference_fraction : float or None
        Warn when a tile's reference share is below it (SA4); None = no warning.
    consistency_max_fraction : float
        Share of overlap pixels allowed to deviate by more than π/2 from the
        constant ``k · 2π`` difference before the tile is flagged (start value;
        final after K11-T).
    min_overlap_px : int
        Minimum overlap per axis for registration and the global adjustment.
    reference_detection : {"threshold", "none"}
        Without an a-priori mask: derive the reference region per tile from
        the structure detection (``"threshold"``, MI-43) or use the whole
        field at once (``"none"``).
    detection_min_contrast : float
        Minimum structure / substrate contrast of the detection in units of
        the robust phase noise of the detected substrate; below it the tile
        falls back to ``whole_field`` (start value until M7).
    detection_min_size_px : int
        Detected structure parts smaller than this are ignored.
    structure_sign : {1, -1}
        Sign of the structure phase relative to the substrate (G4: more
        optical path → positive phase).
    """

    mode: str
    border_band_um: float
    pose_tolerance_um: float
    min_confidence: float
    min_isotropy: float
    phase_from: str = "tiles"
    unwrap: str = "fast2d"
    combine: str = "mean"
    fit_tilt: bool = True
    min_count: int = 1
    reference_margin_um: float = 0.0
    anchor_min_fraction: float = 0.2
    min_reference_fraction: float | None = None
    consistency_max_fraction: float = 0.01
    min_overlap_px: int = 16
    reference_detection: str = "threshold"
    detection_min_contrast: float = 5.0
    detection_min_size_px: int = 50
    structure_sign: int = 1

    def __post_init__(self) -> None:
        if self.reference_detection not in ("threshold", "none"):
            raise ValidationError("reference_detection must be 'threshold' or 'none'.")
        if self.structure_sign not in (1, -1):
            raise ValidationError("structure_sign must be 1 or -1.")
        if not self.detection_min_contrast > 0:
            raise ValidationError("detection_min_contrast must be > 0.")
        if self.mode not in ("fusion", "stitching"):
            raise ValidationError(
                f"mode must be 'fusion' or 'stitching'; got {self.mode!r}."
            )
        if self.phase_from not in ("field", "tiles"):
            raise ValidationError("phase_from must be 'field' or 'tiles'.")
        if self.unwrap not in ("fast2d", "none"):
            raise ValidationError("unwrap must be 'fast2d' or 'none'.")
        if self.combine not in ("mean", "median"):
            raise ValidationError("combine must be 'mean' or 'median'.")
        for name in ("border_band_um", "pose_tolerance_um", "reference_margin_um"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value >= 0):
                raise ValidationError(f"{name} must be finite and >= 0.")
        if self.min_count < 1:
            raise ValidationError("min_count must be >= 1.")

    @classmethod
    def fusion(cls, **kwargs: Any) -> MosaicConfig:
        """Speckle fusion: complex mean, phase from the fused field (M3)."""
        params = {"mode": "fusion", "phase_from": "field"}
        params.update(kwargs)
        return cls(**params)

    @classmethod
    def stitching(cls, **kwargs: Any) -> MosaicConfig:
        """Stitching (SA6): unwrapping per tile, phase from the tiles."""
        params = {"mode": "stitching", "phase_from": "tiles"}
        params.update(kwargs)
        return cls(**params)

    def to_dict(self) -> dict:
        """Parameters as a JSON-ready mapping."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class _Tile:
    """Processed tile on the mosaic lattice (internal)."""

    index: int
    product_id: str
    sl: tuple[slice, slice]
    field: np.ndarray  # complex64, corrected (MI-42: 32-bit state)
    phase: np.ndarray  # float32, unwrapped (or wrapped if unwrap == "none")
    valid: np.ndarray
    reference: np.ndarray  # reference-region mask on the crop
    weight: float
    record: dict
    noise_rad: float | None
    box_field: np.ndarray | None = None
    box_grid: Grid | None = None


def _unwrap(phase: np.ndarray, valid: np.ndarray) -> np.ndarray:
    from skimage.restoration import unwrap_phase

    if valid.all():
        return unwrap_phase(phase)
    masked = np.ma.masked_array(phase, mask=~valid)
    out = unwrap_phase(masked)
    return np.asarray(np.ma.filled(out, 0.0), float)


@dataclass(frozen=True)
class ReferenceDetection:
    """Result of :func:`detect_reference_region`.

    Parameters
    ----------
    reference : numpy.ndarray of bool or None
        Reference (substrate) region, or None when the detection found no
        clear structure / substrate contrast.
    structure : numpy.ndarray of bool
        Detected structure (before the margin); empty when ``reference`` is
        None.
    contrast : float
        Median structure phase minus median substrate phase, in units of the
        robust substrate noise (0 when nothing was detected).
    noise_rad : float or None
        Robust (MAD) phase noise of the detected substrate after the plane fit.
    """

    reference: np.ndarray | None
    structure: np.ndarray
    contrast: float
    noise_rad: float | None

    def to_record(self) -> dict:
        """JSON-ready summary for the tile record."""
        return {
            "method": "threshold",
            "accepted": self.reference is not None,
            "contrast": float(self.contrast),
            "noise_rad": self.noise_rad,
            "structure_px": int(self.structure.sum()),
        }


def _plane(phase: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Least-squares plane through ``phase[mask]`` evaluated on the full array."""
    yy, xx = np.indices(phase.shape, dtype=float)
    a = np.column_stack([np.ones(mask.sum()), yy[mask], xx[mask]])
    coef, *_ = np.linalg.lstsq(a, phase[mask], rcond=None)
    return coef[0] + coef[1] * yy + coef[2] * xx


def detect_reference_region(
    field: np.ndarray,
    valid: np.ndarray,
    pixel_um: Sequence[float],
    *,
    min_contrast: float = 5.0,
    margin_um: float = 0.0,
    min_size_px: int = 50,
    structure_sign: int = 1,
    iterations: int = 3,
) -> ReferenceDetection:
    """A-posteriori reference region of one tile (G10 SA4, MI-43).

    The wrapped phase of the tile is unwrapped (``fast2d``); a plane is fitted
    to the current substrate estimate (first: all valid pixels) and removed;
    the structure is the Otsu class above the threshold
    (:func:`mira.processing.detection.threshold_detection`, cleaned with
    :func:`~mira.processing.detection.clean_structure_mask`). Plane fit and
    detection alternate ``iterations`` times. The detection is accepted only
    if the structure / substrate contrast exceeds ``min_contrast`` times the
    robust substrate noise; a tile that lies entirely on the structure or on
    the substrate shows no such contrast and is rejected (the caller falls
    back to ``whole_field``).

    Parameters
    ----------
    field : numpy.ndarray (complex)
        Tile field; only ``valid`` pixels are used.
    valid : numpy.ndarray of bool
    pixel_um : sequence of float
        Pixel size ``(y, x)`` for the margin.
    min_contrast : float, optional
        Acceptance threshold in units of the substrate noise.
    margin_um : float, optional
        Dilation of the structure before the complement is taken.
    min_size_px : int, optional
        Structure parts smaller than this are ignored.
    structure_sign : {1, -1}, optional
        Sign of the structure phase relative to the substrate.
    iterations : int, optional
        Alternations of plane fit and detection.

    Returns
    -------
    ReferenceDetection
    """
    from .processing.detection import clean_structure_mask, threshold_detection

    valid = np.asarray(valid, bool)
    empty = np.zeros(valid.shape, bool)
    if valid.sum() < 4 * max(min_size_px, 16):
        return ReferenceDetection(None, empty, 0.0, None)
    phase = _unwrap(np.angle(field), valid)
    substrate = valid.copy()
    structure = empty
    resid = phase
    for _ in range(max(1, int(iterations))):
        if substrate.sum() < 16:
            return ReferenceDetection(None, empty, 0.0, None)
        # first pass: piston only (a plane through a step would tilt the
        # residual and lock Otsu onto the wrong half); then plane on substrate
        base = (
            np.median(phase[valid]) if structure is empty else _plane(phase, substrate)
        )
        resid = structure_sign * (phase - base)
        values = resid[valid]
        if np.ptp(values) <= 0:  # constant field: nothing to separate
            return ReferenceDetection(None, empty, 0.0, None)
        raw = empty.copy()
        raw[valid] = threshold_detection(values, threshold_method="otsu")
        structure = (
            clean_structure_mask(
                raw,
                remove_small=True,
                min_size=int(min_size_px),
                fill_holes=True,
                keep_largest=False,
            )
            & valid
        )
        substrate = valid & ~structure
    if not structure.any() or substrate.sum() < 16:
        return ReferenceDetection(None, empty, 0.0, None)
    sub = resid[substrate]
    noise = float(1.4826 * np.median(np.abs(sub - np.median(sub))))
    contrast = float((np.median(resid[structure]) - np.median(sub)) / max(noise, 1e-12))
    if contrast < min_contrast:
        return ReferenceDetection(None, structure, contrast, noise)
    excluded = structure
    if margin_um > 0:
        dist = ndimage.distance_transform_edt(
            ~structure, sampling=tuple(float(v) for v in pixel_um)
        )
        excluded = dist <= margin_um
    reference = valid & ~excluded
    if reference.sum() < 16:
        return ReferenceDetection(None, structure, contrast, noise)
    return ReferenceDetection(reference, structure, contrast, noise)


def _valid_box(field: np.ndarray, valid: np.ndarray, grid: Grid):
    """Crop a field to the bounding box of its valid pixels."""
    rows = np.nonzero(valid.any(axis=1))[0]
    cols = np.nonzero(valid.any(axis=0))[0]
    if rows.size == 0:
        raise ValidationError("Tile has no valid pixels.")
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    box = Grid(
        (
            grid.origin_um[0] + r0 * grid.pixel_um[0],
            grid.origin_um[1] + c0 * grid.pixel_um[1],
        ),
        grid.pixel_um,
        (r1 - r0, c1 - c0),
    )
    return field[r0:r1, c0:c1], box


def _overlap_area(a: Grid, b: Grid) -> float:
    ext = []
    for axis in (0, 1):
        lo = max(a.origin_um[axis], b.origin_um[axis])
        hi = min(
            a.origin_um[axis] + a.extent_um[axis], b.origin_um[axis] + b.extent_um[axis]
        )
        ext.append(max(hi - lo, 0.0))
    return ext[0] * ext[1]


def _field_of(product: Any) -> np.ndarray:
    field = getattr(product, "field", None)
    if field is None:
        phase = np.asarray(product.phase_rad, float)
        amp = np.sqrt(np.clip(np.asarray(product.intensity_rel, float), 0, None))
        field = amp * np.exp(1j * phase)
    return np.asarray(field).astype(np.complex128)


class MosaicBuilder:
    """Incremental fusion / stitching of the tiles of one structure and layer.

    Parameters
    ----------
    grid : Grid
        Mosaic lattice (see :func:`mosaic_grid`); same pixel size as the tiles.
    config : MosaicConfig
    occupied_um : sequence of polygons, optional
        A-priori structure footprint in stage µm (``(y, x)`` vertices); the
        reference region is its complement after ``reference_margin_um``.
    product_id : str, optional
    metadata : mapping, optional
        Extra keyword arguments for the :class:`LayerProduct` (``structure_id``,
        ``layer_index``, ``group_id``, …).
    """

    def __init__(
        self,
        grid: Grid,
        config: MosaicConfig,
        *,
        occupied_um: Sequence | None = None,
        product_id: str = "layer-product",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.grid = grid
        self.config = config
        self.product_id = product_id
        self.metadata = dict(metadata or {})
        self._a_priori = bool(occupied_um)
        if self._a_priori:
            self._reference = ~rasterise_polygons(
                occupied_um, grid, config.reference_margin_um
            )
        else:
            self._reference = np.ones(grid.shape, bool)
        # a-posteriori masks (MI-43): substrate seen by a tile's detection and
        # structure seen by any tile, on the mosaic lattice
        self._ref_seen = np.zeros(grid.shape, bool)
        self._struct_seen = np.zeros(grid.shape, bool)
        # 32-bit state (MI-42); per-tile computations run in 64 bit
        self._acc = np.zeros(grid.shape, np.complex64)
        self._wacc = np.zeros(grid.shape, np.float32)
        self._pacc = np.zeros(grid.shape, np.float32)
        self._pw = np.zeros(grid.shape, np.float32)
        self._tiles: list[_Tile] = []
        self._meta: dict[str, float] = {}
        self._timings: list[float] = []
        self._closed = False

    # ------------------------------------------------------------ helpers
    @property
    def tiles(self) -> tuple:
        """Records of the tiles added so far."""
        return tuple(dict(t.record) for t in self._tiles)

    @property
    def reference_source(self) -> str:
        """D13 key of the mosaic: a-priori mask, a-posteriori mask or whole field."""
        if self._a_priori:
            return "mask_a_priori"
        if any(
            t.record["reference_source"] == "mask_a_posteriori" for t in self._tiles
        ):
            return "mask_a_posteriori"
        return "whole_field"

    def _global_reference(self) -> np.ndarray:
        """Reference region on the whole lattice for the product-level numbers."""
        if self._a_priori or not self._ref_seen.any():
            return self._reference
        return self._ref_seen & ~self._struct_seen

    def _tile_reference(self, field: np.ndarray, valid: np.ndarray, sl):
        """Reference region, D13 source and detection record of one placed tile."""
        if self._a_priori:
            return self._reference[sl] & valid, "mask_a_priori", None
        cfg = self.config
        if cfg.reference_detection == "none":
            return valid.copy(), "whole_field", None
        det = detect_reference_region(
            field,
            valid,
            self.grid.pixel_um,
            min_contrast=cfg.detection_min_contrast,
            margin_um=cfg.reference_margin_um,
            min_size_px=cfg.detection_min_size_px,
            structure_sign=cfg.structure_sign,
        )
        record = det.to_record()
        if det.reference is None:
            return valid.copy(), "whole_field", record
        # A raised feature on a plateau looks like structure on substrate. The
        # tiles measured before are ground truth (SA6, ring first): if most of
        # the detected substrate that they already saw is structure there, the
        # detection is rejected; otherwise their structure is removed from it.
        seen_struct = self._struct_seen[sl] & det.reference
        known = (self._ref_seen[sl] | self._struct_seen[sl]) & det.reference
        record["conflict_fraction"] = float(seen_struct.sum() / max(known.sum(), 1))
        reference = det.reference & ~seen_struct
        if (
            known.sum() >= 16 and record["conflict_fraction"] > 0.5
        ) or reference.sum() < 16:
            record["accepted"] = False
            return valid.copy(), "whole_field", record
        self._ref_seen[sl] |= reference
        self._struct_seen[sl] |= valid & ~det.reference
        return reference, "mask_a_posteriori", record

    def _placement(self, tile_grid: Grid):
        """Integer placement and fractional shift of a tile on the lattice."""
        if not np.allclose(tile_grid.pixel_um, self.grid.pixel_um, rtol=1e-6, atol=0):
            raise ValidationError("Tile and mosaic pixel sizes differ; resample first.")
        k, f = [], []
        for axis in (0, 1):
            delta = (
                tile_grid.origin_um[axis] - self.grid.origin_um[axis]
            ) / self.grid.pixel_um[axis]
            ki = math.floor(delta + 0.5)
            k.append(ki)
            f.append(delta - ki)
        return k, f

    def _resample(self, field: np.ndarray, valid: np.ndarray, tile_grid: Grid):
        k, f = self._placement(tile_grid)
        if any(abs(v) > 1e-6 for v in f):
            real = ndimage.shift(field.real, f, order=3, mode="nearest")
            imag = ndimage.shift(field.imag, f, order=3, mode="nearest")
            field = real + 1j * imag
            valid = (
                ndimage.shift(valid.astype(float), f, order=0, mode="constant", cval=0)
                > 0.5
            )
            valid = ndimage.binary_erosion(valid, border_value=0)
        sl_m, sl_t = [], []
        for axis in (0, 1):
            start = max(0, k[axis])
            stop = min(self.grid.shape[axis], k[axis] + field.shape[axis])
            if stop <= start:
                return None
            sl_m.append(slice(start, stop))
            sl_t.append(slice(start - k[axis], stop - k[axis]))
        return tuple(sl_m), field[tuple(sl_t)], valid[tuple(sl_t)]

    def _mosaic_view(self, sl):
        """Current mosaic field and unwrapped phase on a slice (NaN-free)."""
        w = self._wacc[sl]
        covered = w > 0
        field = np.where(covered, self._acc[sl] / np.maximum(w, 1e-30), 0)
        pw = self._pw[sl]
        phase = np.where(pw > 0, self._pacc[sl] / np.maximum(pw, 1e-30), 0.0)
        return field, phase, covered

    def _box(self, tile: _Tile):
        g = Grid(
            (
                self.grid.origin_um[0] + tile.sl[0].start * self.grid.pixel_um[0],
                self.grid.origin_um[1] + tile.sl[1].start * self.grid.pixel_um[1],
            ),
            self.grid.pixel_um,
            tile.field.shape,
        )
        return _valid_box(tile.field, tile.valid, g)

    def _accumulate(self, tile: _Tile, sign: float = 1.0) -> None:
        sl = tile.sl
        amp = np.abs(tile.field)
        w = tile.weight * tile.valid
        self._acc[sl] += sign * w * tile.field
        self._wacc[sl] += sign * w * amp
        self._pacc[sl] += sign * w * amp * tile.phase
        self._pw[sl] += sign * w * amp

    # ------------------------------------------------------------ add
    def add_tile(
        self,
        product: TileProduct | LayerProduct,
        *,
        tile_index: int | None = None,
        planned_um: Mapping[str, float] | None = None,
        weight: float = 1.0,
        role: str | None = None,
    ) -> dict:
        """Process one tile and add it to the mosaic.

        Parameters
        ----------
        product : TileProduct or LayerProduct
            The capture (a fused speckle set may itself be a tile).
        tile_index : int, optional
            Index in the planned position list; default the arrival order.
        planned_um : mapping, optional
            Planned stage position (recorded).
        weight : float, optional
            Quality weight of the tile (M3).
        role : str, optional
            Role from the ``GroupSpec`` (``"ring"``, ``"center"``, …; recorded).

        Returns
        -------
        dict
            The tile record (also kept for the product).
        """
        if self._closed:
            raise ValidationError("The mosaic is closed.")
        t0 = time.perf_counter()
        cfg = self.config
        index = len(self._tiles) if tile_index is None else int(tile_index)
        field = _field_of(product)
        grid = product.grid
        if field.shape != grid.shape:
            raise ValidationError("Tile field does not match its grid.")
        valid = np.isfinite(field)
        band = (
            int(math.ceil(cfg.border_band_um / grid.pixel_um[0])),
            int(math.ceil(cfg.border_band_um / grid.pixel_um[1])),
        )
        if 2 * band[0] >= grid.shape[0] or 2 * band[1] >= grid.shape[1]:
            raise ValidationError("The border band removes the whole tile.")
        valid[: band[0], :] = valid[grid.shape[0] - band[0] :, :] = False
        valid[:, : band[1]] = valid[:, grid.shape[1] - band[1] :] = False
        field = np.where(valid, field, 0)
        if not self._meta:
            self._meta = {
                "z_um": product.z_um,
                "propagation_index": product.propagation_index,
                "wavelength_um": product.wavelength_um,
                "n_surround": getattr(product, "n_surround", None),
            }

        # 2. registration against the earlier tile with the largest overlap
        #    (its valid area is a rectangle; the partly covered mosaic would
        #    add coverage edges that correlate at the prior)
        registration = None
        if self._tiles:
            mov_field, mov_grid = _valid_box(field, valid, grid)
            best = max(
                self._tiles,
                key=lambda t: (
                    _overlap_area(t.box_grid, mov_grid) > 0 and not t.record["flagged"],
                    _overlap_area(t.box_grid, mov_grid),
                ),
            )
            registration = register_pair(
                mov_field,
                mov_grid,
                best.box_field,
                best.box_grid,
                pose_tolerance_um=cfg.pose_tolerance_um,
                min_confidence=cfg.min_confidence,
                min_isotropy=cfg.min_isotropy,
                min_overlap_px=cfg.min_overlap_px,
            )
            registration_ref = best.index
            grid = corrected_grid(grid, registration)

        # 3. resampling onto the lattice
        placed = self._resample(field, valid, grid)
        if placed is None:
            raise ValidationError(f"Tile {index} lies outside the mosaic grid.")
        sl, field, valid = placed
        reference, ref_source, detection = self._tile_reference(field, valid, sl)
        ref_fraction = float(reference.sum() / max(valid.sum(), 1))
        sub_grid = Grid(
            (
                self.grid.origin_um[0] + sl[0].start * self.grid.pixel_um[0],
                self.grid.origin_um[1] + sl[1].start * self.grid.pixel_um[1],
            ),
            self.grid.pixel_um,
            field.shape,
        )
        flagged = bool(registration.flagged) if registration else False
        warnings = []

        # 4. piston / tilt
        mosaic_field, mosaic_phase, covered = self._mosaic_view(sl)
        overlap = covered & valid
        fit = None
        if not self._tiles:
            if reference.sum() >= 16:
                fit = fit_piston_tilt(
                    field,
                    np.ones_like(field),
                    sub_grid,
                    mask=reference,
                    fit_tilt=cfg.fit_tilt,
                )
            anchor = "first_tile"
        elif overlap.sum() >= cfg.min_overlap_px**2 // 4:
            fit = fit_piston_tilt(
                field, mosaic_field, sub_grid, mask=overlap, fit_tilt=cfg.fit_tilt
            )
            anchor = "overlap"
        else:
            flagged = True
            anchor = "none"
            warnings.append("too little overlap for piston / tilt")
        if fit is not None:
            field = np.asarray(remove_piston_tilt(field, sub_grid, fit))

        # 5. unwrapping and k * 2 pi
        wrapped = np.angle(field)
        unwrap_info: dict[str, Any] = {
            "method": cfg.unwrap if cfg.phase_from == "tiles" else "none"
        }
        phase = wrapped
        if cfg.phase_from == "tiles" and cfg.unwrap == "fast2d":
            phase = _unwrap(wrapped, valid)
            if self._tiles and overlap.sum() > 0:
                diff = (phase - mosaic_phase)[overlap]
                k = int(np.round(np.median(diff) / _TWO_PI))
                phase = phase - _TWO_PI * k
                bad = float(np.mean(np.abs(diff - _TWO_PI * k) > 0.5 * math.pi))
                consistent = bad <= cfg.consistency_max_fraction
                unwrap_info.update(
                    {
                        "k_offset": k,
                        "inconsistent_fraction": bad,
                        "consistent": consistent,
                    }
                )
                if not consistent:
                    flagged = True
                    warnings.append("unwrapping inconsistent with the mosaic")
            else:
                level = phase[reference].mean() if reference.any() else 0.0
                k = int(np.round(level / _TWO_PI))
                phase = phase - _TWO_PI * k
                unwrap_info.update({"k_offset": k, "consistent": True})

        if (
            cfg.min_reference_fraction is not None
            and ref_fraction < cfg.min_reference_fraction
        ):
            warnings.append(f"reference fraction {ref_fraction:.3f} below the minimum")
        noise = None
        if reference.sum() >= 16:
            noise = reference_noise(field, reference, detrend="plane").sigma_rad

        tile = _Tile(
            index=index,
            product_id=str(product.product_id),
            sl=sl,
            field=field.astype(np.complex64),
            phase=np.asarray(phase, np.float32),
            valid=valid,
            reference=reference,
            weight=float(weight),
            noise_rad=noise,
            record={
                "product_id": str(product.product_id),
                "tile_index": index,
                "planned_um": dict(planned_um) if planned_um else None,
                "actual_um": dict(getattr(product, "position_actual_um", None) or {})
                or None,
                "registration": (
                    registration.to_record()
                    if registration
                    else {"method": "first_tile"}
                ),
                "piston_rad": fit.piston_rad if fit else None,
                "tilt_rad_per_um": list(fit.tilt_rad_per_um) if fit else None,
                "anchor": anchor,
                "residual_rad": fit.residual_rad if fit else None,
                "border_band_um": cfg.border_band_um,
                "reference_source": ref_source,
                "reference_fraction": (
                    ref_fraction if ref_source != "whole_field" else 0.0
                ),
                "unwrap": unwrap_info,
                "flagged": flagged,
                "role": role,
                "weight": float(weight),
                "phase_noise_rad": noise,
                "warnings": warnings,
            },
        )
        if registration is not None:
            tile.record["registration"]["reference_tile"] = registration_ref
        if detection is not None:
            tile.record["reference_detection"] = detection
        tile.box_field, tile.box_grid = self._box(tile)
        self._tiles.append(tile)
        self._accumulate(tile)
        dt = time.perf_counter() - t0
        self._timings.append(dt)
        tile.record["time_s"] = dt
        return dict(tile.record)

    # ------------------------------------------------------------ close
    def _global_adjustment(self) -> np.ndarray:
        """One piston per tile from all overlaps; anchors on the reference region."""
        n = len(self._tiles)
        rows, rhs, wts = [], [], []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self._tiles[i], self._tiles[j]
                sy = slice(
                    max(a.sl[0].start, b.sl[0].start), min(a.sl[0].stop, b.sl[0].stop)
                )
                sx = slice(
                    max(a.sl[1].start, b.sl[1].start), min(a.sl[1].stop, b.sl[1].stop)
                )
                if sy.stop - sy.start < 1 or sx.stop - sx.start < 1:
                    continue

                def local(t, s0=sy, s1=sx):
                    return (
                        slice(s0.start - t.sl[0].start, s0.stop - t.sl[0].start),
                        slice(s1.start - t.sl[1].start, s1.stop - t.sl[1].start),
                    )

                la, lb = local(a), local(b)
                both = a.valid[la] & b.valid[lb]
                if both.sum() < self.config.min_overlap_px:
                    continue
                if self.config.phase_from == "tiles":
                    d = float(np.mean((a.phase[la] - b.phase[lb])[both]))
                else:
                    d = float(
                        np.angle(np.sum((a.field[la] * np.conj(b.field[lb]))[both]))
                    )
                row = np.zeros(n)
                row[i], row[j] = 1.0, -1.0
                rows.append(row)
                rhs.append(d)
                wts.append(math.sqrt(both.sum()))
        anchors = 0
        for i, t in enumerate(self._tiles):
            ref = t.reference
            frac = ref.sum() / max(t.valid.sum(), 1)
            if (
                t.record["reference_source"] == "whole_field"
                or frac < self.config.anchor_min_fraction
            ):
                continue
            if self.config.phase_from == "tiles":
                level = float(np.mean(t.phase[ref]))
            else:
                level = float(np.angle(np.sum(t.field[ref])))
            row = np.zeros(n)
            row[i] = 1.0
            rows.append(row)
            rhs.append(level)
            wts.append(10.0 * math.sqrt(ref.sum()))
            t.record["anchor"] = "reference_region"
            anchors += 1
        if anchors == 0:
            row = np.zeros(n)
            row[0] = 1.0
            rows.append(row)
            rhs.append(0.0)
            wts.append(1e3)
        a = np.array(rows) * np.array(wts)[:, None]
        b = np.array(rhs) * np.array(wts)
        corr, *_ = np.linalg.lstsq(a, b, rcond=None)
        resid = np.array(rows) @ corr - np.array(rhs)
        for i, t in enumerate(self._tiles):
            used = np.array(rows)[:, i] != 0
            t.record["residual_rad"] = (
                float(np.sqrt(np.mean(resid[used] ** 2))) if used.any() else None
            )
            t.record["global_piston_rad"] = float(corr[i])
        return corr

    def close(self, *, include_field: bool = True) -> LayerProduct:
        """Run the global adjustment and return the :class:`LayerProduct`."""
        if not self._tiles:
            raise ValidationError("No tiles were added.")
        t0 = time.perf_counter()
        cfg = self.config
        corr = self._global_adjustment()
        self._acc[:] = 0
        self._wacc[:] = 0
        self._pacc[:] = 0
        self._pw[:] = 0
        count = np.zeros(self.grid.shape, np.int32)
        for i, t in enumerate(self._tiles):
            t.field = (t.field * np.exp(-1j * corr[i])).astype(np.complex64)
            t.phase = (t.phase - corr[i]).astype(np.float32)
            if t.record["piston_rad"] is not None:
                t.record["piston_rad"] = float(t.record["piston_rad"] + corr[i])
            self._accumulate(t)
            count[t.sl] += t.valid
        field, phase_tiles, covered = self._mosaic_view((slice(None), slice(None)))
        valid = covered & (count >= cfg.min_count)
        if cfg.phase_from == "field":
            phase = np.angle(field)
            if cfg.unwrap == "fast2d" and valid.any():
                phase = _unwrap(phase, valid)
                ref = valid & self._global_reference()
                level = phase[ref].mean() if ref.any() else phase[valid].mean()
                phase = phase - _TWO_PI * np.round(level / _TWO_PI)
        elif cfg.combine == "median":
            stack = np.full((len(self._tiles),) + self.grid.shape, np.nan, np.float32)
            for i, t in enumerate(self._tiles):
                stack[i][t.sl] = np.where(t.valid, t.phase, np.nan)
            with warnings.catch_warnings():  # uncovered pixels are all-NaN
                warnings.simplefilter("ignore", RuntimeWarning)
                phase = np.nan_to_num(np.nanmedian(stack, axis=0))
        else:
            phase = phase_tiles
        intensity = np.where(valid, np.abs(field) ** 2, 0.0)
        phase = np.where(valid, phase, 0.0)
        ref = valid & self._global_reference()
        noise_after = (
            reference_noise(field, ref, detrend="plane").sigma_rad
            if ref.sum() >= 16
            else None
        )
        noise_before = [t.noise_rad for t in self._tiles if t.noise_rad is not None]
        records = [
            TileRecord.from_dict({k: v for k, v in t.record.items()})
            for t in self._tiles
        ]
        n_flagged = sum(1 for t in self._tiles if t.record["flagged"])
        timings = {
            "tiles": float(sum(self._timings)),
            "close": time.perf_counter() - t0,
        }
        quality = {
            "mode": cfg.mode,
            "n_tiles": len(self._tiles),
            "n_flagged": n_flagged,
            "unwrap_method": cfg.unwrap,
            "coverage_min": int(count[valid].min()) if valid.any() else 0,
            "coverage_max": int(count.max()),
            "phase_noise_rad": noise_after,
            "phase_noise_tiles_median_rad": (
                float(np.median(noise_before)) if noise_before else None
            ),
            "reference_source": self.reference_source,
            "config": cfg.to_dict(),
        }
        meta = dict(self.metadata)
        meta.setdefault("group_kind", "speckle" if cfg.mode == "fusion" else "layer")
        product = LayerProduct(
            phase_rad=phase.astype(np.float32),
            intensity_rel=intensity.astype(np.float32),
            valid=valid,
            grid=self.grid,
            tiles=tuple(records),
            z_um=self._meta["z_um"],
            propagation_index=self._meta["propagation_index"],
            wavelength_um=self._meta["wavelength_um"],
            n_surround=self._meta.get("n_surround"),
            product_id=self.product_id,
            field=field.astype(np.complex64) if include_field else None,
            member_ids=tuple(t.product_id for t in self._tiles),
            quality=quality,
            timings_s=timings,
            warnings=tuple(w for t in self._tiles for w in t.record["warnings"]),
            **meta,
        )
        self._closed = True
        return product


def build_mosaic(
    products: Sequence[TileProduct | LayerProduct],
    config: MosaicConfig,
    *,
    grid: Grid | None = None,
    occupied_um: Sequence | None = None,
    margin_um: float | None = None,
    **kwargs: Any,
) -> LayerProduct:
    """Fuse or stitch a list of products in the given order (convenience).

    Parameters
    ----------
    products : sequence
        Tiles in measurement order (ring first, then inwards for stitching).
    config : MosaicConfig
    grid : Grid, optional
        Mosaic lattice; default covers all tiles plus the pose tolerance.
    occupied_um : sequence of polygons, optional
    margin_um : float, optional
        Margin of the default grid; default ``config.pose_tolerance_um``.
    **kwargs
        Passed to :class:`MosaicBuilder`.

    Returns
    -------
    LayerProduct
    """
    if not products:
        raise ValidationError("No products given.")
    if grid is None:
        first = products[0].grid
        grid = mosaic_grid(
            [p.grid.centre_um for p in products],
            first.shape,
            first.pixel_um,
            config.pose_tolerance_um if margin_um is None else margin_um,
        )
    builder = MosaicBuilder(grid, config, occupied_um=occupied_um, **kwargs)
    for i, p in enumerate(products):
        builder.add_tile(
            p,
            tile_index=(
                getattr(p, "tile_index", None)
                if getattr(p, "tile_index", None) is not None
                else i
            ),
        )
    return builder.close()
