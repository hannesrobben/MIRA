"""Group plug-in for the DHMConnect service on the DHM-PC (MI-60, G5-new).

The service opens one :class:`GroupSession` per ``open_group(GroupSpec)``
(N2), passes every tile product to :meth:`GroupSession.add_tile` while the
stage moves to the next position, and calls :meth:`GroupSession.close` on
``close_group`` to get the stitched / fused
:class:`~mira.products.LayerProduct` (N1). The session wraps the shared
fusion / stitching pipeline :class:`mira.mosaic.MosaicBuilder` (M4).

Design rules:

* No network, no file access, no import of ``dhmconnect``: inputs are the
  ``GroupSpec`` mapping (JSON), plain parameters and tile products or tile
  messages (JSON header + NumPy buffers); outputs are records (JSON) and the
  ``LayerProduct`` or its message. Nothing needs to be pickled, so the
  session can live in a worker process (D8, M8: CPU) and receive messages
  through any transport.
* Thresholds without decided values (M7) are not defaulted: the service
  passes ``pose_tolerance_um``, ``min_confidence`` and ``min_isotropy`` from
  its parameter set (:class:`PluginParameters`).
* Fusion or stitching (M4: one pipeline, two configurations): when the
  footprint plus margin fits into the usable area of every planned tile, the
  group is a speckle set and is fused; otherwise it is stitched. The service
  can force one with ``PluginParameters.mode``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .. import __version__
from ..mosaic import MosaicBuilder, MosaicConfig, mosaic_grid
from ..products import LayerProduct, TileProduct
from ..utils.exceptions import ValidationError

__all__ = [
    "PLUGIN_NAME",
    "PluginParameters",
    "GroupSession",
    "describe",
    "open_group",
    "run_group",
]

PLUGIN_NAME = "mira.mosaic"
_GROUP_SPEC_MAJOR = "1"
_SUPPORTED_UNWRAP = ("fast2d", "none")
_SUPPORTED_KINDS = ("layer", "single")


@dataclass(frozen=True)
class PluginParameters:
    """Parameters of the plug-in that the service takes from its parameter set.

    Parameters
    ----------
    pose_tolerance_um : float
        Search radius of the registration and margin of the mosaic grid
        (to be measured per objective, T104; no default).
    min_confidence : float
        Registration confidence below which the stage position is kept (M7;
        no default).
    min_isotropy : float
        Minimum gradient isotropy of the registration (M7; no default).
    optical_margin_um : float, optional
        Margin of the reference mask added to ``pose_tolerance_um`` (D11).
    mode : {"fusion", "stitching"} or None, optional
        Force a configuration; None decides from the ``GroupSpec``.
    include_field : bool, optional
        Put the complex field into the product (N1: on request only).
    min_overlap_px : int, optional
        Minimum overlap per axis for registration and global adjustment.
    config : mapping, optional
        Further :class:`~mira.mosaic.MosaicConfig` fields (``fit_tilt``,
        ``combine``, ``min_count``, ``detection_min_contrast``, …).
    """

    pose_tolerance_um: float
    min_confidence: float
    min_isotropy: float
    optical_margin_um: float = 0.0
    mode: str | None = None
    include_field: bool = False
    min_overlap_px: int = 16
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("pose_tolerance_um", "optical_margin_um"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value >= 0):
                raise ValidationError(f"{name} must be finite and >= 0.")
        if self.mode not in (None, "fusion", "stitching"):
            raise ValidationError("mode must be None, 'fusion' or 'stitching'.")
        reserved = {
            "mode",
            "border_band_um",
            "pose_tolerance_um",
            "min_confidence",
            "min_isotropy",
            "unwrap",
            "reference_margin_um",
            "min_overlap_px",
        }
        clash = reserved & set(self.config)
        if clash:
            raise ValidationError(
                f"config must not set {sorted(clash)}; use the GroupSpec or the "
                "named parameters."
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PluginParameters:
        """Build from a JSON mapping (unknown keys are an error)."""
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValidationError(f"Unknown plug-in parameters: {sorted(unknown)}.")
        return cls(**dict(data))

    def to_dict(self) -> dict:
        """JSON-ready mapping."""
        out = {k: getattr(self, k) for k in self.__dataclass_fields__}
        out["config"] = dict(self.config)
        return out


def describe() -> dict:
    """Capabilities of the plug-in for the service's ``describe`` reply.

    Returns
    -------
    dict
    """
    return {
        "name": PLUGIN_NAME,
        "version": __version__,
        "group_spec_versions": [f"{_GROUP_SPEC_MAJOR}.x"],
        "group_kinds": list(_SUPPORTED_KINDS),
        "modes": ["fusion", "stitching"],
        "unwrap_methods": list(_SUPPORTED_UNWRAP),
        "required_parameters": ["pose_tolerance_um", "min_confidence", "min_isotropy"],
        "output": "LayerProduct 1.0",
    }


def _yx(position: Mapping[str, float]) -> tuple[float, float]:
    try:
        return float(position["y"]), float(position["x"])
    except (KeyError, TypeError) as exc:
        raise ValidationError("A planned position needs 'x' and 'y' in µm.") from exc


def _choose_mode(
    spec: Mapping[str, Any],
    centres: Sequence[tuple[float, float]],
    tile_shape: Sequence[int],
    pixel_um: Sequence[float],
    border_band_um: float,
    margin_um: float,
) -> str:
    """Fusion if the footprint fits into the usable area of every tile."""
    polygons = spec["structure"].get("occupied_um") or []
    vertices = [np.asarray(p, float) for p in polygons if len(p) >= 3]
    if not vertices:
        return "stitching"
    pts = np.concatenate(vertices)
    lo = pts.min(axis=0) - margin_um
    hi = pts.max(axis=0) + margin_um
    half = [0.5 * (tile_shape[a] - 1) * pixel_um[a] - border_band_um for a in (0, 1)]
    for cy, cx in centres:
        for axis, c in ((0, cy), (1, cx)):
            if lo[axis] < c - half[axis] or hi[axis] > c + half[axis]:
                return "stitching"
    return "fusion"


class GroupSession:
    """Incremental fusion / stitching of one group (``open_group`` … ``close_group``).

    Parameters
    ----------
    group_spec : mapping
        ``GroupSpec`` 1.x message (from nanofactorysystem, N2).
    params : PluginParameters or mapping
        Plug-in parameters from the service's parameter set.
    tile_shape : sequence of int
        ``(ny, nx)`` of one tile product.
    pixel_um : sequence of float
        Pixel size ``(y, x)`` of the tile products.
    group_id : str, optional
        Group id assigned by the service (recorded in the product).
    product_id : str, optional
        Id of the resulting ``LayerProduct``; default ``"<group_id>-layer"``.
    metadata : mapping, optional
        Further ``LayerProduct`` fields. The contract requires
        ``experiment_uuid``, ``context_id``, ``job_id``, ``parameter_set_id``
        and ``config_hash`` (strings) and ``n_surround`` (from the tiles'
        capture context); the service passes the ids it owns here.
    """

    def __init__(
        self,
        group_spec: Mapping[str, Any],
        params: PluginParameters | Mapping[str, Any],
        *,
        tile_shape: Sequence[int],
        pixel_um: Sequence[float],
        group_id: str | None = None,
        product_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(params, PluginParameters):
            params = PluginParameters.from_dict(params)
        self.params = params
        spec = dict(group_spec)
        if spec.get("message_type") != "GroupSpec":
            raise ValidationError("group_spec is not a GroupSpec message.")
        version = str(spec.get("schema_version", ""))
        if version.split(".")[0] != _GROUP_SPEC_MAJOR:
            raise ValidationError(
                f"GroupSpec {version!r} is not supported (need {_GROUP_SPEC_MAJOR}.x)."
            )
        kind = spec.get("group_kind")
        if kind not in _SUPPORTED_KINDS:
            raise ValidationError(
                f"group_kind {kind!r} is not a MIRA job (supported: "
                f"{', '.join(_SUPPORTED_KINDS)}); background groups stay in the service."
            )
        stitching = spec.get("stitching") or {}
        unwrap = stitching.get("unwrap", "fast2d")
        if unwrap not in _SUPPORTED_UNWRAP:
            raise ValidationError(
                f"unwrap method {unwrap!r} is not implemented in MIRA "
                f"(supported: {', '.join(_SUPPORTED_UNWRAP)}; K11-T decides the others)."
            )
        positions = spec.get("positions") or []
        if not positions:
            raise ValidationError("The GroupSpec has no positions.")
        self._positions: dict[int, Mapping[str, Any]] = {}
        for pos in positions:
            index = int(pos["tile_index"])
            if index in self._positions:
                raise ValidationError(f"tile_index {index} appears twice.")
            self._positions[index] = pos
        self.spec = spec
        self.tile_shape = (int(tile_shape[0]), int(tile_shape[1]))
        self.pixel_um = (float(pixel_um[0]), float(pixel_um[1]))
        border_band = float(stitching.get("border_band_um", 0.0))
        margin = params.pose_tolerance_um + params.optical_margin_um
        centres = [_yx(p["planned_um"]) for p in positions]
        self.mode = params.mode or (
            "fusion"
            if kind == "single"
            else _choose_mode(
                spec, centres, self.tile_shape, self.pixel_um, border_band, margin
            )
        )
        factory = (
            MosaicConfig.fusion if self.mode == "fusion" else MosaicConfig.stitching
        )
        self.config = factory(
            border_band_um=border_band,
            pose_tolerance_um=params.pose_tolerance_um,
            min_confidence=params.min_confidence,
            min_isotropy=params.min_isotropy,
            unwrap=unwrap,
            reference_margin_um=margin,
            min_overlap_px=params.min_overlap_px,
            **dict(params.config),
        )
        self.grid = mosaic_grid(
            centres, self.tile_shape, self.pixel_um, params.pose_tolerance_um
        )
        self.group_id = group_id
        structure = spec["structure"]
        meta = {
            "structure_id": structure.get("structure_id"),
            "layer_index": spec.get("layer_index"),
            "group_id": group_id,
            "group_kind": kind,
        }
        meta.update(dict(metadata or {}))
        self._builder = MosaicBuilder(
            self.grid,
            self.config,
            occupied_um=structure.get("occupied_um") or None,
            product_id=product_id or f"{group_id or 'group'}-layer",
            metadata=meta,
        )
        self._added: dict[int, dict] = {}
        self._timings: list[dict] = []
        self._opened = time.perf_counter()
        self._product: LayerProduct | None = None

    # ------------------------------------------------------------ status
    @property
    def n_planned(self) -> int:
        """Number of planned positions."""
        return len(self._positions)

    @property
    def n_added(self) -> int:
        """Number of tiles added so far."""
        return len(self._added)

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has run."""
        return self._product is not None

    @property
    def timings(self) -> list[dict]:
        """Per tile: ``tile_index``, ``mosaic_s`` (pipeline) and ``total_s``."""
        return [dict(t) for t in self._timings]

    def status(self) -> dict:
        """JSON-ready progress of the group."""
        return {
            "plugin": PLUGIN_NAME,
            "mode": self.mode,
            "n_planned": self.n_planned,
            "n_added": self.n_added,
            "n_flagged": sum(1 for r in self._added.values() if r["flagged"]),
            "closed": self.closed,
        }

    # ------------------------------------------------------------ tiles
    def add_tile(
        self,
        tile: TileProduct | tuple[Mapping[str, Any], Sequence],
        *,
        weight: float = 1.0,
    ) -> dict:
        """Add one tile product (or a ``ProductSet`` message) to the group.

        A tile whose ``tile_index`` was already added with the same
        ``product_id`` is ignored and its earlier record returned (``measure``
        is idempotent by ``capture_id``); a different product for the same
        index is an error.

        Parameters
        ----------
        tile : TileProduct or (header, buffers)
            The tile; its ``tile_index`` must be a planned position.
        weight : float, optional
            Quality weight of the tile (M3).

        Returns
        -------
        dict
            The tile record of the mosaic plus ``time_s`` and
            ``duplicate``.
        """
        if self.closed:
            raise ValidationError("The group is closed.")
        t0 = time.perf_counter()
        if not isinstance(tile, TileProduct):
            header, buffers = tile
            tile = TileProduct.from_message(header, buffers)
        if tile.tile_index is None:
            raise ValidationError("The tile product has no tile_index.")
        index = int(tile.tile_index)
        if index not in self._positions:
            raise ValidationError(f"tile_index {index} is not a planned position.")
        if index in self._added:
            earlier = self._added[index]
            if earlier["product_id"] != str(tile.product_id):
                raise ValidationError(
                    f"tile_index {index} was already added with another product."
                )
            return {**earlier, "duplicate": True}
        if tuple(tile.grid.shape) != self.tile_shape:
            raise ValidationError(
                f"Tile shape {tuple(tile.grid.shape)} differs from {self.tile_shape}."
            )
        if self.group_id is not None and tile.group_id not in (None, self.group_id):
            raise ValidationError(
                f"Tile belongs to group {tile.group_id!r}, not {self.group_id!r}."
            )
        pos = self._positions[index]
        record = self._builder.add_tile(
            tile,
            tile_index=index,
            planned_um=pos["planned_um"],
            weight=weight,
            role=pos.get("role"),
        )
        total = time.perf_counter() - t0
        self._timings.append(
            {"tile_index": index, "mosaic_s": record["time_s"], "total_s": total}
        )
        record = {**record, "duplicate": False}
        self._added[index] = record
        return record

    # ------------------------------------------------------------ close
    def close(self) -> LayerProduct:
        """Run the global adjustment and return the ``LayerProduct``.

        Missing planned positions and a short group (fewer tiles than the
        ``GroupSpec`` ``min_positions``) are recorded as warnings, never
        silently ignored.
        """
        if self._product is not None:
            return self._product
        if not self._added:
            raise ValidationError("No tiles were added to the group.")
        product = self._builder.close(include_field=self.params.include_field)
        warnings = list(product.warnings)
        missing = sorted(set(self._positions) - set(self._added))
        if missing:
            warnings.append(f"missing planned positions: {missing}")
        min_positions = (self.spec.get("stitching") or {}).get("min_positions")
        if min_positions is not None and self.n_added < int(min_positions):
            warnings.append(
                f"{self.n_added} tiles is below min_positions = {min_positions}"
            )
        quality = dict(product.quality)
        quality.update(
            {
                "plugin": PLUGIN_NAME,
                "plugin_version": __version__,
                "n_planned": self.n_planned,
                "missing_positions": missing,
            }
        )
        # contract: timings_s holds numbers only; per-tile times are in the
        # tile records (time_s) and in :attr:`timings`
        timings = dict(product.timings_s)
        timings["group"] = time.perf_counter() - self._opened
        versions = dict(product.versions)
        versions.setdefault("mira", __version__)
        self._product = replace(
            product,
            quality=quality,
            timings_s=timings,
            versions=versions,
            warnings=tuple(warnings),
        )
        return self._product

    def close_message(self) -> tuple[dict, list[tuple[str, np.ndarray]]]:
        """:meth:`close` and serialise as a ``LayerProduct`` 1.0 message."""
        product = self.close()
        return product.to_message(include_field=self.params.include_field)


def open_group(
    group_spec: Mapping[str, Any],
    params: PluginParameters | Mapping[str, Any],
    **kwargs: Any,
) -> GroupSession:
    """Open a :class:`GroupSession` (the plug-in's ``open_group``).

    Parameters
    ----------
    group_spec : mapping
    params : PluginParameters or mapping
    **kwargs
        ``tile_shape``, ``pixel_um`` and the optional keywords of
        :class:`GroupSession`.

    Returns
    -------
    GroupSession
    """
    return GroupSession(group_spec, params, **kwargs)


def run_group(
    group_spec: Mapping[str, Any],
    params: PluginParameters | Mapping[str, Any],
    tiles: Sequence[TileProduct | tuple[Mapping[str, Any], Sequence]],
    **kwargs: Any,
) -> tuple[dict, list[tuple[str, np.ndarray]]]:
    """One-shot helper for a worker pool: open, add all tiles, close.

    Parameters
    ----------
    group_spec : mapping
    params : PluginParameters or mapping
    tiles : sequence
        Tile products or ``ProductSet`` messages in measurement order.
    **kwargs
        Passed to :class:`GroupSession`.

    Returns
    -------
    header : dict
    buffers : list of (str, numpy.ndarray)
        The ``LayerProduct`` 1.0 message.
    """
    session = GroupSession(group_spec, params, **kwargs)
    for tile in tiles:
        session.add_tile(tile)
    return session.close_message()
