"""Stage-to-image geometry (MI-11).

A capture taken with the stage at position ``s`` (µm, ``(y, x)``) shows the
sample point ``p`` at the pixel ``q = (row, col)``:

.. math::

    p = s + d + J \\, (q - c)

with ``c`` the image centre in pixels, ``J`` the 2 × 2 Jacobian in µm per
pixel (rotation, scale, shear and axis flips in one matrix) and ``d`` the
offset in µm between the stage position and the sample point imaged at the
image centre (writing field ↔ camera / DHM). One model exists per objective
and per device (``"camera"`` or ``"dhm"``).

The versioned calibration file of nanofactorysystem T104 fills ``J`` and
``d`` (fit: MI-41). Until one exists :meth:`StageToImage.nominal` gives the
fallback of decision G10 SA2: rotation 0, the nominal pixel size and a
generous pose tolerance stated by the caller.

Relation to nanofactorysystem ``tools/transform.Transform`` (camera case,
reviewed for reuse): ``Transform`` maps image ``(x, y)`` pixels relative to
the image centre to object ``(x, y)`` µm with the matrix ``P2D`` and an
offset ``-P[:2, 2]``. It is the same affine model in ``(x, y)`` order;
:meth:`StageToImage.from_xy_matrix` converts it, so the camera calibration of
nanofactorysystem can be used without importing that package.

Calibration file format (``StageImageCalibration`` 1.0, JSON; contract in
``dhmconnect/schemas`` since DL-A11, example in ``tests/contracts``)::

    {
      "message_type": "StageImageCalibration",
      "schema_version": "1.0",
      "calibration_id": "cal-63x-dhm-2026-11-19",
      "created_utc": "2026-11-19T10:00:00Z",
      "objective": "Zeiss 63x",
      "device": "dhm",
      "jacobian_um_per_px": [[jyy, jyx], [jxy, jxx]],
      "offset_um": [dy, dx],
      "residuals_um": {"rms": 0.05, "max": 0.1, "n_points": 9},
      "pose_tolerance_um": 1.0,
      "source": "nanofactorysystem T104 / mira.calibration_marks"
    }

``jacobian_um_per_px[i][j]`` is the change of stage coordinate ``i``
(0 = y, 1 = x) per pixel along image axis ``j`` (0 = row, 1 = column).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Any

import numpy as np

from .products import Grid, _freeze, _thaw, _to_jsonable, _utc_now
from .utils.exceptions import ValidationError

__all__ = [
    "CALIBRATION_SCHEMA_VERSION",
    "DEVICES",
    "StageToImage",
    "AffineFit",
    "fit_stage_to_image",
    "load_calibration",
    "save_calibration",
]

CALIBRATION_SCHEMA_VERSION = "1.0"
DEVICES = ("camera", "dhm")
_ROTATION_FREE_ATOL = 1e-9


def _matrix(value: Any, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a 2 x 2 matrix of numbers.") from exc
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValidationError(f"{name} must be a finite 2 x 2 matrix; got {value!r}.")
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise ValidationError(f"{name} is singular; got {value!r}.")
    return matrix


def _vector(value: Any, name: str) -> np.ndarray:
    try:
        vec = np.asarray(value, dtype=float).reshape(2)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a pair (y, x); got {value!r}.") from exc
    if not np.all(np.isfinite(vec)):
        raise ValidationError(f"{name} must be finite; got {value!r}.")
    return vec


def _stage_yx(stage_um: Any) -> np.ndarray:
    """Accept ``{"y", "x"}`` mappings or ``(y, x)`` pairs."""
    if isinstance(stage_um, Mapping):
        try:
            return np.array([float(stage_um["y"]), float(stage_um["x"])])
        except KeyError as exc:
            raise ValidationError(f"stage position lacks {exc.args[0]!r}.") from exc
    return _vector(stage_um, "stage_um")


def _centre_px(shape: Sequence[int]) -> np.ndarray:
    ny, nx = (int(v) for v in shape)
    if ny < 1 or nx < 1:
        raise ValidationError(f"shape must be positive; got {(ny, nx)}.")
    return np.array([0.5 * (ny - 1), 0.5 * (nx - 1)])


@dataclass(frozen=True)
class StageToImage:
    """Affine map between sample coordinates and image pixels of one device.

    Parameters
    ----------
    jacobian_um_per_px : array_like, shape (2, 2)
        µm of stage ``(y, x)`` per pixel along ``(row, col)``.
    offset_um : array_like, shape (2,)
        Sample point at the image centre minus the stage position, ``(y, x)``.
    objective : str
        Objective name, e.g. ``"Zeiss 63x"``.
    device : str
        ``"camera"`` or ``"dhm"``.
    pose_tolerance_um : float
        How far a stage pose may deviate from the model (search radius of the
        image registration, MI-20).
    calibrated : bool
        False for the nominal fallback (G10 SA2).
    calibration_id : str or None
        Id of the calibration file; None for the fallback.
    created_utc : str or None
        Creation time of the calibration.
    residuals_um : mapping
        Fit residuals (``rms``, ``max``, ``n_points``).
    source : str or None
        Free text naming the procedure.
    """

    jacobian_um_per_px: np.ndarray
    offset_um: np.ndarray
    objective: str
    device: str
    pose_tolerance_um: float
    calibrated: bool = True
    calibration_id: str | None = None
    created_utc: str | None = None
    residuals_um: Mapping[str, Any] = dc_field(default_factory=dict)
    source: str | None = None

    def __post_init__(self) -> None:
        jac = _matrix(self.jacobian_um_per_px, "jacobian_um_per_px")
        jac.setflags(write=False)
        off = _vector(self.offset_um, "offset_um")
        off.setflags(write=False)
        object.__setattr__(self, "jacobian_um_per_px", jac)
        object.__setattr__(self, "offset_um", off)
        if self.device not in DEVICES:
            raise ValidationError(
                f"device must be one of {DEVICES}; got {self.device!r}."
            )
        tol = float(self.pose_tolerance_um)
        if not (math.isfinite(tol) and tol >= 0):
            raise ValidationError(f"pose_tolerance_um must be >= 0; got {tol}.")
        object.__setattr__(self, "pose_tolerance_um", tol)
        object.__setattr__(self, "residuals_um", _freeze(dict(self.residuals_um or {})))

    # ------------------------------------------------------------ builders
    @classmethod
    def nominal(
        cls,
        pixel_um: Sequence[float],
        *,
        objective: str,
        device: str,
        pose_tolerance_um: float,
    ) -> StageToImage:
        """Fallback without calibration: rotation 0, nominal pixel size.

        Parameters
        ----------
        pixel_um : sequence of float
            Nominal pixel size ``(y, x)`` in µm (from the parameter set, never
            from Koala, B17).
        objective, device : str
            See the class.
        pose_tolerance_um : float
            Generous tolerance for the uncalibrated case; the caller states it
            (the measured values follow from T104).

        Returns
        -------
        StageToImage
        """
        py, px = (float(v) for v in pixel_um)
        if not (py > 0 and px > 0):
            raise ValidationError(f"pixel_um must be positive; got {pixel_um!r}.")
        return cls(
            jacobian_um_per_px=np.diag([py, px]),
            offset_um=np.zeros(2),
            objective=objective,
            device=device,
            pose_tolerance_um=pose_tolerance_um,
            calibrated=False,
            source="nominal fallback (G10 SA2)",
        )

    @classmethod
    def from_parameters(
        cls,
        *,
        pixel_um: Sequence[float],
        rotation_rad: float = 0.0,
        shear: float = 0.0,
        flip_y: bool = False,
        flip_x: bool = False,
        offset_um: Sequence[float] = (0.0, 0.0),
        **kwargs: Any,
    ) -> StageToImage:
        """Build the model from rotation, scale, shear and flips.

        ``J = R(rotation) · [[py, shear · px], [0, px]] · F`` with ``F`` the
        diagonal flip matrix. A positive rotation turns the image axes
        counter-clockwise in the ``(x, y)`` plane.

        Parameters
        ----------
        pixel_um : sequence of float
            Pixel size ``(y, x)`` in µm.
        rotation_rad : float, optional
            Rotation of the image axes against the stage axes.
        shear : float, optional
            Dimensionless shear of the column axis into the row axis.
        flip_y, flip_x : bool, optional
            Mirror the row / column axis.
        offset_um : sequence of float, optional
            See the class.
        **kwargs
            ``objective``, ``device``, ``pose_tolerance_um`` and the optional
            metadata of the class.

        Returns
        -------
        StageToImage
        """
        py, px = (float(v) for v in pixel_um)
        c, s = math.cos(rotation_rad), math.sin(rotation_rad)
        # (y, x) order: rotating (x, y) by +a gives y' = s x + c y, x' = c x - s y.
        rot = np.array([[c, s], [-s, c]])
        upper = np.array([[py, shear * px], [0.0, px]])
        flips = np.diag([-1.0 if flip_y else 1.0, -1.0 if flip_x else 1.0])
        return cls(
            jacobian_um_per_px=rot @ upper @ flips, offset_um=offset_um, **kwargs
        )

    @classmethod
    def from_xy_matrix(
        cls,
        matrix_xy_um_per_px: Sequence[Sequence[float]],
        offset_xy_um: Sequence[float] = (0.0, 0.0),
        **kwargs: Any,
    ) -> StageToImage:
        """Convert a model given in ``(x, y)`` order.

        nanofactorysystem ``Transform`` uses ``P2D`` (µm per pixel, rows =
        object ``x, y``, columns = image ``x, y``) and the offset
        ``-P[:2, 2]`` in ``(x, y)``. Image ``y`` is the row axis.

        Parameters
        ----------
        matrix_xy_um_per_px : 2 × 2 sequence
            Jacobian in ``(x, y)`` order.
        offset_xy_um : sequence of float, optional
            Offset in ``(x, y)`` order.
        **kwargs
            Remaining class parameters.

        Returns
        -------
        StageToImage
        """
        m = _matrix(matrix_xy_um_per_px, "matrix_xy_um_per_px")
        swap = np.array([[0.0, 1.0], [1.0, 0.0]])
        off = _vector(offset_xy_um, "offset_xy_um")
        return cls(jacobian_um_per_px=swap @ m @ swap, offset_um=off[::-1], **kwargs)

    # ----------------------------------------------------------- mapping
    @property
    def pixel_um(self) -> tuple[float, float]:
        """Pixel size ``(y, x)``: length of the Jacobian columns in µm."""
        norms = np.linalg.norm(self.jacobian_um_per_px, axis=0)
        return (float(norms[0]), float(norms[1]))

    @property
    def rotation_rad(self) -> float:
        """Direction of the row axis against the stage y axis.

        Equals ``rotation_rad`` of :meth:`from_parameters` without a row flip
        (with ``flip_y`` it differs by π).
        """
        jy, jx = self.jacobian_um_per_px[:, 0]
        return float(math.atan2(-jx, jy))

    @property
    def is_axis_aligned(self) -> bool:
        """True if ``J`` is diagonal with positive entries (a :class:`Grid`)."""
        j = self.jacobian_um_per_px
        return bool(
            abs(j[0, 1]) <= _ROTATION_FREE_ATOL * abs(j[1, 1])
            and abs(j[1, 0]) <= _ROTATION_FREE_ATOL * abs(j[0, 0])
            and j[0, 0] > 0
            and j[1, 1] > 0
        )

    def to_sample(
        self, pixels_rc: Any, stage_um: Any, shape: Sequence[int]
    ) -> np.ndarray:
        """Sample position in µm ``(y, x)`` of pixels ``(row, col)``.

        Parameters
        ----------
        pixels_rc : array_like, shape (..., 2)
            Pixel coordinates (sub-pixel allowed).
        stage_um : mapping or pair
            Stage position of the capture.
        shape : sequence of int
            Image shape ``(ny, nx)``.

        Returns
        -------
        numpy.ndarray, shape (..., 2)
        """
        q = np.asarray(pixels_rc, dtype=float)
        rel = q - _centre_px(shape)
        return _stage_yx(stage_um) + self.offset_um + rel @ self.jacobian_um_per_px.T

    def to_pixel(
        self, points_um: Any, stage_um: Any, shape: Sequence[int]
    ) -> np.ndarray:
        """Pixel ``(row, col)`` at which sample points ``(y, x)`` appear.

        Parameters
        ----------
        points_um : array_like, shape (..., 2)
            Sample positions in µm.
        stage_um : mapping or pair
            Stage position of the capture.
        shape : sequence of int
            Image shape ``(ny, nx)``.

        Returns
        -------
        numpy.ndarray, shape (..., 2)
        """
        p = np.asarray(points_um, dtype=float)
        rel = p - _stage_yx(stage_um) - self.offset_um
        inv = np.linalg.inv(self.jacobian_um_per_px)
        return _centre_px(shape) + rel @ inv.T

    def grid(self, stage_um: Any, shape: Sequence[int]) -> Grid:
        """Product grid of a capture taken at ``stage_um``.

        Parameters
        ----------
        stage_um : mapping or pair
            Actual stage position of the capture.
        shape : sequence of int
            ``(ny, nx)``.

        Returns
        -------
        Grid

        Raises
        ------
        ValidationError
            If the model is rotated, sheared or flipped: an axis-aligned
            :class:`Grid` cannot represent it; resample the product first.
        """
        if not self.is_axis_aligned:
            raise ValidationError(
                "The stage-to-image model is not axis-aligned; a Grid cannot represent it."
            )
        centre = _stage_yx(stage_um) + self.offset_um
        return Grid.centred(
            {"y": centre[0], "x": centre[1]},
            (self.jacobian_um_per_px[0, 0], self.jacobian_um_per_px[1, 1]),
            shape,
        )

    # ------------------------------------------------------------ file I/O
    def to_dict(self) -> dict:
        """Return the ``StageImageCalibration`` 1.0 object."""
        if not self.calibrated:
            raise ValidationError("The nominal fallback is not a calibration file.")
        return _to_jsonable(
            {
                "message_type": "StageImageCalibration",
                "schema_version": CALIBRATION_SCHEMA_VERSION,
                "calibration_id": self.calibration_id,
                "created_utc": self.created_utc or _utc_now(),
                "objective": self.objective,
                "device": self.device,
                "jacobian_um_per_px": self.jacobian_um_per_px.tolist(),
                "offset_um": self.offset_um.tolist(),
                "residuals_um": _thaw(self.residuals_um),
                "pose_tolerance_um": self.pose_tolerance_um,
                "source": self.source,
            }
        )

    @classmethod
    def from_dict(cls, data: Mapping) -> StageToImage:
        """Build the model from a ``StageImageCalibration`` object."""
        if data.get("message_type") != "StageImageCalibration":
            raise ValidationError(
                f"Expected message_type 'StageImageCalibration'; got {data.get('message_type')!r}."
            )
        version = str(data.get("schema_version", ""))
        if version.split(".")[0] != CALIBRATION_SCHEMA_VERSION.split(".")[0]:
            raise ValidationError(
                f"Unsupported calibration schema_version {version!r}."
            )
        try:
            return cls(
                jacobian_um_per_px=data["jacobian_um_per_px"],
                offset_um=data["offset_um"],
                objective=data["objective"],
                device=data["device"],
                pose_tolerance_um=data["pose_tolerance_um"],
                calibrated=True,
                calibration_id=data["calibration_id"],
                created_utc=data.get("created_utc"),
                residuals_um=data.get("residuals_um") or {},
                source=data.get("source"),
            )
        except KeyError as exc:
            raise ValidationError(f"Calibration lacks {exc.args[0]!r}.") from exc


def save_calibration(model: StageToImage, path: str | Path) -> Path:
    """Write a calibrated model as JSON.

    Parameters
    ----------
    model : StageToImage
        Calibrated model (``calibrated=True`` with an id).
    path : str or Path
        Target file.

    Returns
    -------
    Path
    """
    if not model.calibration_id:
        raise ValidationError("A calibration file needs a calibration_id.")
    path = Path(path)
    path.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")
    return path


def load_calibration(
    path: str | Path | None,
    *,
    objective: str,
    device: str,
    nominal_pixel_um: Sequence[float],
    fallback_tolerance_um: float,
) -> StageToImage:
    """Load the calibration of one objective and device, or the fallback.

    Parameters
    ----------
    path : str, Path or None
        Calibration file; a missing file or None gives the nominal fallback.
    objective, device : str
        Expected objective and device; a file for another one raises.
    nominal_pixel_um : sequence of float
        Pixel size ``(y, x)`` of the fallback.
    fallback_tolerance_um : float
        Generous pose tolerance of the fallback.

    Returns
    -------
    StageToImage
    """
    if path is None or not Path(path).is_file():
        return StageToImage.nominal(
            nominal_pixel_um,
            objective=objective,
            device=device,
            pose_tolerance_um=fallback_tolerance_um,
        )
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    model = StageToImage.from_dict(data)
    if model.objective != objective or model.device != device:
        raise ValidationError(
            f"Calibration {model.calibration_id!r} is for {model.objective!r} / {model.device!r},"
            f" not {objective!r} / {device!r}."
        )
    return model


@dataclass(frozen=True)
class AffineFit:
    """Result of :func:`fit_stage_to_image`.

    Parameters
    ----------
    jacobian_um_per_px : numpy.ndarray
        Fitted 2 × 2 Jacobian.
    point_minus_offset_um : numpy.ndarray or None
        ``p - d`` for marks at an unknown common sample point (3 × 3
        raster); None if the sample points were known.
    offset_um : numpy.ndarray or None
        Offset ``d`` if the sample points were known.
    residuals_um : numpy.ndarray
        Per-point residual vectors ``(y, x)``.
    rms_um, max_um : float
        RMS and maximum residual length.
    """

    jacobian_um_per_px: np.ndarray
    point_minus_offset_um: np.ndarray | None
    offset_um: np.ndarray | None
    residuals_um: np.ndarray
    rms_um: float
    max_um: float

    def model(
        self, *, objective: str, device: str, calibration_id: str, **kwargs: Any
    ) -> StageToImage:
        """Return the fitted model (needs a known offset)."""
        if self.offset_um is None:
            raise ValidationError(
                "The offset is unknown (marks at an unknown point); pass point_um to the fit."
            )
        return StageToImage(
            jacobian_um_per_px=self.jacobian_um_per_px,
            offset_um=self.offset_um,
            objective=objective,
            device=device,
            calibration_id=calibration_id,
            residuals_um={
                "rms": self.rms_um,
                "max": self.max_um,
                "n_points": int(self.residuals_um.shape[0]),
            },
            **kwargs,
        )


def fit_stage_to_image(
    stage_um: Any,
    pixels_rc: Any,
    shape: Sequence[int],
    *,
    points_um: Any = None,
) -> AffineFit:
    """Least-squares fit of ``J`` (and ``d``) from mark observations.

    Each observation ``i`` is a mark seen at pixel ``q_i`` with the stage at
    ``s_i``: ``p_i = s_i + d + J (q_i - c)``.

    * ``points_um=None``: all observations show the **same** unknown sample
      point (a cross imaged at known stage offsets, e.g. 3 × 3). Unknowns are
      ``J`` and ``p - d``; the fit is independent of the galvo scale. The
      offset stays unknown (``offset_um=None``).
    * ``points_um`` given (one per observation): unknowns are ``J`` and ``d``.

    Parameters
    ----------
    stage_um : array_like, shape (n, 2)
        Stage positions ``(y, x)``.
    pixels_rc : array_like, shape (n, 2)
        Detected mark positions ``(row, col)``.
    shape : sequence of int
        Image shape.
    points_um : array_like, shape (n, 2), optional
        Known sample positions of the marks.

    Returns
    -------
    AffineFit
    """
    s = np.asarray(stage_um, dtype=float).reshape(-1, 2)
    q = np.asarray(pixels_rc, dtype=float).reshape(-1, 2) - _centre_px(shape)
    n = s.shape[0]
    if q.shape[0] != n:
        raise ValidationError("stage_um and pixels_rc need the same number of rows.")
    if n < 3:
        raise ValidationError(f"An affine fit needs at least 3 observations; got {n}.")
    if np.linalg.matrix_rank(np.column_stack([q, np.ones(n)])) < 3:
        raise ValidationError(
            "The observations are collinear; the fit is undetermined."
        )
    design = np.column_stack([q, np.ones(n)])
    if points_um is None:
        # p - d = s_i + J q_i  ->  -s_i = J q_i - (p - d)
        rhs = -s
    else:
        p = np.asarray(points_um, dtype=float).reshape(-1, 2)
        if p.shape[0] != n:
            raise ValidationError("points_um needs one row per observation.")
        # p_i - s_i = J q_i + d
        rhs = p - s
    coef, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    jac = coef[:2].T
    const = coef[2]
    residuals = design @ coef - rhs
    lengths = np.linalg.norm(residuals, axis=1)
    if points_um is None:
        point_minus_offset = -const
        offset = None
    else:
        point_minus_offset = None
        offset = const
    return AffineFit(
        jacobian_um_per_px=jac,
        point_minus_offset_um=point_minus_offset,
        offset_um=offset,
        residuals_um=residuals,
        rms_um=float(np.sqrt(np.mean(lengths**2))),
        max_um=float(lengths.max()),
    )
