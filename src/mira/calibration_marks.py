"""Detection and fit of the printed calibration marks (MI-41, G10 SA2).

nanofactorysystem T104 prints a **cross** and a **square** with the writing
field (galvo / stage) at known sample positions. Imaged by the camera or the
DHM they give the stage-to-image model of MI-11:

* the **cross** centre (:func:`detect_cross`) gives the offset between the
  writing field and the image centre; imaged at known stage offsets (e.g.
  3 × 3) it gives an affine least-squares fit of the Jacobian that is
  independent of the galvo scale (:func:`fit_cross_raster`);
* the **square** (:func:`detect_square`) gives rotation, scale and shear from
  its four sub-pixel corners (edge lines fitted by total least squares and
  intersected). A square is symmetric under 90° turns and flips; the
  ambiguity is resolved with the nominal model (the corner assignment whose
  Jacobian is closest to the prior).

Inputs are 2-D images (phase, height or intensity); the mark may be brighter
or darker than the background. Results feed the calibration file
(:func:`mira.geometry.save_calibration`); residual shifts of the stitching
(MI-40) can refine it later.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from .geometry import AffineFit, StageToImage, fit_stage_to_image
from .utils.exceptions import ValidationError

__all__ = [
    "MarkDetection",
    "SquareFit",
    "segment_mark",
    "detect_cross",
    "detect_square",
    "fit_cross_raster",
]


@dataclass(frozen=True)
class MarkDetection:
    """Detected mark centre.

    Parameters
    ----------
    centre_rc : tuple of float
        Sub-pixel centre ``(row, col)``.
    n_pixels : int
        Size of the mark mask.
    polarity : int
        +1 if the mark is above the background, -1 if below.
    """

    centre_rc: tuple[float, float]
    n_pixels: int
    polarity: int


@dataclass(frozen=True)
class SquareFit:
    """Square fitted to its four corners.

    Parameters
    ----------
    corners_rc : numpy.ndarray, shape (4, 2)
        Sub-pixel corners in the order of the design corners
        ``(-, -), (-, +), (+, +), (+, -)`` (``(y, x)`` signs).
    jacobian_um_per_px : numpy.ndarray
        2 × 2 Jacobian (µm per pixel) from the design square to the image.
    centre_rc : tuple of float
        Image position of the square centre.
    residual_px : float
        RMS corner residual of the affine fit in pixels.
    """

    corners_rc: np.ndarray
    jacobian_um_per_px: np.ndarray
    centre_rc: tuple[float, float]
    residual_px: float

    def model(self, prior: StageToImage, **kwargs: Any) -> StageToImage:
        """Stage-to-image model with this Jacobian and the prior's offset."""
        params = {
            "objective": prior.objective,
            "device": prior.device,
            "pose_tolerance_um": prior.pose_tolerance_um,
            "offset_um": prior.offset_um,
        }
        params.update(kwargs)
        return StageToImage(jacobian_um_per_px=self.jacobian_um_per_px, **params)


def segment_mark(image: Any, *, smooth_px: float = 1.0) -> tuple[np.ndarray, int]:
    """Binary mask of the largest object that differs from the background.

    The background level is the image median; the threshold is Otsu's on
    the absolute deviation, and the largest connected component is kept.

    Parameters
    ----------
    image : array_like
        2-D image.
    smooth_px : float, optional
        Gaussian smoothing before thresholding.

    Returns
    -------
    mask : numpy.ndarray of bool
    polarity : int
        +1 if the mark is brighter than the background, else -1.
    """
    from skimage.filters import threshold_otsu

    img = np.asarray(image, float)
    if img.ndim != 2:
        raise ValidationError("image must be 2-D.")
    if smooth_px > 0:
        img = ndimage.gaussian_filter(img, smooth_px)
    dev = img - np.median(img)
    level = threshold_otsu(np.abs(dev))
    mask = np.abs(dev) > level
    labels, n = ndimage.label(mask)
    if n == 0:
        raise ValidationError("No mark found.")
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    keep = labels == (int(np.argmax(sizes)) + 1)
    keep = ndimage.binary_fill_holes(keep)
    polarity = 1 if np.mean(dev[keep]) > 0 else -1
    return keep, polarity


def detect_cross(image: Any, *, smooth_px: float = 1.0) -> MarkDetection:
    """Centre of a cross: intensity-weighted centroid of the mark mask.

    A cross is point-symmetric, so its centroid is its centre; weighting
    with the deviation from the background gives sub-pixel accuracy.

    Parameters
    ----------
    image : array_like
        2-D image with one cross.
    smooth_px : float, optional

    Returns
    -------
    MarkDetection
    """
    img = np.asarray(image, float)
    mask, polarity = segment_mark(img, smooth_px=smooth_px)
    mask = ndimage.binary_dilation(mask, iterations=2)
    weight = np.clip(polarity * (img - np.median(img)), 0, None) * mask
    total = weight.sum()
    if total <= 0:
        raise ValidationError("Cross has no contrast.")
    rows, cols = np.indices(img.shape)
    centre = (
        float((rows * weight).sum() / total),
        float((cols * weight).sum() / total),
    )
    return MarkDetection(centre_rc=centre, n_pixels=int(mask.sum()), polarity=polarity)


def _fit_line(points: np.ndarray):
    """Total-least-squares line: (point, unit direction)."""
    centre = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centre)
    return centre, vt[0]


def _intersect(l1, l2) -> np.ndarray:
    p1, d1 = l1
    p2, d2 = l2
    a = np.column_stack([d1, -d2])
    t = np.linalg.solve(a, p2 - p1)
    return p1 + t[0] * d1


def detect_square(
    image: Any,
    side_um: float,
    prior: StageToImage,
    *,
    smooth_px: float = 1.0,
    edge_trim: float = 0.2,
) -> SquareFit:
    """Corners and affine Jacobian of a printed square.

    Parameters
    ----------
    image : array_like
        2-D image with one filled square.
    side_um : float
        Design side length.
    prior : StageToImage
        Nominal or previous model; resolves the symmetry of the square.
    smooth_px : float, optional
    edge_trim : float, optional
        Share of every side next to the corners left out of the line fit.

    Returns
    -------
    SquareFit
    """
    import cv2

    from skimage.measure import find_contours

    img = np.asarray(image, float)
    mask, polarity = segment_mark(img, smooth_px=smooth_px)
    smooth = ndimage.gaussian_filter(img, smooth_px) if smooth_px > 0 else img
    dev = polarity * (smooth - np.median(smooth))
    level = 0.5 * float(np.median(dev[ndimage.binary_erosion(mask, iterations=3)]))
    found = find_contours(
        np.where(ndimage.binary_dilation(mask, iterations=3), dev, 0.0), level
    )
    if not found:
        raise ValidationError("Square contour not found.")
    contour = max(found, key=len)[:, ::-1]  # sub-pixel (x, y)
    if np.allclose(contour[0], contour[-1]):
        contour = contour[:-1]
    c32 = contour.astype(np.float32)
    approx = cv2.approxPolyDP(c32, 0.05 * cv2.arcLength(c32, True), True)
    if len(approx) != 4:
        raise ValidationError(f"Square not found ({len(approx)} corners).")
    seeds = approx[:, 0, :].astype(float)
    # Assign contour points to sides between consecutive seed corners.
    idx = [int(np.argmin(np.hypot(*(contour - s).T))) for s in seeds]
    order = np.argsort(idx)
    idx = [idx[i] for i in order]
    seeds = seeds[order]
    lines = []
    for k in range(4):
        start, stop = idx[k], idx[(k + 1) % 4]
        seg = (
            contour[start:stop]
            if stop > start
            else np.vstack([contour[start:], contour[:stop]])
        )
        trim = int(edge_trim * len(seg))
        seg = seg[trim : len(seg) - trim] if len(seg) > 2 * trim + 2 else seg
        if len(seg) < 3:
            raise ValidationError("Square side too short for a line fit.")
        lines.append(_fit_line(seg))
    corners_xy = np.array([_intersect(lines[k - 1], lines[k]) for k in range(4)])
    corners_rc = corners_xy[:, ::-1]
    design = (
        0.5 * side_um * np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]], float)
    )  # (y, x)
    best = None
    for perm in itertools.permutations(range(4)):
        pts = corners_rc[list(perm)]
        # px = c + A design  ->  LS for A (2 x 2) and c
        design_h = np.column_stack([design, np.ones(4)])
        coef, *_ = np.linalg.lstsq(design_h, pts, rcond=None)
        a_px_per_um = coef[:2].T
        if abs(np.linalg.det(a_px_per_um)) < 1e-12:
            continue
        jac = np.linalg.inv(a_px_per_um)
        resid = design_h @ coef - pts
        score = np.linalg.norm(jac - prior.jacobian_um_per_px) + 1e-3 * np.sqrt(
            np.mean(resid**2)
        )
        if best is None or score < best[0]:
            best = (
                score,
                pts,
                jac,
                coef[2],
                float(np.sqrt(np.mean(np.sum(resid**2, axis=1)))),
            )
    _, pts, jac, centre, resid = best
    return SquareFit(
        corners_rc=pts,
        jacobian_um_per_px=jac,
        centre_rc=(float(centre[0]), float(centre[1])),
        residual_px=resid,
    )


def fit_cross_raster(
    images: Sequence[Any],
    stage_um: Sequence[Sequence[float]],
    *,
    cross_um: Sequence[float] | None = None,
    smooth_px: float = 1.0,
) -> tuple[AffineFit, list[MarkDetection]]:
    """Affine fit from one cross imaged at several stage positions.

    Parameters
    ----------
    images : sequence of 2-D arrays
        One image per stage position (e.g. a 3 × 3 raster).
    stage_um : sequence of (y, x)
        Actual stage positions of the captures.
    cross_um : sequence of float, optional
        Sample position ``(y, x)`` of the cross as printed by the writing
        field; with it the offset is fitted too.
    smooth_px : float, optional

    Returns
    -------
    fit : AffineFit
        See :func:`mira.geometry.fit_stage_to_image`.
    detections : list of MarkDetection
    """
    if len(images) != len(stage_um):
        raise ValidationError("Need one stage position per image.")
    detections = [detect_cross(img, smooth_px=smooth_px) for img in images]
    shape = np.asarray(images[0]).shape
    points = (
        None if cross_um is None else np.repeat([list(cross_um)], len(images), axis=0)
    )
    fit = fit_stage_to_image(
        stage_um, [d.centre_rc for d in detections], shape, points_um=points
    )
    return fit, detections
