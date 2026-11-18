"""Piston and tilt between two products from their complex overlap (MI-21).

This is the **one** implementation used by fusion (MI-30) and stitching
(MI-40) (decision M3). With ``P = U_new · conj(U_ref)`` over the overlap:

* tilt: the mean phase increment between neighbouring pixels,
  ``angle(Σ P[i+1] · conj(P[i]))`` per axis, divided by the pixel size. It
  needs no unwrapping and is robust as long as the tilt is below π per pixel;
* refinement: one weighted least-squares pass on the remaining angles;
* piston: ``angle(Σ P · exp(-i · tilt · r))`` with ``r`` relative to a stated
  centre (decision G10 SA6: angle of Σ U_new · conj(U_ref)).

The products ``U_new · conj(U_ref)`` weight every pixel by
``|U_new| · |U_ref|`` (implicit amplitude weighting, M3); an optional extra
weight per pixel multiplies it. Works on NumPy and CuPy arrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .products import Grid, _xp
from .utils.exceptions import ValidationError

__all__ = ["OverlapFit", "fit_piston_tilt", "remove_piston_tilt", "overlap_slices"]


@dataclass(frozen=True)
class OverlapFit:
    """Result of :func:`fit_piston_tilt`.

    Parameters
    ----------
    piston_rad : float
        Phase of ``U_new`` minus ``U_ref`` at ``centre_um`` (in (-π, π]).
    tilt_rad_per_um : tuple of float
        Linear phase difference ``(y, x)`` in rad / µm.
    centre_um : tuple of float
        Position ``(y, x)`` in stage µm the piston refers to.
    residual_rad : float
        Weighted RMS of the remaining phase difference after the fit.
    coherence : float
        ``|Σ w P'| / Σ w |P'|`` after removing the fit (1 = perfect match,
        about 0 = unrelated fields).
    n_pixels : int
        Number of pixels used.
    """

    piston_rad: float
    tilt_rad_per_um: tuple[float, float]
    centre_um: tuple[float, float]
    residual_rad: float
    coherence: float
    n_pixels: int

    def phase_at(self, grid: Grid):
        """Return the fitted phase difference on ``grid`` (NumPy, float64)."""
        y, x = _coords(grid, np)
        return (
            self.piston_rad
            + self.tilt_rad_per_um[0] * (y - self.centre_um[0])
            + self.tilt_rad_per_um[1] * (x - self.centre_um[1])
        )


def _coords(grid: Grid, xp):
    y = grid.origin_um[0] + grid.pixel_um[0] * xp.arange(grid.shape[0])
    x = grid.origin_um[1] + grid.pixel_um[1] * xp.arange(grid.shape[1])
    return y[:, None], x[None, :]


def fit_piston_tilt(
    u_new,
    u_ref,
    grid: Grid,
    *,
    mask=None,
    weight=None,
    fit_tilt: bool = True,
    centre_um: tuple[float, float] | None = None,
    min_pixels: int = 16,
    refine: bool = True,
) -> OverlapFit:
    """Estimate piston and tilt of ``u_new`` relative to ``u_ref``.

    Parameters
    ----------
    u_new, u_ref : array (complex)
        The two fields on the same ``grid`` (the overlap region).
    grid : Grid
        Grid of both arrays in stage coordinates.
    mask : array (bool), optional
        Pixels to use (e.g. substrate or valid pixels); default all.
    weight : array (float), optional
        Extra weight per pixel (e.g. a quality weight), multiplied with the
        implicit amplitude weight.
    fit_tilt : bool, optional
        Estimate the tilt; if False only the piston is fitted.
    centre_um : tuple of float, optional
        Position the piston refers to; default the grid centre.
    min_pixels : int, optional
        Minimum number of usable pixels.
    refine : bool, optional
        Refine the tilt by one weighted least-squares pass on the remaining
        angles (assumes they stay well below π, i.e. matching fields).

    Returns
    -------
    OverlapFit

    Raises
    ------
    ValidationError
        Shapes differ from the grid or too few usable pixels.
    """
    xp = _xp(u_new)
    u_new = xp.asarray(u_new)
    u_ref = xp.asarray(u_ref)
    if u_new.shape != grid.shape or u_ref.shape != grid.shape:
        raise ValidationError(
            f"Fields {u_new.shape} / {u_ref.shape} do not match grid {grid.shape}."
        )
    w = xp.ones(grid.shape, dtype=xp.float64)
    if weight is not None:
        w = w * xp.asarray(weight, dtype=xp.float64)
    if mask is not None:
        w = w * xp.asarray(mask, dtype=bool)
    product = u_new.astype(xp.complex128) * xp.conj(u_ref.astype(xp.complex128))
    good = xp.isfinite(product) & (w > 0)
    product = xp.where(good, product, 0)
    w = xp.where(good, w, 0)
    n = int(good.sum())
    if n < min_pixels:
        raise ValidationError(f"Only {n} usable overlap pixels (< {min_pixels}).")
    centre = grid.centre_um if centre_um is None else (float(centre_um[0]), float(centre_um[1]))
    wp = w * product
    tilt = (0.0, 0.0)
    if fit_tilt:
        slopes = []
        for axis, step in ((0, grid.pixel_um[0]), (1, grid.pixel_um[1])):
            if grid.shape[axis] < 2:
                slopes.append(0.0)
                continue
            a = [slice(None)] * 2
            b = [slice(None)] * 2
            a[axis] = slice(1, None)
            b[axis] = slice(None, -1)
            pair = wp[tuple(a)] * xp.conj(wp[tuple(b)])
            total = complex(pair.sum())
            slopes.append(math.atan2(total.imag, total.real) / step if total != 0 else 0.0)
        tilt = (slopes[0], slopes[1])
    y, x = _coords(grid, xp)
    ramp = tilt[0] * (y - centre[0]) + tilt[1] * (x - centre[1])
    detilted = wp * xp.exp(-1j * ramp)
    total = complex(detilted.sum())
    piston = math.atan2(total.imag, total.real)
    if fit_tilt and refine:
        # One weighted least-squares pass on the (small) remaining angles
        # improves the efficiency of the increment estimator.
        angles = xp.angle(detilted * np.exp(-1j * piston))
        weights = w * xp.abs(product)
        dy = xp.broadcast_to(y - centre[0], grid.shape)
        dx = xp.broadcast_to(x - centre[1], grid.shape)
        sw = xp.sqrt(weights).ravel()
        design = xp.stack([sw, sw * dy.ravel(), sw * dx.ravel()], axis=1)
        coef = xp.linalg.lstsq(design, sw * angles.ravel(), rcond=None)[0]
        tilt = (tilt[0] + float(coef[1]), tilt[1] + float(coef[2]))
        ramp = tilt[0] * (y - centre[0]) + tilt[1] * (x - centre[1])
        detilted = wp * xp.exp(-1j * ramp)
        total = complex(detilted.sum())
        piston = math.atan2(total.imag, total.real)
    remaining = detilted * np.exp(-1j * piston)
    magnitude = float((w * xp.abs(product)).sum())
    coherence = abs(total) / magnitude if magnitude > 0 else 0.0
    angles = xp.angle(remaining)
    weights = w * xp.abs(product)
    residual = math.sqrt(float((weights * angles**2).sum()) / float(weights.sum()))
    return OverlapFit(
        piston_rad=piston,
        tilt_rad_per_um=(float(tilt[0]), float(tilt[1])),
        centre_um=centre,
        residual_rad=residual,
        coherence=float(coherence),
        n_pixels=n,
    )


def remove_piston_tilt(field, grid: Grid, fit: OverlapFit):
    """Return ``field`` with the fitted piston and tilt removed.

    Parameters
    ----------
    field : array (complex)
        Field on ``grid`` (the whole tile, not only the overlap).
    grid : Grid
        Grid of ``field``.
    fit : OverlapFit
        Fit of this field against the reference.

    Returns
    -------
    array (complex64)
        ``field · exp(-i · (piston + tilt · (r - centre)))``, same backend.
    """
    xp = _xp(field)
    y, x = _coords(grid, xp)
    ramp = (
        fit.piston_rad
        + fit.tilt_rad_per_um[0] * (y - fit.centre_um[0])
        + fit.tilt_rad_per_um[1] * (x - fit.centre_um[1])
    )
    return (xp.asarray(field) * xp.exp(-1j * ramp)).astype(xp.complex64)


def overlap_slices(
    grid_a: Grid, grid_b: Grid, *, tol_px: float = 1e-3
) -> tuple[tuple[slice, slice], tuple[slice, slice], Grid] | None:
    """Pixel slices of the common area of two grids on the same lattice.

    Parameters
    ----------
    grid_a, grid_b : Grid
        Grids with the same pixel size whose origins differ by a whole
        number of pixels (within ``tol_px``), e.g. tiles resampled onto a
        mosaic lattice.
    tol_px : float, optional
        Allowed deviation from a whole-pixel offset.

    Returns
    -------
    (slices_a, slices_b, grid) or None
        Slices into both arrays and the grid of the common area; None if the
        grids do not overlap.

    Raises
    ------
    ValidationError
        Different pixel sizes or a sub-pixel offset.
    """
    if not np.allclose(grid_a.pixel_um, grid_b.pixel_um, rtol=1e-9, atol=0):
        raise ValidationError("Grids have different pixel sizes.")
    offsets = []
    for axis in (0, 1):
        delta = (grid_b.origin_um[axis] - grid_a.origin_um[axis]) / grid_a.pixel_um[axis]
        k = round(delta)
        if abs(delta - k) > tol_px:
            raise ValidationError(
                f"Grids are offset by {delta:.4f} px on axis {axis}; resample first."
            )
        offsets.append(k)
    sa, sb, shape = [], [], []
    for axis in (0, 1):
        k = offsets[axis]
        start_a = max(0, k)
        stop_a = min(grid_a.shape[axis], k + grid_b.shape[axis])
        if stop_a <= start_a:
            return None
        sa.append(slice(start_a, stop_a))
        sb.append(slice(start_a - k, stop_a - k))
        shape.append(stop_a - start_a)
    origin = (
        grid_a.origin_um[0] + sa[0].start * grid_a.pixel_um[0],
        grid_a.origin_um[1] + sa[1].start * grid_a.pixel_um[1],
    )
    return (sa[0], sa[1]), (sb[0], sb[1]), Grid(origin, grid_a.pixel_um, tuple(shape))
