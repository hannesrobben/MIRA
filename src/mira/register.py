"""Registration of products by stage position and image (MI-20).

Coarse placement comes from the stage position: every product carries the
grid of its **reported** actual stage position (built with the stage-to-image
model of MI-11). The image registration only refines that prior:

1. the overlap of the two prior grids is cut out of both fields (whole-pixel
   part of the offset); the fractional part ``f`` is the shift the prior
   predicts;
2. phase correlation of the two overlap crops (Hann window, normalised
   cross-power spectrum), with the search **restricted to the pose
   tolerance** around ``f``. This excludes the correlation peak of the
   optics-fixed coherent pattern, which sits at the full tile offset;
3. sub-pixel refinement with a locally upsampled DFT (Guizar-Sicairos et al.,
   Opt. Lett. 33, 156 (2008); same scheme as
   ``skimage.registration.phase_cross_correlation``, which cannot restrict
   its search window, cf. ``processing.registration.PhaseCorrelationRefinement``).

The confidence is ``1 - sidelobe / peak`` of the correlation magnitude, with
the sidelobe searched in the tolerance window enlarged by a margin, outside a
small radius around the peak. Below ``min_confidence`` (no default until M7)
the stage position is kept and the result is flagged.

Works on the complex field by default (piston is irrelevant; tilts between
tiles of one structure are small). ``signal="amplitude"`` or ``"phase"`` are
available for comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .products import Grid
from .utils.exceptions import ValidationError

__all__ = ["TileRegistration", "register_pair", "corrected_grid"]

_SIGNALS = ("field", "amplitude", "phase")


@dataclass(frozen=True)
class TileRegistration:
    """Result of :func:`register_pair`.

    Parameters
    ----------
    shift_um : tuple of float
        Correction ``(y, x)`` to add to the moving product's grid origin
        (true minus reported position). ``(0, 0)`` in a fallback.
    confidence : float
        ``1 - sidelobe / peak`` in [0, 1]; 0 without a correlation.
    method : str
        ``"phase_correlation"`` or ``"stage_prior"`` (fallback).
    flagged : bool
        True if the stage prior was kept because of low confidence or too
        little overlap.
    reason : str or None
        Why the result is flagged.
    overlap_px : tuple of int
        Size of the overlap used.
    measured_shift_um : tuple of float or None
        Shift found by the correlation even when it was rejected.
    isotropy : float or None
        Smallest over largest eigenvalue of the structure tensor of the
        reference overlap (0 = a straight edge, 1 = isotropic).
    undetermined_directions : tuple
        Unit vectors ``(y, x)`` along which the shift was dropped.
    """

    shift_um: tuple[float, float]
    confidence: float
    method: str
    flagged: bool
    reason: str | None
    overlap_px: tuple[int, int]
    measured_shift_um: tuple[float, float] | None = None
    isotropy: float | None = None
    undetermined_directions: tuple = ()

    def to_record(self) -> dict:
        """Return the ``registration`` mapping of a ``TileRecord``."""
        return {
            "shift_um": list(self.shift_um),
            "method": self.method,
            "confidence": self.confidence,
            "flagged": self.flagged,
            "reason": self.reason,
            "overlap_px": list(self.overlap_px),
            "isotropy": self.isotropy,
            "undetermined_directions": [list(d) for d in self.undetermined_directions],
            "measured_shift_um": (
                list(self.measured_shift_um)
                if self.measured_shift_um is not None
                else None
            ),
        }


def corrected_grid(grid: Grid, registration: TileRegistration) -> Grid:
    """Grid of the moving product after applying the registration shift."""
    return Grid(
        (
            grid.origin_um[0] + registration.shift_um[0],
            grid.origin_um[1] + registration.shift_um[1],
        ),
        grid.pixel_um,
        grid.shape,
    )


def _crops(grid_ref: Grid, grid_mov: Grid):
    """Integer-pixel overlap of two grids plus the fractional prior shift."""
    if not np.allclose(grid_ref.pixel_um, grid_mov.pixel_um, rtol=1e-6, atol=0):
        raise ValidationError("Products have different pixel sizes; resample first.")
    sl_ref, sl_mov, frac = [], [], []
    for axis in (0, 1):
        delta = (
            grid_mov.origin_um[axis] - grid_ref.origin_um[axis]
        ) / grid_ref.pixel_um[axis]
        k = math.floor(delta + 0.5)
        frac.append(delta - k)
        start = max(0, k)
        stop = min(grid_ref.shape[axis], k + grid_mov.shape[axis])
        if stop <= start:
            return None
        sl_ref.append(slice(start, stop))
        sl_mov.append(slice(start - k, stop - k))
    return tuple(sl_ref), tuple(sl_mov), np.array(frac)


def _signal(field: np.ndarray, kind: str) -> np.ndarray:
    if kind == "field":
        u = field.astype(np.complex128)
        return u - u.mean()
    if kind == "amplitude":
        a = np.abs(field).astype(np.float64)
        return a - a.mean()
    p = np.angle(field).astype(np.float64)
    return p - p.mean()


def _structure_tensor(u: np.ndarray, sigma_px: float, window: np.ndarray):
    """Eigenvalues (ascending) and eigenvectors of the windowed gradient tensor."""
    from scipy import ndimage

    tensor = np.zeros((2, 2))
    for part in (u.real, u.imag):
        smooth = (
            ndimage.gaussian_filter(part, max(sigma_px, 0.5)) if sigma_px > 0 else part
        )
        gy, gx = np.gradient(smooth)
        tensor += np.array(
            [
                [np.sum(window * gy * gy), np.sum(window * gy * gx)],
                [np.sum(window * gx * gy), np.sum(window * gx * gx)],
            ]
        )
    return np.linalg.eigh(tensor)


def _upsampled_dft(
    data: np.ndarray, size: int, factor: int, offsets: np.ndarray
) -> np.ndarray:
    """Inverse DFT of ``data`` on a ``size`` × ``size`` patch upsampled by ``factor``."""
    out = data
    for n_items, offset in zip(data.shape[::-1], offsets[::-1]):
        kernel = (np.arange(size) - offset)[:, None] * np.fft.fftfreq(n_items, factor)
        out = np.tensordot(np.exp(-2j * np.pi * kernel), out, axes=(1, -1))
    return out


def register_pair(
    field_mov: np.ndarray,
    grid_mov: Grid,
    field_ref: np.ndarray,
    grid_ref: Grid,
    *,
    pose_tolerance_um: float,
    min_confidence: float,
    min_isotropy: float,
    min_overlap_px: int = 16,
    upsample_factor: int = 20,
    signal: str = "field",
    sidelobe_margin_px: int = 4,
    peak_exclusion_px: float = 2.0,
    whitening: float = 0.5,
    lowpass_sigma_px: float = 1.0,
    confidence_lowpass_px: float = 1.5,
    iterations: int = 2,
) -> TileRegistration:
    """Register a moving product onto a reference product.

    Parameters
    ----------
    field_mov, field_ref : numpy.ndarray (complex)
        Fields of the two products.
    grid_mov, grid_ref : Grid
        Prior grids from the reported stage positions (same pixel size).
    pose_tolerance_um : float
        Search radius around the prior (``pose_tolerance_um`` of the capture
        context or of the stage-to-image model).
    min_confidence : float
        Results below it keep the stage position and are flagged. No default:
        acceptance tolerances wait for M7.
    min_isotropy : float
        Directions whose gradient information (eigenvalue of the structure
        tensor of the reference overlap) is below ``min_isotropy`` times the
        largest one are undetermined, e.g. along a straight edge; the shift
        along them is dropped (stage prior kept) and recorded. No default
        (M7).
    min_overlap_px : int, optional
        Minimum overlap per axis.
    upsample_factor : int, optional
        Sub-pixel resolution is ``1 / upsample_factor`` px.
    signal : {"field", "amplitude", "phase"}, optional
        What is correlated.
    sidelobe_margin_px : int, optional
        How far beyond the tolerance window the sidelobe is searched.
    peak_exclusion_px : float, optional
        Radius around the peak excluded from the sidelobe search.
    whitening : float, optional
        Exponent ``a`` of the spectrum normalisation ``P / |P|**a`` used for
        the estimate: 1 is classic phase correlation (it amplifies the frame
        noise and sharpens the ridge of a straight edge into a false peak),
        0 plain cross-correlation. 0.5 with ``lowpass_sigma_px = 1`` was the
        most accurate on MI-22 fields (2-D features and edge-only overlaps).
    lowpass_sigma_px : float, optional
        Gaussian low-pass of the cross-power spectrum for the estimate,
        equivalent to blurring both images with ``sigma / sqrt(2)`` px.
    confidence_lowpass_px : float, optional
        Low-pass of the classic phase correlation that yields the confidence
        (its peak / sidelobe ratio separates featureless overlaps best).
    iterations : int, optional
        Number of passes; every further pass re-crops the overlap with the
        corrected prior and searches ±1 px (removes the window bias).

    Returns
    -------
    TileRegistration
    """
    if signal not in _SIGNALS:
        raise ValidationError(f"signal must be one of {_SIGNALS}; got {signal!r}.")
    if not (pose_tolerance_um >= 0 and math.isfinite(pose_tolerance_um)):
        raise ValidationError("pose_tolerance_um must be finite and >= 0.")
    field_mov = np.asarray(field_mov)
    field_ref = np.asarray(field_ref)
    if field_mov.shape != grid_mov.shape or field_ref.shape != grid_ref.shape:
        raise ValidationError("Field shapes do not match their grids.")

    def fallback(reason: str, overlap=(0, 0), confidence=0.0, measured=None):
        return TileRegistration(
            shift_um=(0.0, 0.0),
            confidence=float(confidence),
            method="stage_prior",
            flagged=True,
            reason=reason,
            overlap_px=tuple(int(v) for v in overlap),
            measured_shift_um=measured,
        )

    pixel = np.array(grid_ref.pixel_um)
    tol_px = pose_tolerance_um / pixel
    factor = max(int(upsample_factor), 1)

    def correlate(grid_m: Grid, tol: np.ndarray):
        """One pass: (shift relative to the prior in px, confidence, overlap) or a reason."""
        cut = _crops(grid_ref, grid_m)
        if cut is None:
            return "no overlap", (0, 0)
        sl_ref, sl_mov, frac = cut
        ref = field_ref[sl_ref]
        mov = field_mov[sl_mov]
        overlap = ref.shape
        if min(overlap) < min_overlap_px:
            return f"overlap {overlap} below {min_overlap_px} px", overlap
        window = np.outer(np.hanning(overlap[0]), np.hanning(overlap[1]))
        eig = _structure_tensor(
            _signal(ref, signal) * (1 + 0j), lowpass_sigma_px, window
        )
        raw = np.fft.fft2(_signal(ref, signal) * window) * np.conj(
            np.fft.fft2(_signal(mov, signal) * window)
        )
        fy = np.fft.fftfreq(overlap[0])[:, None]
        fx = np.fft.fftfreq(overlap[1])[None, :]
        floor = 1e-12 * np.abs(raw).max()

        def surface(power: float, sigma: float):
            prod = raw / np.maximum(np.abs(raw), floor) ** power if power > 0 else raw
            if sigma > 0:
                prod = prod * np.exp(-2.0 * (np.pi * sigma) ** 2 * (fy**2 + fx**2))
            prod = prod / max(np.abs(prod).sum() / prod.size, floor)
            return prod, np.abs(np.fft.ifft2(prod))

        product, corr = surface(whitening, lowpass_sigma_px)
        sy = np.fft.fftfreq(overlap[0], 1.0 / overlap[0])[:, None]  # signed shifts
        sx = np.fft.fftfreq(overlap[1], 1.0 / overlap[1])[None, :]
        ry, rx = max(tol[0], 0.5), max(tol[1], 0.5)
        inside = ((sy - frac[0]) / ry) ** 2 + ((sx - frac[1]) / rx) ** 2 <= 1.0 + 1e-9
        if not inside.any():
            return "empty search window", overlap
        iy, ix = np.unravel_index(
            np.argmax(np.where(inside, corr, -np.inf)), corr.shape
        )
        shift = np.array([sy[iy, 0], sx[0, ix]], dtype=float)

        # Confidence from classic phase correlation (sharpest peak): peak in
        # the window against the sidelobe in the window enlarged by a margin.
        _, corr_c = surface(1.0, confidence_lowpass_px)
        cy, cx = np.unravel_index(
            np.argmax(np.where(inside, corr_c, -np.inf)), corr.shape
        )
        peak = corr_c[cy, cx]
        big = ((sy - frac[0]) / (ry + sidelobe_margin_px)) ** 2 + (
            (sx - frac[1]) / (rx + sidelobe_margin_px)
        ) ** 2 <= 1.0
        dy_, dx_ = sy - sy[cy, 0], sx - sx[0, cx]
        near = dy_**2 + dx_**2 <= peak_exclusion_px**2
        vals, vecs = eig
        for k in range(2):  # undetermined direction: ignore the ridge along it
            if vals[k] / max(float(vals[-1]), 1e-300) < min_isotropy:
                normal = vecs[:, 1 - k]
                near |= np.abs(dy_ * normal[0] + dx_ * normal[1]) <= peak_exclusion_px
        lobe = big & ~near
        sidelobe = corr_c[lobe].max() if lobe.any() else corr_c.mean()
        if factor > 1:  # sub-pixel refinement around the integer peak
            size = int(math.ceil(factor * 1.5))
            centre = np.trunc(size / 2.0)
            shift = np.round(shift * factor) / factor
            local = np.abs(
                _upsampled_dft(
                    product.conj(), size, factor, centre - shift * factor
                ).conj()
            )
            my, mx = np.unravel_index(np.argmax(local), local.shape)
            shift = shift + (np.array([my, mx]) - centre) / factor
        if peak <= 0:
            return shift - frac, 0.0, overlap, eig
        confidence = float(np.clip(1.0 - sidelobe / peak, 0.0, 1.0))
        return shift - frac, confidence, overlap, eig

    first = correlate(grid_mov, tol_px)
    if isinstance(first[0], str):
        return fallback(*first)
    total_px, confidence, overlap, (eigvals, eigvecs) = first
    lam_max = max(float(eigvals[-1]), 1e-300)
    weak = [eigvecs[:, k] for k in range(2) if eigvals[k] / lam_max < min_isotropy]
    undetermined = [[float(d[0]), float(d[1])] for d in weak]

    def project(v: np.ndarray) -> np.ndarray:
        for d in weak:
            v = v - np.dot(v, d) * d
        return v

    total_px = project(total_px)
    # Further passes re-crop with the corrected prior: the residual shift is
    # then small, which removes the bias of the window towards zero shift.
    for _ in range(max(int(iterations), 1) - 1):
        if len(weak) == 2:
            break
        moved = Grid(
            tuple(np.array(grid_mov.origin_um) + total_px * pixel),
            grid_mov.pixel_um,
            grid_mov.shape,
        )
        step = correlate(moved, np.full(2, 1.0))
        if isinstance(step[0], str):
            break
        total_px = total_px + project(step[0])
    isotropy = float(max(eigvals[0], 0.0) / lam_max)
    measured = tuple(float(v) for v in total_px * pixel)
    if len(undetermined) == 2:
        return fallback(
            "no gradient information in the overlap", overlap, confidence, measured
        )
    if np.any(np.abs(total_px) > tol_px + 1.0 / factor):
        return fallback(
            "peak outside the pose tolerance", overlap, confidence, measured
        )
    if confidence < min_confidence:
        return fallback(
            f"confidence {confidence:.3f} below {min_confidence}",
            overlap,
            confidence,
            measured,
        )
    return TileRegistration(
        shift_um=measured,
        confidence=confidence,
        method="phase_correlation_partial" if undetermined else "phase_correlation",
        flagged=False,
        reason="undetermined direction (stage prior kept)" if undetermined else None,
        isotropy=isotropy,
        undetermined_directions=tuple(tuple(d) for d in undetermined),
        overlap_px=(int(overlap[0]), int(overlap[1])),
        measured_shift_um=measured,
    )
