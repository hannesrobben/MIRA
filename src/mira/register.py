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
    """

    shift_um: tuple[float, float]
    confidence: float
    method: str
    flagged: bool
    reason: str | None
    overlap_px: tuple[int, int]
    measured_shift_um: tuple[float, float] | None = None

    def to_record(self) -> dict:
        """Return the ``registration`` mapping of a ``TileRecord``."""
        return {
            "shift_um": list(self.shift_um),
            "method": self.method,
            "confidence": self.confidence,
            "flagged": self.flagged,
            "reason": self.reason,
            "overlap_px": list(self.overlap_px),
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
    min_overlap_px: int = 16,
    upsample_factor: int = 20,
    signal: str = "field",
    sidelobe_margin_px: int = 4,
    peak_exclusion_px: float = 2.0,
    whitening: float = 1.0,
    lowpass_sigma_px: float = 1.5,
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
        Exponent ``a`` of the spectrum normalisation ``P / |P|**a``: 1 is
        classic phase correlation (sharpest peak, but it amplifies the
        frame noise at high frequencies), 0 plain cross-correlation.
    lowpass_sigma_px : float, optional
        Gaussian low-pass of the cross-power spectrum, equivalent to blurring
        both images with ``sigma / sqrt(2)`` px; suppresses the frame noise
        that whitening amplifies. 0 disables it.
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
        product = np.fft.fft2(_signal(ref, signal) * window) * np.conj(
            np.fft.fft2(_signal(mov, signal) * window)
        )
        magnitude = np.abs(product)
        floor = 1e-12 * magnitude.max()
        if whitening > 0:
            product = product / np.maximum(magnitude, floor) ** whitening
        if lowpass_sigma_px > 0:
            fy = np.fft.fftfreq(overlap[0])[:, None]
            fx = np.fft.fftfreq(overlap[1])[None, :]
            product = product * np.exp(
                -2.0 * (np.pi * lowpass_sigma_px) ** 2 * (fy**2 + fx**2)
            )
        product = product / max(np.abs(product).sum() / product.size, floor)
        corr = np.abs(np.fft.ifft2(product))

        sy = np.fft.fftfreq(overlap[0], 1.0 / overlap[0])[
            :, None
        ]  # signed integer shifts
        sx = np.fft.fftfreq(overlap[1], 1.0 / overlap[1])[None, :]
        ry, rx = max(tol[0], 0.5), max(tol[1], 0.5)
        inside = ((sy - frac[0]) / ry) ** 2 + ((sx - frac[1]) / rx) ** 2 <= 1.0 + 1e-9
        if not inside.any():
            return "empty search window", overlap
        iy, ix = np.unravel_index(
            np.argmax(np.where(inside, corr, -np.inf)), corr.shape
        )
        shift = np.array([sy[iy, 0], sx[0, ix]], dtype=float)
        peak = corr[iy, ix]
        # Sidelobe: window enlarged by a margin, outside a radius around the peak.
        big = ((sy - frac[0]) / (ry + sidelobe_margin_px)) ** 2 + (
            (sx - frac[1]) / (rx + sidelobe_margin_px)
        ) ** 2 <= 1.0
        near = (sy - shift[0]) ** 2 + (sx - shift[1]) ** 2 <= peak_exclusion_px**2
        lobe = big & ~near
        sidelobe = corr[lobe].max() if lobe.any() else corr.mean()
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
            peak = max(peak, local[my, mx] / product.size)
        confidence = (
            float(np.clip(1.0 - sidelobe / peak, 0.0, 1.0)) if peak > 0 else 0.0
        )
        return shift - frac, confidence, overlap

    first = correlate(grid_mov, tol_px)
    if isinstance(first[0], str):
        return fallback(*first)
    total_px, confidence, overlap = first
    # Further passes re-crop with the corrected prior: the residual shift is
    # then small, which removes the bias of the window towards zero shift.
    for _ in range(max(int(iterations), 1) - 1):
        moved = Grid(
            tuple(np.array(grid_mov.origin_um) + total_px * pixel),
            grid_mov.pixel_um,
            grid_mov.shape,
        )
        step = correlate(moved, np.full(2, 1.0))
        if isinstance(step[0], str):
            break
        total_px = total_px + step[0]
    measured = tuple(float(v) for v in total_px * pixel)
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
        method="phase_correlation",
        flagged=False,
        reason=None,
        overlap_px=(int(overlap[0]), int(overlap[1])),
        measured_shift_um=measured,
    )
