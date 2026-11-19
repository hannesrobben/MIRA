"""Noise metrics shared by fusion, stitching, comparison and the N-study (MI-31).

One definition of "the noise" for everybody (fusion MI-30, stitching MI-40,
the comparison, OffAxisHolo K13 and the speckle / position study of M4):

* :func:`reference_noise` — standard deviation of the phase in a reference
  region (substrate) after removing piston or a plane; computed on the
  complex field relative to its weighted mean, so wrapped data are fine as
  long as the noise stays well below π;
* :func:`radial_psd` — radially averaged power spectral density of the same
  region (frequency in 1/µm);
* :func:`correlation_length_um` — 1/e radius of the normalised
  autocorrelation (for the optics-fixed coherent pattern: the minimum useful
  offset between capture positions);
* :func:`fit_noise_model` — least-squares fit of
  ``sigma² = sigma_t² / (N_frames · N_pos) + sigma_coh² / N_pos`` (decision
  M4) to measured noise values, and :func:`positions_needed` from it.

The PSD here is a minimal implementation; once OffAxisHolo P15 provides the
radial PSD, MI-75 replaces it by a thin wrapper.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .utils.exceptions import ValidationError

__all__ = [
    "NoiseStats",
    "reference_noise",
    "radial_psd",
    "correlation_length_um",
    "NoiseModel",
    "fit_noise_model",
    "positions_needed",
]

_DETREND = ("none", "piston", "plane")


@dataclass(frozen=True)
class NoiseStats:
    """Noise in a reference region.

    Parameters
    ----------
    sigma_rad : float
        Weighted standard deviation of the detrended phase.
    mad_sigma_rad : float
        Robust estimate ``1.4826 · median(|phase - median|)``.
    n_pixels : int
        Pixels used.
    detrend : str
        ``"none"``, ``"piston"`` or ``"plane"``.
    """

    sigma_rad: float
    mad_sigma_rad: float
    n_pixels: int
    detrend: str


def _as_phase(values, mask, weight):
    """Phase relative to the weighted complex mean, plus the used mask."""
    arr = np.asarray(values)
    if mask is None:
        mask = np.ones(arr.shape, bool)
    mask = np.asarray(mask, bool) & np.isfinite(arr)
    if mask.shape != arr.shape:
        raise ValidationError(f"mask {mask.shape} does not match data {arr.shape}.")
    w = np.ones(arr.shape) if weight is None else np.asarray(weight, float)
    if np.iscomplexobj(arr):
        ref = np.sum((arr * w)[mask])
        phase = np.angle(arr * np.conj(ref)) if abs(ref) > 0 else np.angle(arr)
    else:
        phase = arr.astype(float)
    return phase, mask, w


def reference_noise(
    values,
    mask=None,
    *,
    detrend: str = "plane",
    weight=None,
    min_pixels: int = 16,
) -> NoiseStats:
    """Noise of a phase map or complex field in a reference region.

    Parameters
    ----------
    values : array_like
        Phase in rad (float) or complex field.
    mask : array_like of bool, optional
        Reference region (e.g. substrate); default all finite pixels.
    detrend : {"none", "piston", "plane"}, optional
        What is removed before the standard deviation.
    weight : array_like, optional
        Weight per pixel (e.g. ``|U|²``).
    min_pixels : int, optional
        Minimum number of pixels.

    Returns
    -------
    NoiseStats
    """
    if detrend not in _DETREND:
        raise ValidationError(f"detrend must be one of {_DETREND}; got {detrend!r}.")
    phase, mask, w = _as_phase(values, mask, weight)
    n = int(mask.sum())
    if n < min_pixels:
        raise ValidationError(f"Reference region has {n} pixels; need {min_pixels}.")
    yy, xx = np.nonzero(mask)
    p = phase[mask]
    ww = w[mask]
    if detrend == "plane":
        design = np.column_stack([np.ones(n), yy, xx]) * np.sqrt(ww)[:, None]
        coef, *_ = np.linalg.lstsq(design, p * np.sqrt(ww), rcond=None)
        p = p - (coef[0] + coef[1] * yy + coef[2] * xx)
    elif detrend == "piston":
        p = p - np.sum(ww * p) / np.sum(ww)
    mean = np.sum(ww * p) / np.sum(ww)
    sigma = math.sqrt(float(np.sum(ww * (p - mean) ** 2) / np.sum(ww)))
    mad = 1.4826 * float(np.median(np.abs(p - np.median(p))))
    return NoiseStats(sigma_rad=sigma, mad_sigma_rad=mad, n_pixels=n, detrend=detrend)


def _detrended_patch(values, mask, detrend):
    phase, mask, _ = _as_phase(values, mask, None)
    if detrend != "none":
        stats_mask = mask
        yy, xx = np.nonzero(stats_mask)
        p = phase[stats_mask]
        if detrend == "plane":
            design = np.column_stack([np.ones(p.size), yy, xx])
            coef, *_ = np.linalg.lstsq(design, p, rcond=None)
            ys, xs = np.indices(phase.shape)
            phase = phase - (coef[0] + coef[1] * ys + coef[2] * xs)
        else:
            phase = phase - p.mean()
    out = np.where(mask, phase, 0.0)
    return out, mask


def radial_psd(
    values,
    pixel_um: Sequence[float],
    mask=None,
    *,
    n_bins: int = 32,
    detrend: str = "plane",
) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectral density.

    Pixels outside ``mask`` are set to zero after detrending and the power is
    rescaled by the masked fraction. Normalised so that the integral of the
    2-D PSD over frequency equals the variance (Parseval).

    Parameters
    ----------
    values : array_like
        Phase in rad or complex field.
    pixel_um : sequence of float
        Pixel size ``(y, x)``.
    mask : array_like of bool, optional
        Region used.
    n_bins : int, optional
        Number of radial bins up to the Nyquist frequency of the coarser axis.
    detrend : {"none", "piston", "plane"}, optional

    Returns
    -------
    freq_per_um : numpy.ndarray
        Bin centres in 1/µm.
    psd_rad2_um2 : numpy.ndarray
        Mean PSD per bin in rad² · µm²; NaN for empty bins.
    """
    if detrend not in _DETREND:
        raise ValidationError(f"detrend must be one of {_DETREND}; got {detrend!r}.")
    patch, mask = _detrended_patch(values, mask, detrend)
    py, px = (float(v) for v in pixel_um)
    ny, nx = patch.shape
    fraction = max(mask.mean(), 1e-12)
    spec = np.abs(np.fft.fft2(patch)) ** 2 * (py * px) / (ny * nx) / fraction
    fy = np.fft.fftfreq(ny, py)[:, None]
    fx = np.fft.fftfreq(nx, px)[None, :]
    fr = np.hypot(fy, fx)
    f_max = 0.5 / max(py, px)
    edges = np.linspace(0.0, f_max, n_bins + 1)
    idx = np.digitize(fr.ravel(), edges) - 1
    valid = (idx >= 0) & (idx < n_bins)
    sums = np.bincount(idx[valid], spec.ravel()[valid], minlength=n_bins)
    counts = np.bincount(idx[valid], minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        psd = np.where(counts > 0, sums / counts, np.nan)
    return 0.5 * (edges[:-1] + edges[1:]), psd


def correlation_length_um(
    values,
    pixel_um: Sequence[float],
    mask=None,
    *,
    detrend: str = "plane",
) -> float:
    """1/e radius of the normalised, radially averaged autocorrelation.

    For a Gaussian-filtered white field with kernel standard deviation ``s``
    (as :mod:`mira.synthetic` draws the coherent pattern with ``s =
    corr_um``) the result is ``2 · s``.

    Parameters
    ----------
    values : array_like
        Phase in rad or complex field.
    pixel_um : sequence of float
        Pixel size ``(y, x)``.
    mask : array_like of bool, optional
        Region used.
    detrend : {"none", "piston", "plane"}, optional

    Returns
    -------
    float
        Correlation length in µm (linear interpolation between pixels).
    """
    patch, mask = _detrended_patch(values, mask, detrend)
    ny, nx = patch.shape
    pad = np.zeros((2 * ny, 2 * nx))
    pad[:ny, :nx] = patch
    mpad = np.zeros((2 * ny, 2 * nx))
    mpad[:ny, :nx] = mask
    acf = np.fft.ifft2(np.abs(np.fft.fft2(pad)) ** 2).real
    support = np.fft.ifft2(np.abs(np.fft.fft2(mpad)) ** 2).real
    acf = np.where(support > 0.5, acf / np.maximum(support, 1e-12), 0.0)
    if acf[0, 0] <= 0:
        raise ValidationError("Zero variance; no correlation length.")
    acf = np.fft.fftshift(acf / acf[0, 0]).ravel()
    support = np.fft.fftshift(support).ravel()
    py, px = (float(v) for v in pixel_um)
    yy, xx = np.indices((2 * ny, 2 * nx))
    r = np.hypot((yy - ny) * py, (xx - nx) * px).ravel()
    step = min(py, px)
    bins = np.arange(0.0, min(ny * py, nx * px) / 2, step)
    idx = np.digitize(r, bins) - 1
    ok = (idx >= 0) & (idx < bins.size) & (support > 0.5)
    sums = np.bincount(idx[ok], acf[ok], minlength=bins.size)
    counts = np.bincount(idx[ok], minlength=bins.size)
    profile = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    radii = np.bincount(idx[ok], r[ok], minlength=bins.size) / np.maximum(counts, 1)
    target = math.exp(-1.0)
    below = np.nonzero(profile < target)[0]
    if below.size == 0 or below[0] == 0:
        raise ValidationError(
            "Autocorrelation does not fall below 1/e inside the region."
        )
    i = below[0]
    r0, r1 = radii[i - 1], radii[i]
    a0, a1 = profile[i - 1], profile[i]
    return float(r0 + (a0 - target) * (r1 - r0) / (a0 - a1))


@dataclass(frozen=True)
class NoiseModel:
    """Fitted M4 noise model ``sigma² = sigma_t² / (Nf · Np) + sigma_coh² / Np``.

    Parameters
    ----------
    sigma_t_rad : float
        Frame-to-frame (white) part.
    sigma_coh_rad : float
        Optics-fixed coherent part, reduced only by more positions.
    rms_residual_rad2 : float
        RMS residual of the fit in rad².
    """

    sigma_t_rad: float
    sigma_coh_rad: float
    rms_residual_rad2: float

    def sigma(self, n_frames, n_positions):
        """Predicted noise in rad for the given frame and position counts."""
        nf = np.asarray(n_frames, float)
        npos = np.asarray(n_positions, float)
        return np.sqrt(self.sigma_t_rad**2 / (nf * npos) + self.sigma_coh_rad**2 / npos)


def fit_noise_model(n_frames, n_positions, sigma_rad) -> NoiseModel:
    """Fit the M4 noise model to measured noise values.

    Parameters
    ----------
    n_frames, n_positions : array_like of int
        Frames per position and number of positions of every measurement.
    sigma_rad : array_like of float
        Measured noise (e.g. :func:`reference_noise`).

    Returns
    -------
    NoiseModel
        Non-negative variances (a negative LS solution is clipped to 0 and the
        other term refitted).
    """
    nf = np.asarray(n_frames, float).ravel()
    npos = np.asarray(n_positions, float).ravel()
    s2 = np.asarray(sigma_rad, float).ravel() ** 2
    if not (nf.size == npos.size == s2.size) or nf.size < 2:
        raise ValidationError("Need at least two measurements of equal length.")
    if np.any(nf < 1) or np.any(npos < 1):
        raise ValidationError("n_frames and n_positions must be >= 1.")
    design = np.column_stack([1.0 / (nf * npos), 1.0 / npos])
    if np.linalg.matrix_rank(design) < 2:
        raise ValidationError(
            "The design cannot separate the two terms; vary n_frames at fixed positions."
        )
    coef, *_ = np.linalg.lstsq(design, s2, rcond=None)
    if coef[0] < 0 or coef[1] < 0:
        col = 1 if coef[0] < 0 else 0
        single = max(
            float(np.dot(design[:, col], s2) / np.dot(design[:, col], design[:, col])),
            0.0,
        )
        coef = np.zeros(2)
        coef[col] = single
    resid = design @ coef - s2
    return NoiseModel(
        sigma_t_rad=math.sqrt(coef[0]),
        sigma_coh_rad=math.sqrt(coef[1]),
        rms_residual_rad2=float(np.sqrt(np.mean(resid**2))),
    )


def positions_needed(model: NoiseModel, target_sigma_rad: float, n_frames: int) -> int:
    """Smallest number of positions that reaches ``target_sigma_rad``.

    Parameters
    ----------
    model : NoiseModel
    target_sigma_rad : float
        Wanted noise.
    n_frames : int
        Frames per position.

    Returns
    -------
    int
    """
    if target_sigma_rad <= 0 or n_frames < 1:
        raise ValidationError("target_sigma_rad must be > 0 and n_frames >= 1.")
    per_position = model.sigma_t_rad**2 / n_frames + model.sigma_coh_rad**2
    return max(1, math.ceil(per_position / target_sigma_rad**2 - 1e-12))
