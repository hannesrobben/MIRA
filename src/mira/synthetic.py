"""Synthetic DHM fields with known shift, piston, tilt and speckle-like noise (MI-22).

The scene is analytic in stage coordinates (µm), so a tile can be sampled at
any (sub-pixel) position without interpolation error: every pixel centre of a
:class:`~mira.products.Grid` is evaluated directly. Registration, piston /
tilt estimation, fusion and stitching tests (MI-20, MI-21, MI-30, MI-40) use
it with known ground truth.

Noise model (decision M4, study with MI-31 / OffAxisHolo K13)::

    sigma² ≈ sigma_t² / (N_frames · N_pos) + sigma_coh² / N_pos

``sigma_t`` is white phase noise that changes from frame to frame,
``sigma_coh`` a spatially correlated (speckle-like) phase pattern with
correlation length ``corr_um`` that is fixed in the optics, i.e. in **camera
coordinates**: it moves with the tile, not with the scene, so averaging over
positions reduces it while averaging frames at one position does not.

Tolerances of tests built on these fields are stated in the tests; final
acceptance tolerances wait for M7.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np
from scipy import ndimage

from .products import Grid, TileProduct
from .utils.exceptions import ValidationError

__all__ = ["Feature", "SyntheticScene", "SpeckleNoise", "SyntheticTile", "make_tile"]


@dataclass(frozen=True)
class Feature:
    """One printed feature of the scene, with smooth (tanh) edges.

    Parameters
    ----------
    kind : {"rect", "disc"}
        Shape of the feature.
    centre_um : tuple of float
        Centre ``(y, x)`` in stage µm.
    size_um : tuple of float
        Full size ``(height_y, width_x)`` for ``"rect"``; ``(diameter,
        diameter)`` for ``"disc"`` (only the first value is used).
    phase_rad : float
        Phase of the plateau relative to the substrate.
    amplitude : float, optional
        Relative amplitude ``sqrt(intensity_rel)`` on the plateau (1.0 = no
        absorption).
    edge_um : float, optional
        Edge width of the tanh profile (> 0).
    """

    kind: str
    centre_um: tuple[float, float]
    size_um: tuple[float, float]
    phase_rad: float
    amplitude: float = 1.0
    edge_um: float = 0.3

    def __post_init__(self) -> None:
        if self.kind not in ("rect", "disc"):
            raise ValidationError(f"Unknown feature kind {self.kind!r}.")
        if self.edge_um <= 0:
            raise ValidationError("edge_um must be positive.")

    def weight(self, y_um: np.ndarray, x_um: np.ndarray) -> np.ndarray:
        """Return the plateau weight in [0, 1] at the given stage positions."""
        dy = y_um - self.centre_um[0]
        dx = x_um - self.centre_um[1]
        if self.kind == "disc":
            r = np.hypot(dy, dx)
            return 0.5 * (1.0 - np.tanh((r - 0.5 * self.size_um[0]) / self.edge_um))
        wy = 0.5 * (1.0 - np.tanh((np.abs(dy) - 0.5 * self.size_um[0]) / self.edge_um))
        wx = 0.5 * (1.0 - np.tanh((np.abs(dx) - 0.5 * self.size_um[1]) / self.edge_um))
        return wy * wx


@dataclass(frozen=True)
class SyntheticScene:
    """Analytic scene: substrate (phase 0, amplitude 1) plus features.

    Parameters
    ----------
    features : sequence of Feature
        Features are added (phase) and multiplied (amplitude).
    """

    features: tuple[Feature, ...] = ()

    def sample(self, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the scene at the pixel centres of ``grid``.

        Parameters
        ----------
        grid : Grid
            Grid in stage coordinates.

        Returns
        -------
        phase_rad : numpy.ndarray
            Unwrapped phase, float64.
        amplitude : numpy.ndarray
            ``sqrt(intensity_rel)``, float64.
        """
        y = grid.origin_um[0] + grid.pixel_um[0] * np.arange(grid.shape[0])
        x = grid.origin_um[1] + grid.pixel_um[1] * np.arange(grid.shape[1])
        yy, xx = np.meshgrid(y, x, indexing="ij")
        phase = np.zeros(grid.shape)
        amplitude = np.ones(grid.shape)
        for feature in self.features:
            w = feature.weight(yy, xx)
            phase += feature.phase_rad * w
            amplitude *= 1.0 + (feature.amplitude - 1.0) * w
        return phase, amplitude

    def substrate_mask(self, grid: Grid, threshold: float = 0.01) -> np.ndarray:
        """Return True where no feature contributes more than ``threshold``."""
        y = grid.origin_um[0] + grid.pixel_um[0] * np.arange(grid.shape[0])
        x = grid.origin_um[1] + grid.pixel_um[1] * np.arange(grid.shape[1])
        yy, xx = np.meshgrid(y, x, indexing="ij")
        total = np.zeros(grid.shape)
        for feature in self.features:
            total += feature.weight(yy, xx)
        return total < threshold


@dataclass(frozen=True)
class SpeckleNoise:
    """Noise parameters of a synthetic capture.

    Parameters
    ----------
    sigma_t_rad : float, optional
        Standard deviation of the white phase noise of one frame.
    sigma_coh_rad : float, optional
        Standard deviation of the correlated (speckle-like) phase pattern.
    corr_um : float, optional
        Correlation length (Gaussian sigma) of the coherent pattern.
    amplitude_rel : float, optional
        Relative standard deviation of the coherent amplitude modulation.
    """

    sigma_t_rad: float = 0.0
    sigma_coh_rad: float = 0.0
    corr_um: float = 1.0
    amplitude_rel: float = 0.0


@dataclass(frozen=True)
class SyntheticTile:
    """A synthetic tile with its ground truth.

    Parameters
    ----------
    field : numpy.ndarray
        complex64 field as a reconstruction would deliver it.
    grid : Grid
        Grid of the **actual** (true) position.
    stage_grid : Grid
        Grid of the reported stage position (``true + stage_error_um``) —
        what a product built from the capture context would carry.
    truth_phase_rad : numpy.ndarray
        Noise-free, unwrapped scene phase on ``grid`` (without piston / tilt).
    piston_rad : float
        Piston added to the tile.
    tilt_rad_per_um : tuple of float
        Linear phase ``(y, x)`` added, rad per µm relative to the tile centre.
    stage_error_um : tuple of float
        Reported minus true position ``(y, x)``.
    noise : SpeckleNoise
        Noise parameters used.
    """

    field: np.ndarray
    grid: Grid
    stage_grid: Grid
    truth_phase_rad: np.ndarray
    piston_rad: float
    tilt_rad_per_um: tuple[float, float]
    stage_error_um: tuple[float, float]
    noise: SpeckleNoise = dc_field(default_factory=SpeckleNoise)

    def to_product(self, *, product_id: str = "synthetic", **kwargs) -> TileProduct:
        """Wrap the tile as a :class:`~mira.products.TileProduct`.

        The product carries the **reported** stage grid (as the service
        would). ``kwargs`` override the defaults ``z_um=0``,
        ``propagation_index=1.508``, ``wavelength_um=0.674994``.
        """
        centre_y, centre_x = self.stage_grid.centre_um
        params = {
            "z_um": 0.0,
            "propagation_index": 1.508,
            "wavelength_um": 0.674994,
            "context_id": None,
            "position_actual_um": {"x": centre_x, "y": centre_y, "z": 0.0},
        }
        params.update(kwargs)
        return TileProduct(field=self.field, grid=self.stage_grid, product_id=product_id, **params)


def _correlated(shape, pixel_um, corr_um, rng) -> np.ndarray:
    """Zero-mean, unit-variance Gaussian random field with correlation ``corr_um``."""
    white = rng.standard_normal(shape)
    sigma_px = (corr_um / pixel_um[0], corr_um / pixel_um[1])
    smooth = ndimage.gaussian_filter(white, sigma_px, mode="wrap")
    std = smooth.std()
    return smooth / std if std > 0 else smooth


def make_tile(
    scene: SyntheticScene,
    centre_um: tuple[float, float],
    shape: tuple[int, int],
    pixel_um: tuple[float, float],
    *,
    piston_rad: float = 0.0,
    tilt_rad_per_um: tuple[float, float] = (0.0, 0.0),
    noise: SpeckleNoise | None = None,
    n_frames: int = 1,
    stage_error_um: tuple[float, float] = (0.0, 0.0),
    coherent_seed: int | None = 0,
    rng: np.random.Generator | int | None = None,
) -> SyntheticTile:
    """Sample one synthetic tile.

    Parameters
    ----------
    scene : SyntheticScene
        The analytic scene.
    centre_um : tuple of float
        True stage position ``(y, x)`` of the image centre; any sub-pixel
        value is exact.
    shape : tuple of int
        ``(ny, nx)``.
    pixel_um : tuple of float
        Pixel size ``(y, x)`` in µm.
    piston_rad : float, optional
        Constant phase added.
    tilt_rad_per_um : tuple of float, optional
        Linear phase ``(y, x)`` in rad / µm, relative to the tile centre.
    noise : SpeckleNoise, optional
        Noise parameters; None = noise-free.
    n_frames : int, optional
        Number of frames averaged in the complex field (white noise only).
    stage_error_um : tuple of float, optional
        Error of the reported stage position ``(y, x)`` (the prior of MI-20).
    coherent_seed : int or None, optional
        Seed of the coherent pattern. The pattern is fixed in camera
        coordinates, so tiles with the same seed share it; None draws a new
        one from ``rng``.
    rng : numpy.random.Generator or int or None, optional
        Generator (or seed) for the white noise.

    Returns
    -------
    SyntheticTile
    """
    if n_frames < 1:
        raise ValidationError("n_frames must be >= 1.")
    noise = noise or SpeckleNoise()
    rng = np.random.default_rng(rng)
    grid = Grid.centred({"y": centre_um[0], "x": centre_um[1]}, pixel_um, shape)
    stage_grid = Grid.centred(
        {"y": centre_um[0] + stage_error_um[0], "x": centre_um[1] + stage_error_um[1]},
        pixel_um,
        shape,
    )
    truth, amplitude = scene.sample(grid)
    ny, nx = grid.shape
    yr = (np.arange(ny) - 0.5 * (ny - 1)) * grid.pixel_um[0]
    xr = (np.arange(nx) - 0.5 * (nx - 1)) * grid.pixel_um[1]
    phase = (
        truth
        + piston_rad
        + tilt_rad_per_um[0] * yr[:, None]
        + tilt_rad_per_um[1] * xr[None, :]
    )
    if noise.sigma_coh_rad > 0 or noise.amplitude_rel > 0:
        crng = rng if coherent_seed is None else np.random.default_rng(coherent_seed)
        pattern = _correlated(grid.shape, grid.pixel_um, noise.corr_um, crng)
        phase = phase + noise.sigma_coh_rad * pattern
        if noise.amplitude_rel > 0:
            pattern_a = _correlated(grid.shape, grid.pixel_um, noise.corr_um, crng)
            amplitude = amplitude * np.clip(1.0 + noise.amplitude_rel * pattern_a, 0.05, None)
    field = np.zeros(grid.shape, np.complex128)
    for _ in range(n_frames):
        frame_phase = phase
        if noise.sigma_t_rad > 0:
            frame_phase = phase + noise.sigma_t_rad * rng.standard_normal(grid.shape)
        field += amplitude * np.exp(1j * frame_phase)
    field /= n_frames
    return SyntheticTile(
        field=field.astype(np.complex64),
        grid=grid,
        stage_grid=stage_grid,
        truth_phase_rad=truth,
        piston_rad=float(piston_rad),
        tilt_rad_per_um=(float(tilt_rad_per_um[0]), float(tilt_rad_per_um[1])),
        stage_error_um=(float(stage_error_um[0]), float(stage_error_um[1])),
        noise=noise,
    )
