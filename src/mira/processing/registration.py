"""
Content-based registration refinement.

The geometric alignment in `alignment.py` (de-rotate, rescale, center the
detected-structure mask) is only a *coarse* initialization: it lines up the
bounding geometry of two structures, not their internal content. When two
instruments detect masks of slightly different shape, their centroids do not
coincide with the same physical point, so the actual pattern (e.g. a grating)
ends up shifted -- visible as an out-of-phase profile.

This module adds a *fine* registration stage that estimates a small residual
transform directly from image intensities, locking the real content on top of
the geometric pre-alignment (coarse-to-fine).

Refiners are pluggable via the `RegistrationRefiner` interface so new strategies
(mutual information, feature matching, ...) can be added without touching the
alignment pipeline.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from ..utils.exceptions import AlignmentError

logger = logging.getLogger(__name__)


def _normalize(image: np.ndarray, blur_sigma: float = 1.0) -> np.ndarray:
    """
    Map an arbitrary-range image to float32 in [0, 1], robust to outliers.

    Cross-modality images (e.g. phase in [0, 6] vs [-0.1, 0.4]) must share a
    common intensity scale before correlation-based registration. We clip to the
    1st/99th percentile so a few hot pixels do not dominate the scaling, then
    optionally smooth to stabilise the optimiser.
    """
    img = np.nan_to_num(image.astype(np.float32), nan=0.0)
    lo, hi = np.percentile(img, (1.0, 99.0))
    if hi <= lo:
        hi = lo + 1e-6
    img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    if blur_sigma and blur_sigma > 0:
        import cv2

        ksize = int(2 * round(3 * blur_sigma) + 1)
        img = cv2.GaussianBlur(img, (ksize, ksize), blur_sigma)
    return img


class RegistrationRefiner(ABC):
    """
    Strategy interface for content-based registration.

    An implementation estimates a 2x3 affine matrix (OpenCV convention) mapping
    the already coarsely-aligned ``moving`` image onto ``fixed``. The matrix is
    meant to be applied with ``cv2.warpAffine(moving, M, ...)``.
    """

    @abstractmethod
    def estimate(self, moving: np.ndarray, fixed: np.ndarray) -> np.ndarray:
        """Return a 2x3 float32 affine matrix mapping ``moving`` -> ``fixed``."""
        raise NotImplementedError


class NoRefinement(RegistrationRefiner):
    """Identity refiner -- keeps the pure geometric (centroid) alignment."""

    def estimate(self, moving: np.ndarray, fixed: np.ndarray) -> np.ndarray:
        return np.eye(2, 3, dtype=np.float32)


class ECCRefinement(RegistrationRefiner):
    """
    Enhanced Correlation Coefficient registration (Evangelidis & Psarakis).

    Maximises the normalised correlation between the two images, making it
    invariant to linear intensity differences -- ideal for cross-modality data
    where one channel is scaled/offset relative to the other. Because the
    geometric stage already removes most rotation/scale, ``euclidean`` (rotation
    + translation) is a robust default; use ``translation`` if rotation is
    already trusted, or ``affine`` for residual shear/scale.
    """

    # Names of the cv2 motion constants; resolved in estimate() so that
    # importing this module does not import OpenCV (MI-05).
    _MOTION = {
        "translation": "MOTION_TRANSLATION",
        "euclidean": "MOTION_EUCLIDEAN",
        "affine": "MOTION_AFFINE",
    }

    def __init__(
        self,
        motion: str = "euclidean",
        n_iterations: int = 500,
        eps: float = 1e-6,
        blur_sigma: float = 1.0,
    ) -> None:
        if motion not in self._MOTION:
            raise AlignmentError(
                f"ECCRefinement: unknown motion '{motion}'. "
                f"Choose one of {sorted(self._MOTION)}."
            )
        self.motion = motion
        self.n_iterations = int(n_iterations)
        self.eps = float(eps)
        self.blur_sigma = float(blur_sigma)

    def estimate(self, moving: np.ndarray, fixed: np.ndarray) -> np.ndarray:
        import cv2

        fixed_n = _normalize(fixed, self.blur_sigma)
        moving_n = _normalize(moving, self.blur_sigma)

        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.n_iterations,
            self.eps,
        )
        try:
            # template = fixed, input = moving -> warp maps moving onto fixed.
            _, warp = cv2.findTransformECC(
                fixed_n,
                moving_n,
                warp,
                getattr(cv2, self._MOTION[self.motion]),
                criteria,
                None,
                5,
            )
        except cv2.error as exc:
            # ECC did not converge (e.g. too little overlap). Fall back to the
            # geometric pre-alignment rather than corrupting the result.
            logger.warning(
                "ECCRefinement: did not converge (%s); keeping geometric " "alignment.",
                exc,
            )
            return np.eye(2, 3, dtype=np.float32)
        return warp.astype(np.float32)


class PhaseCorrelationRefinement(RegistrationRefiner):
    """
    Pure-translation refinement via FFT phase cross-correlation.

    Fast and very robust for residual *shifts* once rotation/scale are handled.
    A Hann window suppresses border effects so the global pattern envelope --
    not the canvas edges -- drives the estimate. Requires scikit-image.
    """

    def __init__(self, upsample_factor: int = 10, blur_sigma: float = 1.0) -> None:
        self.upsample_factor = int(upsample_factor)
        self.blur_sigma = float(blur_sigma)

    def estimate(self, moving: np.ndarray, fixed: np.ndarray) -> np.ndarray:
        try:
            from skimage.registration import phase_cross_correlation
        except ImportError as exc:
            raise AlignmentError(
                "PhaseCorrelationRefinement requires scikit-image "
                "(pip install scikit-image)."
            ) from exc

        fixed_n = _normalize(fixed, self.blur_sigma)
        moving_n = _normalize(moving, self.blur_sigma)

        win = np.outer(np.hanning(fixed_n.shape[0]), np.hanning(fixed_n.shape[1]))
        shift, _, _ = phase_cross_correlation(
            fixed_n * win, moving_n * win, upsample_factor=self.upsample_factor
        )
        dy, dx = float(shift[0]), float(shift[1])
        # Translation that moves `moving` onto `fixed`.
        return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)


_REGISTRY = {
    "none": NoRefinement,
    "ecc": ECCRefinement,
    "phase": PhaseCorrelationRefinement,
}


def get_refiner(
    spec: Union[bool, str, RegistrationRefiner, None],
) -> RegistrationRefiner:
    """
    Resolve a user-facing ``refine`` argument into a RegistrationRefiner.

    Accepts a RegistrationRefiner (returned as-is), a name ('ecc' | 'phase' |
    'none'), True (-> default ECC), or False/None (-> NoRefinement).
    """
    if isinstance(spec, RegistrationRefiner):
        return spec
    if spec is True:
        return ECCRefinement()
    if spec is False or spec is None:
        return NoRefinement()
    if isinstance(spec, str):
        key = spec.lower()
        if key not in _REGISTRY:
            raise AlignmentError(
                f"Unknown refiner '{spec}'. Choose one of {sorted(_REGISTRY)}."
            )
        return _REGISTRY[key]()
    raise AlignmentError(f"Cannot interpret refine={spec!r} as a refiner.")
