"""Forward model target height ↔ expected phase, and the comparison domain (MI-51).

The phase ↔ height formulas are owned by OffAxisHolo
(``offaxisholo.reconstruction.fieldops``, K20; decision G4 / G7). MIRA must
run without OffAxisHolo, so they are **mirrored** here unchanged;
``tests/test_forward.py`` checks parity whenever OffAxisHolo is installed::

    phase_rad = phase_sign · 2π · (n_structure − n_surround) · h_um / wavelength_um

In-situ ``n_structure`` is the ``cured_in_resin`` index and ``n_surround`` the
liquid resin (``uncured``); developed structures are surrounded by air. No
index has a default (G7): every value names its source or the call fails.

Comparison domain (M5 = OM4 of nanofactorysystem): chosen automatically from
what is available — the phase domain is preferred; the height domain is used
when it is fixed or when the target is a height map and the optics are not
known — or fixed by the caller.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .utils.exceptions import ValidationError

__all__ = [
    "Optics",
    "phase_to_height_um",
    "height_to_phase_rad",
    "choose_domain",
    "to_domain",
]


def _check_optics(wavelength_um, n_structure, n_surround, phase_sign):
    # Mirror of offaxisholo.reconstruction.fieldops._check_optics (K20).
    if phase_sign not in (1, -1):
        raise ValueError(f"phase_sign must be +1 or -1, got {phase_sign!r}.")
    if not wavelength_um > 0:
        raise ValueError(f"wavelength_um must be positive, got {wavelength_um!r}.")
    if not (n_structure > 0 and n_surround > 0):
        raise ValueError(
            f"Refractive indices must be positive (n_structure={n_structure!r}, "
            f"n_surround={n_surround!r})."
        )
    delta_n = n_structure - n_surround
    if delta_n == 0:
        raise ValueError(
            "n_structure equals n_surround; phase and height are not related."
        )
    return delta_n


def phase_to_height_um(phase_rad, *, wavelength_um, n_structure, n_surround, phase_sign):
    """Height in µm of a homogeneous structure from its unwrapped phase.

    Mirror of ``offaxisholo.reconstruction.fieldops.phase_to_height_um`` (K20).

    Parameters
    ----------
    phase_rad : array or float
        Unwrapped phase relative to the substrate.
    wavelength_um : float
        Vacuum wavelength (DHM: 0.674994 µm).
    n_structure, n_surround : float
        Indices of the structure and of its surrounding.
    phase_sign : {1, -1}
        Sign convention of the setup (parameter set).

    Returns
    -------
    array or float
        Height in µm.
    """
    delta_n = _check_optics(wavelength_um, n_structure, n_surround, phase_sign)
    return phase_sign * wavelength_um * phase_rad / (2 * math.pi * delta_n)


def height_to_phase_rad(height_um, *, wavelength_um, n_structure, n_surround, phase_sign):
    """Phase in rad of a homogeneous structure of known height.

    Inverse of :func:`phase_to_height_um`; mirror of
    ``offaxisholo.reconstruction.fieldops.height_to_phase_rad`` (K20).

    Parameters
    ----------
    height_um : array or float
    wavelength_um, n_structure, n_surround, phase_sign
        See :func:`phase_to_height_um`.

    Returns
    -------
    array or float
        Phase in rad.
    """
    delta_n = _check_optics(wavelength_um, n_structure, n_surround, phase_sign)
    return phase_sign * 2 * math.pi * delta_n * height_um / wavelength_um


@dataclass(frozen=True)
class Optics:
    """Optical parameters of the phase ↔ height conversion, with their sources.

    Parameters
    ----------
    wavelength_um : float
    n_structure, n_surround : float
    phase_sign : {1, -1}
    sources : mapping, optional
        Where each value came from (recorded in the report).
    """

    wavelength_um: float
    n_structure: float
    n_surround: float
    phase_sign: int
    sources: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        try:
            _check_optics(self.wavelength_um, self.n_structure, self.n_surround, self.phase_sign)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

    @property
    def delta_n(self) -> float:
        """``n_structure − n_surround``."""
        return self.n_structure - self.n_surround

    def as_kwargs(self) -> dict:
        """Keyword arguments of :func:`phase_to_height_um` / :func:`height_to_phase_rad`."""
        return {
            "wavelength_um": self.wavelength_um,
            "n_structure": self.n_structure,
            "n_surround": self.n_surround,
            "phase_sign": self.phase_sign,
        }

    def with_delta_n(self, delta_n: float, source: str) -> Optics:
        """Return optics whose Δn is replaced by a calibrated value.

        Use the ``delta_n`` of :meth:`mira.StructureProcessor.calculate_delta_n`
        (``DeltaNResult``) measured against an ex-situ reference; ``n_surround``
        is kept and ``n_structure`` becomes ``n_surround + delta_n``.

        Parameters
        ----------
        delta_n : float
            Calibrated index difference.
        source : str
            Where the calibration comes from (recorded in ``sources``).
        """
        sources = dict(self.sources or {})
        sources["n_structure"] = f"n_surround + calibrated delta_n ({source})"
        return Optics(self.wavelength_um, self.n_surround + float(delta_n),
                      self.n_surround, self.phase_sign, sources)

    @classmethod
    def from_sources(
        cls,
        *,
        wavelength_um: float | None,
        n_structure: float | None,
        n_surround: float | None,
        phase_sign: int | None,
        sources: Mapping[str, str],
    ) -> Optics:
        """Build optics; every missing value raises with the place it was sought.

        Parameters
        ----------
        wavelength_um, n_structure, n_surround, phase_sign
            Values or None.
        sources : mapping
            Name → description of where the value was looked up.

        Raises
        ------
        ValidationError
            Listing every missing value and its expected source (G7).
        """
        values = {
            "wavelength_um": wavelength_um,
            "n_structure": n_structure,
            "n_surround": n_surround,
            "phase_sign": phase_sign,
        }
        missing = [f"{k} (from {sources.get(k, '?')})" for k, v in values.items() if v is None]
        if missing:
            raise ValidationError(
                "Missing optical parameters, no defaults (G7): " + ", ".join(missing)
            )
        return cls(float(wavelength_um), float(n_structure), float(n_surround),
                   int(phase_sign), dict(sources))


def choose_domain(
    target_domain: str, *, optics_known: bool, fixed: str | None = None
) -> str:
    """Choose the comparison domain (M5 = OM4).

    Parameters
    ----------
    target_domain : {"height", "phase"}
        Domain of the target raster.
    optics_known : bool
        Whether wavelength, indices and phase sign are all available.
    fixed : {"height", "phase"} or None, optional
        Domain fixed by the caller.

    Returns
    -------
    str
        ``"phase"`` or ``"height"``.

    Raises
    ------
    ValidationError
        If the requested or required conversion needs optics that are not
        known.
    """
    for name, value in (("target_domain", target_domain), ("fixed", fixed)):
        if value not in ("height", "phase") and not (name == "fixed" and value is None):
            raise ValidationError(f"{name} must be 'height' or 'phase'; got {value!r}.")
    if fixed is not None:
        if fixed != target_domain and not optics_known:
            raise ValidationError(
                f"Comparison fixed to {fixed!r} needs the optics to convert the "
                f"{target_domain!r} target."
            )
        if fixed == "height" and not optics_known:
            raise ValidationError("The height domain needs the optics for the product phase.")
        return fixed
    if optics_known:
        return "phase"
    if target_domain == "phase":
        return "phase"
    raise ValidationError(
        "A height target can only be compared with the phase product when the optics "
        "(wavelength, n_structure, n_surround, phase_sign) are known."
    )


def to_domain(values: Any, source_domain: str, domain: str, optics: Optics | None) -> Any:
    """Convert ``values`` from ``source_domain`` into ``domain``.

    Parameters
    ----------
    values : array or float
        Height in µm or phase in rad.
    source_domain, domain : {"height", "phase"}
    optics : Optics or None
        Needed whenever the two domains differ.

    Returns
    -------
    array or float
    """
    if source_domain == domain:
        return values
    if optics is None:
        raise ValidationError(f"Converting {source_domain} → {domain} needs optics.")
    if domain == "phase":
        return height_to_phase_rad(values, **optics.as_kwargs())
    return phase_to_height_um(values, **optics.as_kwargs())
