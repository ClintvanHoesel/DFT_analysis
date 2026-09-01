"""Convert emission and absorption spectra to display RGB colours.

The colour calculations use the CIE 1931 2° standard observer through
``colour-science``. Spectra are sampled on a one-nanometre grid from 360 to
780 nm before conversion to XYZ and sRGB. Public functions return a NumPy
array containing three clipped channels in the ``[0, 1]`` range.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

try:
    import colour
except ImportError as exc:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "dft_utils.colour_utils requires 'colour-science'; "
        "install the project's visualization dependencies"
    ) from exc

from .units import Units

__all__ = [
    "get_RGB_colour_from_molar_absoprtion",
    "get_RGB_colour_from_molar_absoprtion_length",
    "get_RGB_colour_from_molar_absoprtion_norm",
    "get_RGB_colour_from_molar_absorption",
    "get_RGB_colour_from_molar_absorption_length",
    "get_RGB_colour_from_molar_absorption_norm",
    "get_RGB_colour_from_spectrum",
]

VISIBLE_WAVELENGTHS_NM = np.arange(360.0, 781.0, 1.0)


def _as_float_array(values: Iterable[float], *, name: str) -> np.ndarray:
    """Convert an iterable to a one-dimensional floating-point array."""
    try:
        return np.asarray(list(values), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric iterable") from exc


def _non_negative_scalar(value: float, *, name: str) -> float:
    """Return a finite, non-negative scalar or raise a helpful error."""
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite, non-negative scalar") from exc
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be a finite, non-negative scalar")
    return scalar


def _as_spectrum(
    wavelengths: Iterable[float], values: Iterable[float], *, name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Validate, sort, and deduplicate a spectrum."""
    x = _as_float_array(wavelengths, name=f"{name} wavelengths")
    y = _as_float_array(values, name=f"{name} values")
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(f"{name} wavelengths and values must be one-dimensional")
    if x.size == 0:
        raise ValueError(f"{name} spectrum cannot be empty")
    if x.shape != y.shape:
        raise ValueError(f"{name} wavelengths and values must have the same length")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"{name} wavelengths and values must be finite")

    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]
    unique_x, first_indices, counts = np.unique(
        x, return_index=True, return_counts=True
    )
    if np.any(counts > 1):
        y = np.add.reduceat(y, first_indices) / counts
    return unique_x, y


def _sample_visible(
    wavelengths: Iterable[float], values: Iterable[float], *, name: str
) -> np.ndarray:
    """Interpolate a spectrum onto the visible one-nanometre grid."""
    x, y = _as_spectrum(wavelengths, values, name=name)
    return np.interp(VISIBLE_WAVELENGTHS_NM, x, y, left=0.0, right=0.0)


def _xyz_to_rgb(values: np.ndarray, *, normalize: bool) -> np.ndarray:
    """Convert sampled spectral values to clipped sRGB."""
    values = np.asarray(values, dtype=float)
    if normalize:
        maximum = float(np.max(values))
        if maximum <= 0.0:
            return np.zeros(3, dtype=float)
        values = values / maximum

    distribution = colour.SpectralDistribution(
        dict(zip(VISIBLE_WAVELENGTHS_NM, values, strict=True))
    )
    xyz = np.asarray(colour.sd_to_XYZ(distribution), dtype=float)
    rgb = np.asarray(colour.XYZ_to_sRGB(xyz / 100.0), dtype=float)
    return np.clip(rgb, 0.0, 1.0)


def _absorption_to_rgb(absorption: np.ndarray, factor: float) -> np.ndarray:
    """Convert molar absorption data to RGB using Beer-Lambert transmission."""
    factor = _non_negative_scalar(factor, name="absorption factor")
    absorption = np.clip(absorption, 0.0, None)
    transmittance = np.power(10.0, -absorption * factor)
    return _xyz_to_rgb(transmittance, normalize=False)


def get_RGB_colour_from_spectrum(
    wavelengths: Iterable[float], intensity: Iterable[float]
) -> np.ndarray:
    """Return the approximate sRGB colour of an emission spectrum.

    Wavelengths are in nanometres. Samples may be unsorted and may contain
    duplicates; duplicate values are averaged. Negative intensities are
    clipped to zero because they do not represent emitted power.
    """
    sampled = _sample_visible(wavelengths, intensity, name="emission")
    return _xyz_to_rgb(np.clip(sampled, 0.0, None), normalize=True)


def get_RGB_colour_from_molar_absorption_norm(
    wavelengths: Iterable[float], absorption: Iterable[float], norm: float = 0.001
) -> np.ndarray:
    """Return colour for absorption normalized to a target minimum transfer.

    ``norm`` is the transmission at the wavelength with the greatest
    non-negative absorption and must be between zero and one, exclusive.
    """
    try:
        norm = float(norm)
    except (TypeError, ValueError) as exc:
        raise ValueError("norm must be greater than 0 and less than 1") from exc
    if not np.isfinite(norm) or not 0.0 < norm < 1.0:
        raise ValueError("norm must be greater than 0 and less than 1")
    sampled = np.clip(
        _sample_visible(wavelengths, absorption, name="absorption"), 0.0, None
    )
    maximum = float(np.max(sampled))
    if maximum == 0.0:
        return _xyz_to_rgb(np.ones_like(sampled), normalize=False)
    factor = -np.log10(norm) / maximum
    return _absorption_to_rgb(sampled, factor)


def get_RGB_colour_from_molar_absorption_length(
    wavelengths: Iterable[float],
    absorption: Iterable[float],
    length: float = 0.01,
    concentration: float = 1e-27,
) -> np.ndarray:
    """Return colour using Beer-Lambert transmission for a sample.

    ``length`` and ``concentration`` retain the units expected by the
    original project: their product is divided by Avogadro's constant before
    being applied to the molar absorption data.
    """
    length = _non_negative_scalar(length, name="length")
    concentration = _non_negative_scalar(concentration, name="concentration")
    sampled = _sample_visible(wavelengths, absorption, name="absorption")
    factor = length * concentration / Units.constants["NA"]
    return _absorption_to_rgb(sampled, factor)


def get_RGB_colour_from_molar_absorption(
    wavelengths: Iterable[float],
    absorption: Iterable[float],
    factor: float = 1.0 / 3000.0,
) -> np.ndarray:
    """Return colour using a caller-supplied Beer-Lambert factor."""
    sampled = _sample_visible(wavelengths, absorption, name="absorption")
    return _absorption_to_rgb(sampled, factor)


get_RGB_colour_from_molar_absoprtion_norm = get_RGB_colour_from_molar_absorption_norm
get_RGB_colour_from_molar_absoprtion_length = (
    get_RGB_colour_from_molar_absorption_length
)
get_RGB_colour_from_molar_absoprtion = get_RGB_colour_from_molar_absorption
