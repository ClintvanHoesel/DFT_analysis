"""Radius calculations from emission and absorption spectra."""

from __future__ import annotations

from copy import deepcopy
from math import factorial, isfinite, pi
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.integrate import quad

from .base_utils import extrapolate_diff
from .spectra_averager import Data_Series
from .spectrum_converter import (
    MP_to_Absorption,
    MP_to_Emission,
    Multipole_Absorption_Spectrum,
    Multipole_Emission_Spectrum,
)
from .units import Units

EPS = 1e-13
DEFAULT_ALPHA = 2 * pi * Units.constants["hbar"] * Units.constants["c"]
DEFAULT_MU_0 = 1.0 / (Units.constants["eps0"] * Units.constants["c"] ** 2)

__all__ = [
    "Foerster_Radius_calculator",
    "Foerster_Radius_calculator_energy",
    "Forster_Radius_calculator",
    "Forster_Radius_calculator_energy",
    "MPMP_ReabsRadius_calculator",
    "MPMP_Radius_calculator",
    "MULTIPOLE_NUM_LET_DICT",
    "OD_Radius_calculator",
    "QD_Radius_calculator",
    "delta_E_into_wavelength",
    "e_to_wav",
    "emission_counts_to_intensity",
    "emission_counts_to_internsity",
    "get_normalised_emission_spectrum",
    "get_norm",
    "kappa_sq_f",
    "load_excel_spectrum",
    "normalise_counts",
    "normalise_intensity",
    "wav_to_e",
]

MULTIPOLE_NUM_LET_DICT = {1: "D", 2: "Q", 3: "O"}


def _as_float_array(value: Any, *, name: str) -> np.ndarray:
    """Convert *value* to a finite floating-point array."""
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_pair(
    value_x: Any, value_y: Any, *, x_name: str, y_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Validate two equally sized one-dimensional numeric arrays."""
    x_array = _as_float_array(value_x, name=x_name)
    y_array = _as_float_array(value_y, name=y_name)
    if x_array.shape != y_array.shape:
        raise ValueError(f"{x_name} and {y_name} must have the same shape")
    return x_array, y_array


def _positive(value: float, *, name: str) -> float:
    """Return a finite positive scalar."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive scalar") from exc
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return result


def _non_negative(value: float, *, name: str) -> float:
    """Return a finite non-negative scalar."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative scalar") from exc
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative scalar")
    return result


def _overlap_bounds(first: Any, second: Any, attribute: str) -> tuple[float, float]:
    """Return the shared domain of two spectrum-like objects."""
    first_values = _as_float_array(getattr(first, attribute), name=attribute)
    second_values = _as_float_array(getattr(second, attribute), name=attribute)
    lower = max(float(np.min(first_values)), float(np.min(second_values)))
    upper = min(float(np.max(first_values)), float(np.max(second_values)))
    if upper < lower:
        raise ValueError("the spectra do not have an overlapping integration range")
    return lower, upper


def _radius_from_integral(
    integral: tuple[float, float], prefactor: float, power: float, error: bool
) -> float | list[float]:
    """Convert an integral and its quadrature error into a radius."""
    value, error_estimate = (float(item) for item in integral)
    prefactor = _non_negative(prefactor, name="radius prefactor")
    power = _positive(power, name="radius exponent")
    product = value * prefactor
    if product < 0.0:
        raise ValueError("radius integral must be non-negative")
    radius = product ** (1.0 / power)
    if not error:
        return radius
    uncertainty = radius * abs(error_estimate) / (power * max(abs(value), EPS))
    return [radius, uncertainty]


def delta_E_into_wavelength(
    wavelengths: npt.ArrayLike,
    dE: float,
    alpha: float = DEFAULT_ALPHA,
) -> np.ndarray:
    """Apply an energy shift to wavelengths using the project convention."""
    wavelengths_array = np.asarray(wavelengths, dtype=float)
    shifted_energy = float(dE)
    alpha = _positive(alpha, name="alpha")
    denominator = alpha + shifted_energy * wavelengths_array
    if np.any(denominator == 0.0):
        raise ValueError("energy shift produces a zero wavelength denominator")
    return alpha * wavelengths_array / denominator


def wav_to_e(
    wavelengths: npt.ArrayLike,
    dwavelengths: npt.ArrayLike | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Convert wavelengths to energies and optionally propagate uncertainty."""
    wavelengths_array = np.asarray(wavelengths, dtype=float)
    if np.any(wavelengths_array == 0.0):
        raise ValueError("wavelengths must not contain zero")
    energies = alpha / wavelengths_array
    if dwavelengths is None:
        return energies
    _, delta_wavelengths = _validate_pair(
        wavelengths_array,
        dwavelengths,
        x_name="wavelengths",
        y_name="wavelength uncertainties",
    )
    return energies, np.abs(energies / wavelengths_array) * delta_wavelengths


def e_to_wav(
    energies: npt.ArrayLike,
    denergies: npt.ArrayLike | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Convert energies to wavelengths and optionally propagate uncertainty."""
    energies_array = np.asarray(energies, dtype=float)
    if np.any(energies_array == 0.0):
        raise ValueError("energies must not contain zero")
    wavelengths = alpha / energies_array
    if denergies is None:
        return wavelengths
    _, delta_energies = _validate_pair(
        energies_array,
        denergies,
        x_name="energies",
        y_name="energy uncertainties",
    )
    return wavelengths, np.abs(wavelengths / energies_array) * delta_energies


def load_excel_spectrum(file: str) -> np.ndarray:
    """Load an Excel spectrum into a NumPy array."""
    return pd.read_excel(file).to_numpy()


def emission_counts_to_intensity(x: Any, counts: Any) -> np.ndarray:
    """Convert binned emission counts to an intensity per unit wavelength."""
    wavelengths, counts_array = _validate_pair(
        x, counts, x_name="wavelengths", y_name="counts"
    )
    _, widths = extrapolate_diff(wavelengths)
    if np.any(widths == 0.0):
        raise ValueError("wavelength bins must have non-zero widths")
    return counts_array / widths


emission_counts_to_internsity = emission_counts_to_intensity


def get_norm(x: Any, y: Any) -> float:
    """Return the trapezoidal integral of a spectrum."""
    wavelengths, values = _validate_pair(x, y, x_name="wavelengths", y_name="values")
    return float(Data_Series(wavelengths, values).norm)


def normalise_intensity(x: Any, intensity: Any) -> np.ndarray:
    """Normalize intensity so its integral is one."""
    values = _as_float_array(intensity, name="intensity")
    norm = get_norm(x, values)
    if norm == 0.0:
        raise ValueError("cannot normalize an intensity spectrum with zero integral")
    return values / norm


def normalise_counts(x: Any, counts: Any) -> np.ndarray:
    """Convert counts to intensity and normalize the result."""
    intensity = emission_counts_to_intensity(x, counts)
    return normalise_intensity(x, intensity)


def get_normalised_emission_spectrum(
    data: Any, fix: bool = False
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return wavelengths, normalized counts, and the original count integral."""
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2 or values.shape[0] < 2:
        raise ValueError("data must contain at least two rows and two columns")
    wavelengths, counts = _validate_pair(
        values[:, 0], values[:, 1], x_name="wavelengths", y_name="counts"
    )
    if np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError("wavelengths must be strictly increasing")
    widths = np.diff(wavelengths, prepend=wavelengths[0] - np.diff(wavelengths)[0])
    adjusted_counts = counts / widths if fix else counts.copy()
    total = float(np.sum(widths * adjusted_counts))
    if total == 0.0:
        raise ValueError("cannot normalize a spectrum with zero integral")
    return wavelengths, adjusted_counts / total, total


def kappa_sq_f(m1: int, m2: int) -> float:
    """Return the multipole orientation factor for non-negative integer orders."""
    if not isinstance(m1, (int, np.integer)) or not isinstance(m2, (int, np.integer)):
        raise TypeError("multipole orders must be integers")
    if m1 < 0 or m2 < 0:
        raise ValueError("multipole orders must be non-negative")
    hypergeometric = 0.0
    for index in range(min(m1, m2) + 1):
        numerator = factorial(m1) // factorial(m1 - index)
        numerator *= factorial(m2) // factorial(m2 - index)
        denominator = 1
        for term in range(index):
            denominator *= (m1 + 1 + term) * (m2 + 1 + term)
        hypergeometric += numerator / denominator
    denominator = factorial(m1 + m2) ** 2 * (2.0 * hypergeometric - 1.0)
    numerator = (1 + 2 * m1) * (1 + 2 * m2) * factorial(m1) ** 2 * factorial(m2) ** 2
    if denominator == 0.0:
        raise ValueError("multipole orientation factor is undefined for these orders")
    return float(denominator / numerator)


class _EnergyRadiusCalculator:
    """Shared implementation for energy-domain radius integrals."""

    def _energy_radius(self, error: bool, norm_kr: bool | float) -> float | list[float]:
        """Handle energy radius internally."""
        integral = self.perform_MPMP_integral()
        normalization = self._normalization(norm_kr)
        normalized_integral = (integral[0] / normalization, integral[1] / normalization)
        self.integral = np.asarray(normalized_integral, dtype=float)
        return _radius_from_integral(
            normalized_integral, self.prefactor, self.power, error
        )

    def _normalization(self, norm_kr: bool | float) -> float:
        """Handle normalization internally."""
        if norm_kr is True:
            normalization = self.em.kr
        elif norm_kr is False:
            normalization = 1.0
        else:
            normalization = norm_kr
        return _positive(normalization, name="radiative-rate normalization")

    def perform_MPMP_integral(self) -> tuple[float, float]:
        """Process perform MPMP integral."""
        lower, upper = _overlap_bounds(self.em, self.ab, "energies")
        return quad(
            self.get_integrand, lower, upper, epsabs=0.0, epsrel=1e-6, limit=1000
        )

    def get_integrand(self, energy: float) -> float:
        """Return integrand."""
        return float(self.ab(energy) * self.em(energy))

    def __call__(self, *args: Any, **kwargs: Any) -> float | list[float]:
        """Evaluate the object for the supplied arguments."""
        return self.get_MPMP_radius(*args, **kwargs)


class MPMP_Radius_calculator(_EnergyRadiusCalculator):
    """Calculate a multipole-multipole Förster-like transfer radius."""

    def __init__(
        self,
        em: Multipole_Emission_Spectrum,
        ab: Multipole_Absorption_Spectrum,
        eta_PL: float = 1.0,
        kappa_sq_f: Callable[[int, int], float] = kappa_sq_f,
        b_BL: float = 10.0,
        N_A: float = Units.constants["NA"],
        eps_r: float = 3.0,
        hbar: float = Units.constants["hbar"],
        c: float = Units.constants["c"],
        eps_0: float = Units.constants["eps0"],
        mu_0: float = DEFAULT_MU_0,
        mu_r: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the object."""
        self.eta_PL = _non_negative(eta_PL, name="eta_PL")
        self.kappa_sq_f = kappa_sq_f
        self.b_BL = _positive(b_BL, name="b_BL")
        self.N_A = _positive(N_A, name="N_A")
        self.eps_r = _positive(eps_r, name="eps_r")
        self.hbar = _positive(hbar, name="hbar")
        self.c = _positive(c, name="c")
        self.eps_0 = _positive(eps_0, name="eps_0")
        self.mu_0 = _positive(mu_0, name="mu_0")
        self.mu_r = _positive(mu_r, name="mu_r")
        self.n = np.sqrt(self.eps_r * self.mu_r)
        self.em = em
        self.ab = ab
        if self.em.magn != self.ab.magn:
            raise ValueError("emission and absorption must use the same multipole type")
        self.__dict__.update(kwargs)

    @property
    def power(self) -> float:
        """Process power."""
        return float((self.em.order + self.ab.order + 1) * 2)

    @property
    def magn(self) -> bool:
        """Process magn."""
        return self.em.magn

    @property
    def prefactor(self) -> float:
        """Process prefactor."""
        orientation = self.kappa_sq_f(self.em.order, self.ab.order)
        if self.magn:
            numerator = self.mu_0**2 * self.mu_r**2 * orientation
            denominator = 8 * pi * self.hbar
        else:
            numerator = orientation
            denominator = 8 * pi * self.hbar * self.eps_0**2 * self.eps_r**2
        return _positive(numerator / denominator, name="radius prefactor")

    def get_MPMP_radius(
        self, error: bool = True, norm_kr: bool | float = True
    ) -> float | list[float]:
        """Return MPMP radius."""
        return self._energy_radius(error, norm_kr)


class MPMP_ReabsRadius_calculator(_EnergyRadiusCalculator):
    """Calculate a reabsorption radius from two energy-domain spectra."""

    power = 2.0

    def __init__(
        self,
        em: MP_to_Emission,
        ab: MP_to_Absorption,
        eta_PL: float = 1.0,
        kappa_sq_f: Callable[[int, int], float] = kappa_sq_f,
        b_BL: float = 10.0,
        N_A: float = Units.constants["NA"],
        eps_r: float = 3.0,
        hbar: float = Units.constants["hbar"],
        c: float = Units.constants["c"],
        eps_0: float = Units.constants["eps0"],
        mu_0: float = DEFAULT_MU_0,
        mu_r: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the object."""
        self.eta_PL = _non_negative(eta_PL, name="eta_PL")
        self.kappa_sq_f = kappa_sq_f
        self.b_BL = _positive(b_BL, name="b_BL")
        self.N_A = _positive(N_A, name="N_A")
        self.eps_r = _positive(eps_r, name="eps_r")
        self.hbar = _positive(hbar, name="hbar")
        self.c = _positive(c, name="c")
        self.eps_0 = _positive(eps_0, name="eps_0")
        self.mu_0 = _positive(mu_0, name="mu_0")
        self.mu_r = _positive(mu_r, name="mu_r")
        self.n = np.sqrt(self.eps_r * self.mu_r)
        self.em = em
        self.ab = ab
        if self.em.magn != self.ab.magn:
            raise ValueError("emission and absorption must use the same multipole type")
        self.__dict__.update(kwargs)

    @property
    def magn(self) -> bool:
        """Process magn."""
        return self.em.magn

    @property
    def prefactor(self) -> float:
        """Process prefactor."""
        return _positive(
            np.log(self.b_BL) / (4 * pi * self.N_A), name="radius prefactor"
        )

    def get_MPMP_radius(
        self, error: bool = True, norm_kr: bool | float = True
    ) -> float | list[float]:
        """Return MPMP radius."""
        return self._energy_radius(error, norm_kr)


class Foerster_Radius_calculator:
    """Calculate a wavelength-domain Förster radius."""

    def __init__(
        self,
        I_D: Any,
        eps_A: Any,
        eta_PL: float = 1.0,
        kappa_sq: float = 2.0 / 3.0,
        b_BL: float = 10.0,
        N_A: float = Units.constants["NA"],
        n: float = np.sqrt(3),
    ) -> None:
        """Initialize the object."""
        self.I_D = deepcopy(I_D)
        self.eps_A = deepcopy(eps_A)
        self.eta_PL = _non_negative(eta_PL, name="eta_PL")
        self.kappa_sq = _positive(kappa_sq, name="kappa_sq")
        self.b_BL = _positive(b_BL, name="b_BL")
        self.N_A = _positive(N_A, name="N_A")
        self.n = _positive(n, name="n")

    def get_Forster_radius(self, error: bool = True) -> float | list[float]:
        """Return Forster radius."""
        prefactor = (
            9
            * np.log(self.b_BL)
            * self.eta_PL
            * self.kappa_sq
            / (2**7 * pi**5 * self.n**4 * self.N_A)
        )
        return _radius_from_integral(
            self.perform_forster_integral(), prefactor, 6.0, error
        )

    def perform_forster_integral(self) -> tuple[float, float]:
        """Process perform forster integral."""
        lower, upper = _overlap_bounds(self.eps_A, self.I_D, "wavelengths")
        return quad(
            self.get_integrand, lower, upper, epsabs=0.0, epsrel=1e-5, limit=1000
        )

    def get_integrand(self, wavelength: float) -> float:
        """Return integrand."""
        return float(
            self.eps_A.get_eps_A(wavelength)
            * self.I_D.get_normalised_intensity(wavelength)
            * wavelength**4
        )


class Foerster_Radius_calculator_energy:
    """Calculate an energy-domain Förster radius."""

    def __init__(
        self,
        I_D: Any = None,
        eps_A: Any = None,
        eta_PL: float = 1.0,
        kappa_sq: float = 2.0 / 3.0,
        b_BL: float = 10.0,
        N_A: float = Units.constants["NA"],
        n: float = np.sqrt(3),
        hbar: float = Units.constants["hbar"],
        c: float = Units.constants["c"],
        eps_0: float = Units.constants["eps0"],
        power: float = 6.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the object."""
        self.I_D = I_D
        self.eps_A = eps_A
        self.eta_PL = _non_negative(eta_PL, name="eta_PL")
        self.kappa_sq = _positive(kappa_sq, name="kappa_sq")
        self.b_BL = _positive(b_BL, name="b_BL")
        self.N_A = _positive(N_A, name="N_A")
        self.n = _positive(n, name="n")
        self.hbar = _positive(hbar, name="hbar")
        self.c = _positive(c, name="c")
        self.eps_0 = _positive(eps_0, name="eps_0")
        self.power = _positive(power, name="power")
        self.__dict__.update(kwargs)

    def _require_spectra(self) -> None:
        """Handle require spectra internally."""
        if self.I_D is None or self.eps_A is None:
            raise ValueError("I_D and eps_A spectra are required")

    def get_Forster_radius(self, error: bool = True) -> float | list[float]:
        """Return Forster radius."""
        self._require_spectra()
        prefactor = (
            9
            * np.log(self.b_BL)
            * self.eta_PL
            * self.kappa_sq
            * self.hbar**4
            * self.c**4
            / (2**3 * pi * self.n**4 * self.N_A)
        )
        return _radius_from_integral(
            self.perform_forster_integral(), prefactor, self.power, error
        )

    def perform_forster_integral(self) -> tuple[float, float]:
        """Process perform forster integral."""
        self._require_spectra()
        lower, upper = _overlap_bounds(self.eps_A, self.I_D, "energies")
        if lower <= 0.0:
            raise ValueError(
                "energy-domain Förster integration requires positive energies"
            )
        return quad(
            self.get_integrand, lower, upper, epsabs=0.0, epsrel=1e-5, limit=1000
        )

    def get_integrand(self, energy: float) -> float:
        """Return integrand."""
        return float(
            self.eps_A.get_eps_A(energy)
            * self.I_D.get_normalised_intensity(energy)
            * energy ** (-4)
        )


class QD_Radius_calculator:
    """Calculate a quadrupole-dipole radius for supplied spectra."""

    def __init__(
        self,
        I_D: Any = None,
        widetilde_eps_QD: Any = None,
        eta_PL: float = 1.0,
        kappa_sq: float = 1.0,
        b_BL: float = 10.0,
        N_A: float = Units.constants["NA"],
        n: float = np.sqrt(3),
        hbar: float = Units.constants["hbar"],
        c: float = Units.constants["c"],
        eps_0: float = Units.constants["eps0"],
        power: float = 8.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the object."""
        self.I_D = I_D
        self.widetilde_eps_QD = widetilde_eps_QD
        self.eta_PL = _non_negative(eta_PL, name="eta_PL")
        self.kappa_sq = _positive(kappa_sq, name="kappa_sq")
        self.b_BL = _positive(b_BL, name="b_BL")
        self.N_A = _positive(N_A, name="N_A")
        self.n = _positive(n, name="n")
        self.hbar = _positive(hbar, name="hbar")
        self.c = _positive(c, name="c")
        self.eps_0 = _positive(eps_0, name="eps_0")
        self.eps_r = self.n**2
        self.power = _positive(power, name="power")
        self.__dict__.update(kwargs)

    def _prefactor(self) -> float:
        """Handle prefactor internally."""
        if self.widetilde_eps_QD is None or self.I_D is None:
            raise ValueError("I_D and widetilde_eps_QD spectra are required")
        return _positive(
            3
            * self.hbar**3
            * self.c**3
            * self.kappa_sq
            / (8 * self.eps_0 * self.eps_r * self.n**2),
            name="radius prefactor",
        )

    def get_QD_radius(self, error: bool = True) -> float | list[float]:
        """Return QD radius."""
        self.e_pow = -3.0
        prefactor = self._prefactor()
        return _radius_from_integral(
            self.perform_QD_integral(), prefactor, self.power, error
        )

    def perform_QD_integral(self) -> tuple[float, float]:
        """Process perform QD integral."""
        lower, upper = _overlap_bounds(self.widetilde_eps_QD, self.I_D, "energies")
        if lower <= 0.0:
            raise ValueError("energy-domain QD integration requires positive energies")
        return quad(
            self.get_integrand, lower, upper, epsabs=0.0, epsrel=1e-5, limit=1000
        )

    def get_integrand(self, energy: float) -> float:
        """Return integrand."""
        return float(
            self.widetilde_eps_QD.get_tilde_eps_QD(energy)
            * self.I_D.get_normalised_intensity(energy)
            * energy**self.e_pow
        )


class OD_Radius_calculator:
    """Calculate an octupole-dipole radius for supplied spectra."""

    def __init__(
        self,
        I_D: Any = None,
        widetilde_eps_O: Any = None,
        eta_PL: float = 1.0,
        kappa_sq: float = 4.0 / 3.0,
        b_BL: float = 10.0,
        N_A: float = Units.constants["NA"],
        n: float = np.sqrt(3),
        hbar: float = Units.constants["hbar"],
        c: float = Units.constants["c"],
        eps_0: float = Units.constants["eps0"],
        power: float = 10.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the object."""
        self.I_D = I_D
        self.widetilde_eps_O = widetilde_eps_O
        self.eta_PL = _non_negative(eta_PL, name="eta_PL")
        self.kappa_sq = _positive(kappa_sq, name="kappa_sq")
        self.b_BL = _positive(b_BL, name="b_BL")
        self.N_A = _positive(N_A, name="N_A")
        self.n = _positive(n, name="n")
        self.hbar = _positive(hbar, name="hbar")
        self.c = _positive(c, name="c")
        self.eps_0 = _positive(eps_0, name="eps_0")
        self.eps_r = self.n**2
        self.power = _positive(power, name="power")
        self.__dict__.update(kwargs)

    def _prefactor(self) -> float:
        """Handle prefactor internally."""
        if self.widetilde_eps_O is None or self.I_D is None:
            raise ValueError("I_D and widetilde_eps_O spectra are required")
        return _positive(
            3
            * self.hbar**3
            * self.c**3
            * self.kappa_sq
            / (8 * self.eps_0 * self.eps_r * self.n**2),
            name="radius prefactor",
        )

    def get_OD_radius(self, error: bool = True) -> float | list[float]:
        """Return OD radius."""
        self.e_pow = -3.0
        prefactor = self._prefactor()
        return _radius_from_integral(
            self.perform_OD_integral(), prefactor, self.power, error
        )

    def perform_OD_integral(self) -> tuple[float, float]:
        """Process perform OD integral."""
        lower, upper = _overlap_bounds(self.widetilde_eps_O, self.I_D, "energies")
        if lower <= 0.0:
            raise ValueError("energy-domain OD integration requires positive energies")
        return quad(
            self.get_integrand, lower, upper, epsabs=0.0, epsrel=1e-5, limit=1000
        )

    def get_integrand(self, energy: float) -> float:
        """Return integrand."""
        return float(
            self.widetilde_eps_O.get_tilde_eps_O(energy)
            * self.I_D.get_normalised_intensity(energy)
            * energy**self.e_pow
        )


Forster_Radius_calculator = Foerster_Radius_calculator
Forster_Radius_calculator_energy = Foerster_Radius_calculator_energy
