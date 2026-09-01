"""Spectrum conversion and multipole spectrum helpers."""

import math

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import interp1d

from .base_utils import extrapolate_diff
from .spectra_averager import Data_Series
from .units import Units

EPS = 1e-13

MULTIPOLE_NUM_LET_DICT = {1: "D", 2: "Q", 3: "O"}


def delta_E_into_wavelength(
    wavelengths: npt.NDArray[np.float64],
    dE: float,
    alpha=2 * np.pi * Units.constants["hbar"] * Units.constants["c"],
) -> npt.NDArray[np.float64]:
    """Process delta E into wavelength."""
    numerator = alpha * wavelengths
    denominator = alpha + dE * wavelengths
    out = numerator / denominator
    return out


def wav_to_e(
    wavelengths: npt.NDArray[np.float64],
    dwavelengths=None,
    alpha=2 * np.pi * Units.constants["hbar"] * Units.constants["c"],
) -> npt.NDArray[np.float64]:
    """Process wav to e."""
    energies = alpha / wavelengths
    if dwavelengths is None:
        return energies
    else:
        denergies = np.abs(energies / wavelengths) * dwavelengths
        return energies, denergies


def e_to_wav(
    energies: npt.NDArray[np.float64],
    denergies=None,
    alpha=2 * np.pi * Units.constants["hbar"] * Units.constants["c"],
) -> npt.NDArray[np.float64]:
    """Process e to wav."""
    wavelengths = alpha / energies
    if denergies is None:
        return wavelengths
    else:
        dwavelengths = np.abs(wavelengths / energies) * denergies
        return wavelengths, dwavelengths


def load_excel_spectrum(file):
    """Process load excel spectrum."""
    data = pd.read_excel(file)
    data = data.to_numpy()
    return data


def emission_counts_to_internsity(x, counts):
    """Process emission counts to internsity."""
    x, dx = extrapolate_diff(x)
    countsdx = counts / dx
    return countsdx


def get_norm(x, y):
    """Return norm."""
    serie = Data_Series(x, y)
    return serie.norm


def normalise_intensity(x, intensity):
    """Process normalise intensity."""
    return intensity / get_norm(x, intensity)


def normalise_counts(x, counts):
    """Process normalise counts."""
    intensity = emission_counts_to_internsity(x, counts)
    return normalise_intensity(x, intensity)


def get_normalised_emission_spectrum(data, fix=False):
    """Return normalised emission spectrum."""
    wavelengths = data[:, 0]
    d_wavelengths = np.diff(
        wavelengths, prepend=wavelengths[0] - np.diff(wavelengths)[1]
    )

    counts = data[:, 1]

    if fix:
        counts /= d_wavelengths

    tot_counts = np.sum(d_wavelengths * counts)
    norm_counts = counts / tot_counts
    return wavelengths, norm_counts, tot_counts


def angular_averaging_prefactor_mp(order):
    """Process angular averaging prefactor mp."""
    assert isinstance(order, int)
    assert order > 0
    denom = (
        (2 ** (2 * order - 1)) * math.factorial(order - 1) * math.factorial(order + 1)
    )
    num = math.factorial(2 * order) * math.factorial(2 * order + 1)
    return denom / num


class Multipole_Absorption_Spectrum(Data_Series):
    def __init__(self, order, *args, magn=False, **kwargs):
        """Initialize the object."""
        self.order = order
        self.magn = magn
        super().__init__(*args, **kwargs)

    def __getstate__(self):
        """
        Return only the attributes needed to rebuild the series.
        NumPy arrays, floats and dicts of simple params are pickleable.
        """
        state = {
            "order": self.order,
            "_x": self._x,
            "_y": self._y,
            "_data_processed": self._data_processed,
            "function_type": self.function_type,
            "kind": self.kind,
            "fkwargs": self.fkwargs,
            "weight": self.weight,
            "left": self.left,
            "right": self.right,
        }
        return state

    def __setstate__(self, state):
        """
        Restore the minimal state, re‑initializing any internal caches
        on first use (via your existing property‑based logic).
        """
        self.order = state["order"]
        self._x = state["_x"]
        self._y = state["_y"]
        self._data_processed = state["_data_processed"]
        self.function_type = state["function_type"]
        self.kind = state["kind"]
        self.fkwargs = state["fkwargs"]
        self.weight = state["weight"]
        self.left = state["left"]
        self.right = state["right"]

    def __deepcopy__(self, memo):
        """Handle deepcopy internally."""
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new

        state = self.__getstate__()

        import copy

        for k, v in state.items():
            setattr(new, k, copy.deepcopy(v, memo))

        return new

    @property
    def energies(self):
        """Process energies."""
        return self.x

    @property
    def intensity(self):
        """Process intensity."""
        return self.y


class Multipole_Emission_Spectrum(Data_Series):
    def __getattr__(self, name):
        """Resolve a dynamically requested attribute."""
        if name[:7] == "interp_":
            return Data_Series(self.energies, getattr(self, name[7:]))

    def __init__(
        self,
        order,
        energies,
        intensity,
        kr=None,
        magn=False,
        *args,
        **kwargs,
    ):
        """Initialize the object."""
        self.order = order
        self.magn = magn

        super().__init__(energies, intensity, *args, **kwargs)

        if kr is not None:
            self.kr = kr
        else:
            self.kr = self.norm

    def __getstate__(self):
        """
        Return only the attributes needed to rebuild the series.
        NumPy arrays, floats and dicts of simple params are pickleable.
        """
        state = {
            "order": self.order,
            "kr": self.kr,
            "_x": self._x,
            "_y": self._y,
            "_data_processed": self._data_processed,
            "function_type": self.function_type,
            "kind": self.kind,
            "fkwargs": self.fkwargs,
            "weight": self.weight,
            "left": self.left,
            "right": self.right,
        }
        return state

    def __setstate__(self, state):
        """
        Restore the minimal state, re‑initializing any internal caches
        on first use (via your existing property‑based logic).
        """
        self.order = state["order"]
        self.kr = state["kr"]
        self._x = state["_x"]
        self._y = state["_y"]
        self._data_processed = state["_data_processed"]
        self.function_type = state["function_type"]
        self.kind = state["kind"]
        self.fkwargs = state["fkwargs"]
        self.weight = state["weight"]
        self.left = state["left"]
        self.right = state["right"]

    def __deepcopy__(self, memo):
        """Handle deepcopy internally."""
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new

        state = self.__getstate__()

        import copy

        for k, v in state.items():
            setattr(new, k, copy.deepcopy(v, memo))

        return new

    @property
    def energies(self):
        """Process energies."""
        return self.x

    @property
    def intensity(self):
        """Process intensity."""
        return self.y

    @property
    def norm_intensity(self):
        """Process norm intensity."""
        return self.intensity / self.norm

    @property
    def kr_norm_intensity(self):
        """Process kr norm intensity."""
        return self.norm_intensity * self.kr


class MP_to_Emission(Data_Series):
    def __init__(
        self,
        energies,
        mu_sq,
        kind="linear",
        mp_order=1,
        epsr=3.0,
        magn=False,
        mur=1.0,
        b_BL=10.0,
        **kwargs,
    ):
        """Initialize the object."""
        self.__dict__.update(kwargs)
        self.mp_order = mp_order
        self.order = mp_order
        self.epsr = epsr
        self.kind = kind
        self.magn = magn
        self.mur = mur
        self.b_BL = b_BL

        super().__init__(x=energies, y=mu_sq, **kwargs)

    @property
    def ord2(self):
        """Process ord2."""
        return (self.mp_order - 1) * 2

    @property
    def refractive_index(self):
        """Process refractive index."""
        return np.sqrt(self.epsr * self.mur)

    @property
    def prefactor(self):
        """Process prefactor."""
        if self.magn:
            return (
                (
                    Units.constants["mu0"]
                    * self.mur
                    * (((self.refractive_index) ** ((3 + self.ord2))))
                )
                * angular_averaging_prefactor_mp(self.mp_order)
                / (
                    np.pi
                    * (Units.constants["c"] ** (3 + self.ord2))
                    * (Units.constants["hbar"] ** (4 + self.ord2))
                )
            )
        else:
            return (
                (((self.refractive_index) ** ((3 + self.ord2))))
                * angular_averaging_prefactor_mp(self.mp_order)
                / (
                    np.pi
                    * (Units.constants["c"] ** (3 + self.ord2))
                    * Units.constants["eps0"]
                    * self.epsr
                    * (Units.constants["hbar"] ** (4 + self.ord2))
                )
            )

    @property
    def energies(self):
        """Process energies."""
        return self.x

    @property
    def mu_sq(self):
        """Process mu sq."""
        return super().y

    @property
    def y(self):
        """Process y."""
        if not self._data_processed:
            self._process_data()
        return self.mu_sq * (self.energies ** (3 + self.ord2)) * self.prefactor


class Emission_to_MP(Data_Series):
    def __init__(
        self,
        energies,
        intensity,
        kr=1.0,
        kind="linear",
        mp_order=1,
        epsr=3.0,
        magn=False,
        mur=1.0,
        b_BL=10.0,
        **kwargs,
    ):
        """Initialize the object."""
        self.mp_order = mp_order
        self.order = mp_order
        self.epsr = epsr
        self.__dict__.update(kwargs)
        self.kind = kind
        self.magn = magn
        self.mur = mur
        self.b_BL = b_BL

        self.kr = kr

        super().__init__(x=energies, y=intensity, **kwargs)

    @property
    def ord2(self):
        """Process ord2."""
        return (self.mp_order - 1) * 2

    @property
    def refractive_index(self):
        """Process refractive index."""
        return np.sqrt(self.epsr * self.mur)

    @property
    def prefactor(self):
        """Process prefactor."""
        if self.magn:
            return (
                (
                    Units.constants["mu0"]
                    * self.mur
                    * (((self.refractive_index) ** ((3 + self.ord2))))
                )
                * angular_averaging_prefactor_mp(self.mp_order)
                / (
                    np.pi
                    * (Units.constants["c"] ** (3 + self.ord2))
                    * (Units.constants["hbar"] ** (4 + self.ord2))
                )
            )
        else:
            return (
                (((self.refractive_index) ** ((3 + self.ord2))))
                * angular_averaging_prefactor_mp(self.mp_order)
                / (
                    np.pi
                    * (Units.constants["c"] ** (3 + self.ord2))
                    * Units.constants["eps0"]
                    * self.epsr
                    * (Units.constants["hbar"] ** (4 + self.ord2))
                )
            )

    @property
    def energies(self):
        """Process energies."""
        return self.x

    @property
    def intensity(self):
        """Process intensity."""
        return super().y

    @property
    def y(self):
        """Process y."""
        return (
            self.intensity
            * self.kr
            / ((self.energies ** (3 + self.ord2)) * self.prefactor)
        )


class ExperimentF_Dipole_Spectrum(MP_to_Emission):
    pass


class MP_to_Absorption(MP_to_Emission):
    @property
    def prefactor(self):
        """Process prefactor."""
        if self.magn:
            return (
                (
                    np.pi
                    * Units.constants["mu0"]
                    * self.mur
                    * Units.constants["NA"]
                    * (((self.refractive_index) ** ((1 + self.ord2))))
                )
                * angular_averaging_prefactor_mp(self.mp_order)
                / (
                    (Units.constants["c"] ** (1 + self.ord2))
                    * np.log(self.b_BL)
                    * (Units.constants["hbar"] ** (1 + self.ord2))
                )
            )
        else:
            return (
                (
                    np.pi
                    * Units.constants["NA"]
                    * (((self.refractive_index) ** ((1 + self.ord2))))
                )
                * angular_averaging_prefactor_mp(self.mp_order)
                / (
                    (Units.constants["c"] ** (1 + self.ord2))
                    * np.log(self.b_BL)
                    * Units.constants["eps0"]
                    * self.epsr
                    * (Units.constants["hbar"] ** (1 + self.ord2))
                )
            )

    @property
    def y(self):
        """Process y."""
        if not self._data_processed:
            self._process_data()
        return self.mu_sq * (self.energies ** (1 + self.ord2)) * self.prefactor

    @property
    def cross_section(self):
        """Process cross section."""
        return self.y * np.log(self.b_BL) / Units.constants["NA"]


class Absorption_to_MP(Emission_to_MP):
    @property
    def prefactor(self):
        """Process prefactor."""
        if self.magn:
            return (
                (
                    np.pi
                    * Units.constants["mu0"]
                    * self.mur
                    * Units.constants["NA"]
                    * (((self.refractive_index) ** ((1 + self.ord2))))
                )
                * angular_averaging_prefactor_mp(self.mp_order)
                / (
                    (Units.constants["c"] ** (1 + self.ord2))
                    * np.log(self.b_BL)
                    * (Units.constants["hbar"] ** (1 + self.ord2))
                )
            )
        else:
            return (
                (
                    np.pi
                    * Units.constants["NA"]
                    * (((self.refractive_index) ** ((1 + self.ord2))))
                )
                * angular_averaging_prefactor_mp(self.mp_order)
                / (
                    (Units.constants["c"] ** (1 + self.ord2))
                    * np.log(self.b_BL)
                    * Units.constants["eps0"]
                    * self.epsr
                    * (Units.constants["hbar"] ** (1 + self.ord2))
                )
            )

    @property
    def y(self):
        """Process y."""
        if not self._data_processed:
            self._process_data()
        return self.intensity / ((self.energies ** (1 + self.ord2)) * self.prefactor)


class ExperimentalA_Dipole_Spectrum(Absorption_to_MP):
    pass


class Emission_Spectrum:
    def __init__(self, kind="linear", **kwargs):
        """Initialize the object."""
        self.__dict__.update(kwargs)

        assert "norm_counts" in self.__dict__
        assert ("wavelengths" in self.__dict__) or ("energies" in self.__dict__)
        assert not (("wavelengths" in self.__dict__) and ("energies" in self.__dict__))

        if "wavelengths" in self.__dict__:
            self.f_I_D = interp1d(
                self.wavelengths,
                self.norm_counts,
                kind=kind,
                bounds_error=False,
                fill_value=(0.0, 0.0),
            )
        elif "energies" in self.__dict__:
            self.f_I_D = interp1d(
                self.energies,
                self.norm_counts,
                kind=kind,
                bounds_error=False,
                fill_value=(0.0, 0.0),
            )

    def get_normalised_intensity(self, x, scipy=True):
        """Return normalised intensity."""
        if scipy:
            return self.f_I_D(x)
        else:
            if "wavelengths" in self.__dict__:
                return np.interp(
                    x, self.wavelengths, self.norm_counts, left=0.0, right=0.0
                )
            elif "energies" in self.__dict__:
                return np.interp(
                    x, self.energies, self.norm_counts, left=0.0, right=0.0
                )
            else:
                raise Exception("Emission spectrum setup incorrectly")


class Absorption_Spectrum:
    def __init__(self, kind="linear", **kwargs):
        """Initialize the object."""
        self.__dict__.update(kwargs)
        assert "eps_A" in self.__dict__
        assert ("wavelengths" in self.__dict__) or ("energies" in self.__dict__)
        assert not (("wavelengths" in self.__dict__) and ("energies" in self.__dict__))
        if "wavelengths" in self.__dict__:
            self.f_eps_A = interp1d(
                self.wavelengths,
                self.eps_A,
                kind=kind,
                bounds_error=False,
                fill_value=(0.0, 0.0),
            )
        elif "energies" in self.__dict__:
            self.f_eps_A = interp1d(
                self.energies,
                self.eps_A,
                kind=kind,
                bounds_error=False,
                fill_value=(0.0, 0.0),
            )

    def get_eps_A(self, x, scipy=True):
        """Return eps A."""
        if scipy:
            return self.f_eps_A(x)
        else:
            if "wavelengths" in self.__dict__:
                return np.interp(x, self.wavelengths, self.eps_A, left=0.0, right=0.0)
            elif "energies" in self.__dict__:
                return np.interp(x, self.energies, self.eps_A, left=0.0, right=0.0)


class Theoretical_Quadrupolar_Spectrum:
    def __init__(self, energies, eps_QD, kind="linear"):
        """Initialize the object."""
        self.energies = energies
        self.tilde_eps_QD = eps_QD
        self.f_tilde_eps_QD = interp1d(
            self.energies,
            self.tilde_eps_QD,
            kind=kind,
            bounds_error=False,
            fill_value=(0.0, 0.0),
        )

    def get_tilde_eps_QD(self, x, scipy=True):
        """Return tilde eps QD."""
        if scipy:
            return self.f_tilde_eps_QD(x)
        else:
            return np.interp(x, self.energies, self.tilde_eps_QD, left=0.0, right=0.0)


class Theoretical_Octupolar_Spectrum:
    def __init__(self, energies, eps_O, kind="linear"):
        """Initialize the object."""
        self.energies = energies
        self.tilde_eps_O = eps_O
        self.f_tilde_eps_O = interp1d(
            self.energies,
            self.tilde_eps_O,
            kind=kind,
            bounds_error=False,
            fill_value=(0.0, 0.0),
        )

    def get_tilde_eps_O(self, x, scipy=True):
        """Return tilde eps O."""
        if scipy:
            return self.f_tilde_eps_O(x)
        else:
            return np.interp(x, self.energies, self.tilde_eps_O, left=0.0, right=0.0)


def main():
    """Process main."""
    pass


if __name__ == "__main__":
    main()
