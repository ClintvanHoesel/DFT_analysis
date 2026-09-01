import os
from os.path import join

import numpy as np

from ..frank_condon import Franck_Condon
from ..multipole_calculators import (
    calc_oct_moment,
    dip_moment_calculator,
    quad_moment_calculator,
)
from ..units import Units
from .rkf_reader import Single_RKF_Reader


def zero_diagonals(matrix):
    """
    Sets all elements along the diagonals of the first two axes to zero
    in an N-dimensional matrix with shape (n, n, d1, d2, ..., dk).

    Parameters:
    matrix (numpy.ndarray): Input N-dimensional matrix where first two dimensions are equal

    Returns:
    numpy.ndarray: Matrix with diagonals set to zero
    """

    result = matrix.copy()

    n = matrix.shape[0]

    idx = np.arange(n)

    result[idx, idx, ...] = 0

    return result


class RKF_organizer(object):
    def __init__(self, base_folder=os.getcwd(), file="adf.rkf"):
        """Initialize the object."""
        self.d = dict()
        self.base_folder = base_folder
        self.file = file

    def safe_get(self, key, item):
        """Process safe get."""
        try:
            return getattr(self[key], item)
        except:
            return np.nan

    def __getitem__(self, key):
        """Handle getitem internally."""
        if key in self.d:
            return self.d[key]
        else:
            self.d[key] = Single_RKF_Reader(join(self.base_folder, key, self.file))
            return self.d[key]


class Spectra_RKFs:
    def __init__(self, exc_rkf=None, excgrad_rkf=None, hess_rkf=None):
        """Initialize the object."""
        self.exc_rkf = exc_rkf
        self.excgrad_rkf = excgrad_rkf
        self.hess_rkf = hess_rkf

    def __getattr__(self, name):
        """Try to get the attribute from self first, then from any of the RKF attributes."""
        for rkf in (self.exc_rkf, self.excgrad_rkf, self.hess_rkf):
            if rkf is not None and hasattr(rkf, name):
                return getattr(rkf, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    @property
    def exc_en_so(self):
        """Process exc en so."""
        return np.real(Units.convert(np.array(self.exc_rkf.eigvals_SO), "au", "J"))

    @property
    def exc_en_S(self):
        """Process exc en S."""
        return Units.convert(np.array(self.exc_rkf.exc_en_S), "au", "J")

    @property
    def exc_en_T(self):
        """Process exc en T."""
        return Units.convert(np.array(self.exc_rkf.exc_en_T), "au", "J")

    @property
    def tdms(self):
        """Process tdms."""
        return np.array(
            Units.nconvert(
                Units.convert(self.exc_rkf.transition_dipole_moments, "au", "C_m"),
                "au",
                "m",
                0,
            )
        )

    @property
    def tqms(self):
        """Process tqms."""
        return np.array(
            Units.nconvert(
                Units.convert(self.exc_rkf.transition_quadrupole_moments, "au", "C_m"),
                "au",
                "m",
                1,
            )
        )

    @property
    def toms(self):
        """Process toms."""
        return np.array(
            Units.nconvert(
                Units.convert(self.exc_rkf.transition_octupole_moments, "au", "C_m"),
                "au",
                "m",
                2,
            )
        )

    @property
    def tdms_so(self):
        """Process tdms so."""
        return self.exc_rkf.transpose_singlet(self.tdms)

    @property
    def tqms_so(self):
        """Process tqms so."""
        return self.exc_rkf.transpose_singlet(self.tqms)

    @property
    def toms_so(self):
        """Process toms so."""
        return self.exc_rkf.transpose_singlet(self.toms)

    @property
    def tdms_sq(self):
        """Process tdms sq."""
        return dip_moment_calculator(self.tdms)

    @property
    def tqms_sq(self):
        """Process tqms sq."""
        return quad_moment_calculator(self.tqms)

    @property
    def toms_sq(self):
        """Process toms sq."""
        return calc_oct_moment(self.toms)

    @property
    def tdms_so_sq(self):
        """Process tdms so sq."""
        return dip_moment_calculator(self.tdms_so)

    @property
    def tqms_so_sq(self):
        """Process tqms so sq."""
        return quad_moment_calculator(self.tqms_so)

    @property
    def toms_so_sq(self):
        """Process toms so sq."""
        return calc_oct_moment(self.toms_so)

    @property
    def masses(self):
        """Process masses."""
        return Units.convert(self.hess_rkf.atom_masses, "au", "kg")

    @property
    def hessian(self):
        """Process hessian."""
        return Units.convert(self.hess_rkf.hessian, "au", "J/m^2")

    @property
    def inverted_hessian(self):
        """Process inverted hessian."""
        fc = Franck_Condon(masses=self.masses, hessian=self.hessian)
        return fc.inverted_hessian_reduced

    @property
    def exc_grads_S(self):
        """Process exc grads S."""
        return Units.convert(self.excgrad_rkf.singlet_excited_gradients, "au", "J/m")

    @property
    def exc_grads_T(self):
        """Process exc grads T."""
        return Units.convert(self.excgrad_rkf.triplet_excited_gradients, "au", "J/m")

    @property
    def exc_grads_so(self):
        """Process exc grads so."""
        exc_grads = self.exc_rkf.transpose_singlet(
            self.exc_grads_S, sq=True
        ) + self.exc_rkf.transpose_triplet(self.exc_grads_T, sq=True)

        return np.real(exc_grads)

    @property
    def dexc_grads_S(self):
        """Process dexc grads S."""
        return self.exc_grads_S[None, :, :, :] - self.exc_grads_S[:, None, :, :]

    @property
    def dexc_grads_T(self):
        """Process dexc grads T."""
        return self.exc_grads_T[None, :, :, :] - self.exc_grads_T[:, None, :, :]

    @property
    def dexc_grads_so(self):
        """Process dexc grads so."""
        return self.exc_grads_so[None, :, :, :] - self.exc_grads_so[:, None, :, :]

    @property
    def delta_reorg_energies_S(self):
        """Process delta reorg energies S."""
        exc_grads = self.exc_grads_S.reshape(
            (
                self.exc_grads_S.shape[0],
                self.exc_grads_S.shape[1] * self.exc_grads_S.shape[2],
            )
        )
        gHinvT = np.einsum("ik,kj->ij", exc_grads, self.inverted_hessian)
        term1 = np.einsum("ij,ij->i", gHinvT, 0.5 * exc_grads)
        term2 = np.einsum("ij,kj->ik", gHinvT, exc_grads)
        return term1[:, None] - term2

    @property
    def delta_reorg_energies_T(self):
        """Process delta reorg energies T."""
        exc_grads = self.exc_grads_T.reshape(
            (
                self.exc_grads_T.shape[0],
                self.exc_grads_T.shape[1] * self.exc_grads_T.shape[2],
            )
        )
        gHinvT = np.einsum("ik,kj->ij", exc_grads, self.inverted_hessian)
        term1 = np.einsum("ij,ij->i", gHinvT, 0.5 * exc_grads)
        term2 = np.einsum("ij,kj->ik", gHinvT, exc_grads)
        return term1[:, None] - term2

    @property
    def delta_reorg_energies_so(self):
        """Process delta reorg energies so."""
        exc_grads = self.exc_grads_so.reshape(
            (
                self.exc_grads_so.shape[0],
                self.exc_grads_so.shape[1] * self.exc_grads_so.shape[2],
            )
        )
        gHinvT = np.einsum("ik,kj->ij", exc_grads, self.inverted_hessian)
        term1 = np.einsum("ij,ij->i", gHinvT, 0.5 * exc_grads)
        term2 = np.einsum("ij,kj->ik", gHinvT, exc_grads)
        return term1[:, None] - term2

    @property
    def GSESTDM(self):
        """Process GSESTDM."""
        return Units.nconvert(
            Units.convert(self.exc_rkf.GSESTDM, "au", "C_m"),
            "au",
            "m",
            0,
        )

    @property
    def ESESTDM(self):
        """Process ESESTDM."""
        out = Units.nconvert(
            Units.convert(self.exc_rkf.ESESTDM, "au", "C_m"),
            "au",
            "m",
            0,
        )
        out = zero_diagonals(out)
        return out

    @property
    def ETETTDM(self):
        """Process ETETTDM."""
        out = Units.nconvert(
            Units.convert(self.exc_rkf.ETETTDM, "au", "C_m"),
            "au",
            "m",
            0,
        )
        out = zero_diagonals(out)
        return out

    @property
    def ETETTDMrep(self):
        """Process ETETTDMrep."""
        out = Units.nconvert(
            Units.convert(self.exc_rkf.ETETTDMrep, "au", "C_m"),
            "au",
            "m",
            0,
        )
        out = zero_diagonals(out)
        return out

    @property
    def GSESTDM_so(self):
        """Process GSESTDM so."""
        return self.exc_rkf.transpose_singlet(self.GSESTDM)

    @property
    def ESESTDM_so(self):
        """Process ESESTDM so."""
        out = self.exc_rkf.transpose_singlet(self.ESESTDM)
        out = np.swapaxes(out, 0, 1)
        out = self.exc_rkf.transpose_singlet(out)
        return out

    @property
    def ETETTDM_so(self):
        """Process ETETTDM so."""
        out = self.exc_rkf.transpose_triplet(self.ETETTDM)
        out = np.swapaxes(out, 0, 1)
        out = self.exc_rkf.transpose_triplet(out)
        return out

    @property
    def EETDM_so(self):
        """Process EETDM so."""
        out = self.ESESTDM_so + self.ETETTDM_so
        out[0, :, :] += self.GSESTDM_so
        out[:, 0, :] += self.GSESTDM_so
        out = zero_diagonals(out)
        return out

    @property
    def EETDM_so_sq(self):
        """Process EETDM so sq."""
        out = np.abs(dip_moment_calculator(self.EETDM_so))
        out = zero_diagonals(out)
        return out

    @property
    def ESESTDM_sq(self):
        """Process ESESTDM sq."""
        out = np.abs(dip_moment_calculator(self.ESESTDM))
        out = zero_diagonals(out)
        return out

    @property
    def ETETTDM_sq(self):
        """Process ETETTDM sq."""
        out = np.abs(dip_moment_calculator(self.ETETTDM))
        out = zero_diagonals(out)
        return out

    def to_fc(self, lambda_cl=0.01, epsr=3.0, temperature=300.0, phosp=False, order=1):
        """Process to fc."""
        fckwargs = {
            "masses": self.masses,
            "hessian": self.hessian,
        }
        if phosp:
            fckwargs["_gradients"] = self.exc_grads_so
            fckwargs["energies"] = self.exc_en_so[1:]
            if order == 1:
                fckwargs["constants"] = self.tdms_sq
            elif order == 2:
                fckwargs["constants"] = self.tqms_sq
            elif order == 3:
                fckwargs["constants"] = self.toms_sq
        else:
            fckwargs["_gradients"] = self.exc_grads_S
            fckwargs["energies"] = self.exc_en_S
            if order == 1:
                fckwargs["constants"] = self.tdms_so_sq
            elif order == 2:
                fckwargs["constants"] = self.tqms_so_sq
            elif order == 3:
                fckwargs["constants"] = self.toms_so_sq
        fckwargs["_gradients"] = fckwargs["_gradients"].reshape(
            (
                fckwargs["_gradients"].shape[0],
                fckwargs["_gradients"].shape[1] * fckwargs["_gradients"].shape[2],
            )
        )

        fc = Franck_Condon(
            **fckwargs,
            lambda_cl=Units.convert(lambda_cl, "eV", "J"),
            eps_r=epsr,
            temperature=temperature,
        )
        return fc
