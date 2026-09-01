import math
import multiprocessing
import sys
import warnings
from copy import deepcopy
from functools import cached_property, partial

import adaptive
import loky
import numpy as np
import scipy
import scipy.integrate
import scipy.linalg
import scipy.sparse.linalg
import tqdm
from adaptive.learner.learner1D import curvature_loss_function
from scipy.integrate import quad

from .units import Units


def configure_multiprocessing(start_method="spawn", use_dill=True):
    """Configure multiprocessing explicitly before starting worker processes.

    If the application has already selected a start method, that choice is
    preserved even when ``start_method`` requests a different method. The
    optional ``dill`` pickler is also installed only when this function is
    called and ``use_dill`` is true.

    Args:
        start_method: Preferred multiprocessing start method when none has
            been selected yet.
        use_dill: Whether to use ``dill`` for multiprocessing pickling when
            it is installed.

    Returns:
        The active multiprocessing start method, or ``None`` if no method was
        requested and none was configured.
    """
    active_method = multiprocessing.get_start_method(allow_none=True)
    if active_method is None and start_method is not None:
        multiprocessing.set_start_method(start_method)
        active_method = start_method

    if use_dill:
        try:
            import dill

            multiprocessing.reduction.ForkingPickler = dill.Pickler
        except ImportError:
            pass

    return active_method


max_t = math.sqrt(sys.float_info.max) * 0.1

EPS = 1e-15
EPS_PHONON = Units.convert(1e-8, "eV", "J")
MIN_HESS = 1e22 * Units.convert(1.0, "au", "kg")


def find_nth_smallest(a, n):
    """Process find nth smallest."""
    return np.partition(a, n - 1)[n - 1]


def get_mass_weighted_coords(coords, diag_mass_matrix):
    """Return mass weighted coords."""
    return np.matmul(diag_mass_matrix, coords)


def get_com_coords(coords, mass_array):
    """Return com coords."""
    total_mass = np.sum(mass_array)
    average_pos = mass_array[:, None] * coords
    com = np.sum(average_pos, axis=0)
    com /= total_mass
    return com


def massweight_matrix(matrix, mass_matrix):
    """Process massweight matrix."""
    sqrt_mass_matrix = np.sqrt(mass_matrix)
    inv_sqrt_mass_matrix = np.linalg.inv(sqrt_mass_matrix)
    weighted_matrix = np.tensordot(matrix, inv_sqrt_mass_matrix, axes=(-1, 0))
    return weighted_matrix


def hessian_to_massweighted_hessian(hessian, mass_matrix):
    """Process hessian to massweighted hessian."""
    sqrt_mass_matrix = np.sqrt(mass_matrix)
    inv_sqrt_mass_matrix = np.linalg.inv(sqrt_mass_matrix)
    normalised_hessian = np.matmul(
        np.matmul(inv_sqrt_mass_matrix, hessian), inv_sqrt_mass_matrix
    )
    return normalised_hessian


def mass_vector_to_mass_matrix(masses):
    """Process mass vector to mass matrix."""
    mass_matrix = np.diag(np.repeat(masses, 3))
    return mass_matrix


def construct_translation_vectors(coords):
    """
    Construct normalized translation vectors for N atoms.

    Args:
        coords: (N, 3) array of atomic coordinates

    Returns:
        trans_vecs: (3N, 3) array of translation vectors
    """
    N = coords.shape[0]
    trans_vecs = np.zeros((3 * N, 3))

    for i in range(3):
        trans_vecs[i::3, i] = 1.0

    for i in range(3):
        trans_vecs[:, i] /= np.linalg.norm(trans_vecs[:, i])

    return trans_vecs


def construct_rotation_vectors(coords, masses):
    """
    Construct mass-weighted rotation vectors for N atoms.

    Args:
        coords: (N, 3) array of atomic coordinates
        masses: (N,) array of atomic masses

    Returns:
        rot_vecs: (3N, 3) array of rotation vectors
    """
    N = coords.shape[0]

    total_mass = np.sum(masses)
    com = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass

    r = coords - com
    sqrt_masses = np.sqrt(masses)

    rot_vecs = np.zeros((3 * N, 3))

    rot_vecs[1::3, 0] = sqrt_masses * r[:, 2]
    rot_vecs[2::3, 0] = -sqrt_masses * r[:, 1]

    rot_vecs[0::3, 1] = -sqrt_masses * r[:, 2]
    rot_vecs[2::3, 1] = sqrt_masses * r[:, 0]

    rot_vecs[0::3, 2] = sqrt_masses * r[:, 1]
    rot_vecs[1::3, 2] = -sqrt_masses * r[:, 0]

    for i in range(3):
        norm = np.linalg.norm(rot_vecs[:, i])
        if norm > 1e-10:
            rot_vecs[:, i] /= norm

    return rot_vecs


def construct_tr_projection_matrix(coords, masses):
    """
    Construct projection matrix to remove translation and rotation.

    Args:
        coords: (N, 3) array of atomic coordinates
        masses: (N,) array of atomic masses

    Returns:
        P: (3N, 3N) projection matrix
    """
    N = coords.shape[0]

    trans_vecs = construct_translation_vectors(coords)
    rot_vecs = construct_rotation_vectors(coords, masses)

    tr_vecs = np.hstack([trans_vecs, rot_vecs])

    norms = np.linalg.norm(tr_vecs, axis=0)
    tr_vecs = tr_vecs[:, norms > 1e-10]

    U, s, Vt = scipy.linalg.svd(tr_vecs, full_matrices=False)
    rank = np.sum(s > 1e-10)
    tr_basis = U[:, :rank]

    P = np.eye(3 * N) - tr_basis @ tr_basis.T

    return P


def get_delta_pos(inverted_hessian, gradients):
    """Return delta pos."""
    delta_pos = -1.0 * np.matmul(gradients, inverted_hessian)
    return delta_pos


def get_gradients(delta_pos, hessian):
    """Return gradients."""
    gradients = -1 * np.matmul(delta_pos, hessian)
    return gradients


def get_k_vectors(delta_pos, mass_matrix, normal_system):
    """Return k vectors."""
    sqrt_mass_matrix = np.sqrt(mass_matrix)
    mw_delta_pos = np.matmul(delta_pos, sqrt_mass_matrix)
    k_vectors = np.matmul(mw_delta_pos, normal_system)
    return k_vectors


def get_lambdas(k_vectors, omegas):
    """Return lambdas."""
    while k_vectors.ndim < 2:
        k_vectors = np.expand_dims(k_vectors, axis=0)
    while omegas.ndim < k_vectors.ndim:
        omegas = np.expand_dims(omegas, axis=0)
    return 0.5 * (k_vectors * omegas) ** 2


def get_eigensystem(normalised_hessian, herm=True):
    """Return eigensystem."""
    if herm:
        eigen_vals, eigen_vecs = scipy.linalg.eigh(normalised_hessian)
    else:
        eigen_vals, eigen_vecs = scipy.linalg.eig(normalised_hessian)
    sorted_indices = np.argsort(np.real(eigen_vals))

    eigen_vals = eigen_vals[sorted_indices]
    eigen_vecs = eigen_vecs[:, sorted_indices]

    return eigen_vals, eigen_vecs


def eigensystem_to_matrix(eigen_values, normal_system):
    """Process eigensystem to matrix."""
    mat = np.diag(eigen_values)
    mat = np.matmul(normal_system, mat)
    mat = np.matmul(mat, (normal_system.T))
    return mat


def get_exponential_integrand_VGFC(
    t, E, energies, phonon_energies, lambdas, lambda_cl, kBT, acceptor=1
):
    """Return exponential integrand VGFC."""
    lclkBT = np.sqrt(lambda_cl * kBT)
    phonon_energies = np.expand_dims(
        phonon_energies, axis=tuple(np.arange(0, lambdas.ndim - phonon_energies.ndim))
    )
    invlclkBT = 1.0 / lclkBT
    phonon_energies = np.maximum(phonon_energies, EPS_PHONON)

    term_1 = (1.0j) * (energies - E) * (2 * acceptor - 1)
    term_1 += (1.0j) * lambda_cl
    term_1 = term_1 * t
    term_1 = term_1 * invlclkBT

    term_2 = -(t**2)

    coth_term = 1.0 / (np.tanh(phonon_energies / (2.0 * kBT)))
    cos_term = phonon_energies * t * invlclkBT
    sin_term = 1.0j * np.sin(cos_term)
    cos_term = coth_term * (np.cos(cos_term) - 1)
    lhw_term = lambdas / (phonon_energies)
    lhw_term = np.nan_to_num(
        lhw_term, copy=True, nan=0.0, posinf=np.inf, neginf=-np.inf
    )
    term_3 = lhw_term * (cos_term + sin_term)
    term_3 = np.sum(term_3, axis=-1)

    output = deepcopy(term_1)
    output += term_2
    output += term_3
    return output


def get_integrand_VGFC(
    t,
    E,
    energies,
    phonon_energies,
    lambdas,
    lambda_cl,
    kBT,
    constants,
    acceptor=1,
):
    """Return integrand VGFC."""
    lclkBT = np.sqrt(lambda_cl * kBT)
    if abs(t) > max_t:
        warnings.warn(f"Time t is too high! t = {t}, max = {max_t}.", stacklevel=2)
        return 0.0
    exp_term = get_exponential_integrand_VGFC(
        t, E, energies, phonon_energies, lambdas, lambda_cl, kBT, acceptor
    )
    exp_term = np.real(np.exp(exp_term))

    exp_term = exp_term * constants

    exp_term = np.sum(exp_term, axis=0) / lclkBT
    exp_term = exp_term / (2 * np.pi)
    return exp_term


def get_integrand_VGnonradFC(
    t,
    E,
    energies,
    phonon_energies,
    lambdas,
    lambda_cl,
    kBT,
    constants,
    acceptor=1,
):
    """Return integrand VGnonradFC."""
    lclkBT = np.sqrt(lambda_cl * kBT)
    if abs(t) > max_t:
        warnings.warn(f"Time t is too high! t = {t}, max = {max_t}.", stacklevel=2)
        return 0.0
    phonon_energies = np.expand_dims(
        phonon_energies, axis=tuple(np.arange(0, lambdas.ndim - 1))
    )
    phonon_energies = np.maximum(phonon_energies, EPS_PHONON)
    exp_term = get_exponential_integrand_VGFC(
        t, E, energies, phonon_energies, lambdas, lambda_cl, kBT, acceptor
    )
    exp_term = np.exp(exp_term)

    phonon_energies_kBT = phonon_energies / kBT
    phonon_energies_lclkBT = phonon_energies / lclkBT
    sinh_term = np.sinh(phonon_energies_kBT * 0.5)
    tw_term = phonon_energies_lclkBT * t
    term1 = (
        2 * phonon_energies * sinh_term * np.cos(tw_term + 0.5j * phonon_energies_kBT)
    )
    term2 = np.sin(0.5 * tw_term) * np.sin(0.5 * (tw_term + 1j * phonon_energies_kBT))
    term2 = term2 * term2 * 8 * lambdas
    non_rad_term = sinh_term * sinh_term
    non_rad_term = 0.25 * (term2 + term1) / (non_rad_term)

    exp_term = exp_term * constants * non_rad_term

    exp_term = np.sum(exp_term, axis=1)

    exp_term = np.sum(exp_term, axis=0) / lclkBT
    exp_term = exp_term / (2 * np.pi)
    return np.real(exp_term)


def get_integral_E_noreorgs(E, energies, lambda_cl, kBT, constants, acceptor=1):
    """Return integral E noreorgs."""
    lclkBT = np.sqrt(lambda_cl * kBT)

    out = (E - energies) * (2 * acceptor - 1)
    out -= lambda_cl
    out = out**2
    out /= 4 * lclkBT**2
    out *= -1
    out = np.exp(out)
    out = out * constants
    out = np.sum(out, axis=0)
    out = out / (2 * np.sqrt(np.pi) * lclkBT)
    return out


def get_integral_E_reorgs(
    E,
    energies,
    phonon_energies,
    lambdas,
    lambda_cl,
    kBT,
    constants,
    acceptor=1,
    nonrad=False,
    interval=10.0,
    epsrel=1e-3,
    epsabs=0.0,
    limit=int(1e4),
    maxp1=int(1e2),
):
    """Return integral E reorgs."""
    assert np.all(phonon_energies >= 0.0)
    func_args = tuple(
        [
            E,
            energies,
            phonon_energies,
            lambdas,
            lambda_cl,
            kBT,
            constants,
            acceptor,
        ]
    )
    if nonrad:
        func = get_integrand_VGnonradFC
    else:
        func = get_integrand_VGFC

    integral = np.asarray(
        quad(
            func,
            0.0,
            interval,
            args=func_args,
            epsabs=epsabs,
            limit=limit,
            epsrel=epsrel,
            maxp1=maxp1,
        )
    )
    integral += np.asarray(
        quad(
            func,
            -interval,
            0.0,
            args=func_args,
            epsabs=epsabs,
            limit=limit,
            epsrel=epsrel,
            maxp1=maxp1,
        )
    )
    if interval < np.inf:
        integral += np.asarray(
            quad(
                func,
                -np.inf,
                -interval,
                args=func_args,
                epsabs=epsabs,
                limit=limit,
                epsrel=epsrel,
                maxp1=maxp1,
            )
        )
        integral += np.asarray(
            quad(
                func,
                interval,
                np.inf,
                args=func_args,
                epsabs=epsabs,
                limit=limit,
                epsrel=epsrel,
                maxp1=maxp1,
            )
        )
    return (integral[0], integral[1])


def get_integral_E(
    E,
    energies,
    phonon_energies,
    lambdas,
    lambda_cl,
    kBT,
    constants,
    acceptor=1,
    nonrad=False,
    intkwargs={},
):
    """Return integral E."""
    if lambdas is None:
        out = (
            get_integral_E_noreorgs(
                E,
                energies,
                lambda_cl,
                kBT,
                constants,
                acceptor,
            ),
            0.0,
        )
    else:
        out = get_integral_E_reorgs(
            E,
            energies,
            phonon_energies,
            lambdas,
            lambda_cl,
            kBT,
            constants,
            acceptor,
            nonrad,
            **intkwargs,
        )
    return out


def constant_prefactor(
    E, energies, phonon_energies, lambdas, lambda_cl, kBT, constants, acceptor, val=1.0
):
    """Return a constant value, regardless of the input."""
    return val


def fluor_prefactor(
    E, energies, phonon_energies, lambdas, lambda_cl, kBT, constants, acceptor, val
):
    """Return val multiplied by the cube of the first argument."""

    return val * (E**3)


def absor_prefactor(
    E, energies, phonon_energies, lambdas, lambda_cl, kBT, constants, acceptor, val
):
    """Return val multiplied by the first argument to the first power."""

    return val * (E**1)


def get_spectrum(
    E,
    energies,
    phonon_energies,
    lambdas,
    lambda_cl,
    kBT,
    constants,
    acceptor,
    prefactor=constant_prefactor,
    excl_error=True,
    **kwargs,
):
    """Return spectrum."""
    out = get_integral_E(
        E,
        energies,
        phonon_energies,
        lambdas,
        lambda_cl,
        kBT,
        constants,
        acceptor,
        **kwargs,
    )
    pref = prefactor(
        E, energies, phonon_energies, lambdas, lambda_cl, kBT, constants, acceptor
    )
    out = tuple(f * pref for f in out)

    if excl_error:
        out = out[0]
    return out


def make_spectrum_function(
    energies,
    phonon_energies,
    lambdas,
    lambda_cl,
    kBT,
    constants,
    acceptor,
    **kwargs,
):
    """Process make spectrum function."""
    fkwargs = {
        "energies": energies,
        "phonon_energies": phonon_energies,
        "lambdas": lambdas,
        "lambda_cl": lambda_cl,
        "kBT": kBT,
        "constants": constants,
        "acceptor": acceptor,
        **kwargs,
    }
    return partial(get_spectrum, **fkwargs)


class Franck_Condon:
    """
    Compute Franck-Condon factors and related vibronic quantities.

    Parameters
    ----------
    reduce_dim : int | bool
        Number of low-frequency modes to remove (6 for full rotation-translation), or True to remove all zero modes.
    eps_r : float
        Relative permittivity.
    lambda_cl : float
        Classical reorganization energy (J).
    temperature : float
        Temperature (K).
    sigma_PL : float
        Broadening parameter for photoluminescence (J).
    debug : bool
        Enable debug output.
    clip : float
        Maximum Huang-Rhys factor per mode.
    force_positive_omega : bool
        Force all omegas positive.
    **kwargs
        Additional attributes to set on the instance.
    """

    def __init__(
        self,
        reduce_dim: int | bool = 6,
        eps_r: float = 3.0,
        lambda_cl: float = 0.01 * Units.conversion_ratio("eV", "J"),
        temperature: float = 300.0,
        sigma_PL: float = 0.0,
        debug: bool = False,
        clip: float = np.inf,
        force_positive_omega: bool = False,
        **kwargs,
    ):
        """Initialize the object."""
        self.reduce_dim = reduce_dim
        self.eps_r = eps_r
        self.lambda_cl = lambda_cl
        self.temperature = temperature
        self.sigma_PL = sigma_PL
        self.debug = debug
        self.clip = clip
        self.force_positive_omega = force_positive_omega

        self.b_bl = 10.0

        cte = Units.constants
        self.eps_0 = cte["eps0"]
        self.c = cte["c"]
        self.N_A = cte["NA"]
        self.k_B = cte["k_B"]
        self.hbar = cte["hbar"]

        self.weight_fluor = False
        self.shift_energies_fluor = False
        self.acceptor = True

        for key, val in kwargs.items():
            setattr(self, key, val)

    def check_var(self, var: str) -> bool:
        """Process check var."""
        return hasattr(self, var)

    @property
    def masses(self) -> np.ndarray:
        """Process masses."""
        return self._masses

    @masses.setter
    def masses(self, vals) -> None:
        """Process masses."""
        self._masses = np.asarray(vals, dtype=float)

    @property
    def nacmes(self) -> np.ndarray:
        """Process nacmes."""
        return self._nacmes

    @nacmes.setter
    def nacmes(self, vals) -> None:
        """Process nacmes."""
        self._nacmes = np.asarray(vals, dtype=complex)

    @property
    def coords(self) -> np.ndarray:
        """Process coords."""
        return self._coords

    @coords.setter
    def coords(self, vals) -> None:
        """Process coords."""
        self._coords = np.asarray(vals, dtype=float)

    @property
    def total_mass(self) -> float:
        """Process total mass."""
        return self.mass_matrix.trace() / 3

    @property
    def mass_matrix(self) -> np.ndarray:
        """Process mass matrix."""
        return mass_vector_to_mass_matrix(self.masses)

    @property
    def sqrt_mass_matrix(self) -> np.ndarray:
        """Process sqrt mass matrix."""
        return mass_vector_to_mass_matrix(np.sqrt(self.masses))

    @cached_property
    def inv_mass_matrix(self) -> np.ndarray:
        """Process inv mass matrix."""
        return np.linalg.inv(self.mass_matrix)

    @cached_property
    def inv_sqrt_mass_matrix(self) -> np.ndarray:
        """Process inv sqrt mass matrix."""
        return np.linalg.inv(self.sqrt_mass_matrix)

    @property
    def gradients(self):
        """Process gradients."""
        if not hasattr(self, "_gradients"):
            self._gradients = get_gradients(self.delta_pos, self.hessian)
        return self._gradients

    @cached_property
    def R_com(self) -> np.ndarray:
        """Process R com."""
        return (self.coords * self.masses[:, None]).sum(axis=0) / self.total_mass

    @cached_property
    def rottrans_projector(self) -> np.ndarray:
        """Process rottrans projector."""
        s = self.coords - self.R_com[None, :]

        N = self.mass_matrix.shape[0]
        modes = np.zeros((N, 6))

        e = np.eye(3)

        sqrt_m = np.sqrt(self.masses)
        modes[0::3, 0] = sqrt_m
        modes[1::3, 1] = sqrt_m
        modes[2::3, 2] = sqrt_m

        for i in range(N // 3):
            for a in range(3):
                vec = np.cross(e[a], s[i])
                modes[3 * i : 3 * i + 3, 3 + a] = sqrt_m[i] * vec

        Q, R = np.linalg.qr(modes, mode="reduced")

        P = np.eye(N) - Q @ Q.T

        return P

    @property
    def mass_weighted_gradients(self) -> np.ndarray:
        """Process mass weighted gradients."""
        return np.tensordot(self.gradients, self.inv_sqrt_mass_matrix, axes=(-1, 0))

    @property
    def mass_weighted_gradients_rot(self) -> np.ndarray:
        """Process mass weighted gradients rot."""
        P = self.rottrans_projector
        return self.mass_weighted_gradients @ P.T

    @property
    def mass_weighted_nacmes(self) -> np.ndarray:
        """Process mass weighted nacmes."""
        return np.tensordot(self.nacmes, self.inv_sqrt_mass_matrix, axes=(-1, 0))

    @property
    def mass_weighted_nacmes_rot(self) -> np.ndarray:
        """Process mass weighted nacmes rot."""
        P = self.rottrans_projector
        return self.mass_weighted_nacmes @ self.normal_modes_reduced

    @property
    def mass_weighted_hessian(self) -> np.ndarray:
        """Process mass weighted hessian."""
        return np.matmul(
            np.matmul(self.inv_sqrt_mass_matrix, self.hessian),
            self.inv_sqrt_mass_matrix,
        )

    @property
    def mass_weighted_hessian_rot(self) -> np.ndarray:
        """Process mass weighted hessian rot."""
        P = self.rottrans_projector
        return P @ self.mass_weighted_hessian @ (P.T)

    @cached_property
    def eigsys(self):
        """Process eigsys."""
        return get_eigensystem(self.mass_weighted_hessian_rot)

    @cached_property
    def eigvals(self) -> np.ndarray:
        """Process eigvals."""
        return self.eigsys[0]

    @cached_property
    def eigvecs(self) -> np.ndarray:
        """Process eigvecs."""
        return self.eigsys[1]

    @cached_property
    def mode_mask(self) -> np.ndarray:
        """Process mode mask."""
        if not self.reduce_dim:
            return np.ones_like(self.eigvals, bool)
        vals = np.real(self.omegas)
        if isinstance(self.reduce_dim, int):
            cutoff = find_nth_smallest(vals, self.reduce_dim)
            return vals > cutoff
        return vals > 0.0

    @property
    def omegas(self) -> np.ndarray:
        """Process omegas."""
        out = np.lib.scimath.sqrt(self.eigvals)
        if self.force_positive_omega:
            out = np.abs(out)
        return out

    @cached_property
    def omegas_reduced(self) -> np.ndarray:
        """Process omegas reduced."""
        return np.real(self.omegas[self.mode_mask])

    @cached_property
    def normal_modes_reduced(self) -> np.ndarray:
        """Process normal modes reduced."""
        return self.eigvecs[:, self.mode_mask]

    @cached_property
    def phonon_energies(self) -> np.ndarray:
        """Process phonon energies."""
        return self.omegas_reduced * self.hbar

    @property
    def mass_weighted_hessian_reduced(self):
        """Process mass weighted hessian reduced."""
        return eigensystem_to_matrix(
            self.omegas_reduced * self.omegas_reduced, self.normal_modes_reduced
        )

    @property
    def mass_weighted_hessian_inverted(self):
        """Process mass weighted hessian inverted."""
        return eigensystem_to_matrix(
            1.0 / (self.omegas_reduced * self.omegas_reduced),
            self.normal_modes_reduced,
        )

    @property
    def inverted_hessian_reduced(self):
        """Process inverted hessian reduced."""
        inv_sqrt_mass_matrix = self.inv_sqrt_mass_matrix

        inv_hessian = np.matmul(
            inv_sqrt_mass_matrix, self.mass_weighted_hessian_inverted
        )
        inv_hessian = np.matmul(inv_hessian, inv_sqrt_mass_matrix)
        return inv_hessian

    @property
    def inverted_hessian_stabilised(self):
        """Process inverted hessian stabilised."""
        inv_hessian = scipy.linalg.inv(
            self.hessian + np.eye(self.hessian.shape[0]) * MIN_HESS
        )
        return inv_hessian

    @cached_property
    def inverted_hessian(self):
        """Process inverted hessian."""
        return self.inverted_hessian_reduced

    @property
    def delta_pos(self):
        """Process delta pos."""
        if not hasattr(self, "_delta_pos"):
            self._delta_pos = -1.0 * np.matmul(self.gradients, self.inverted_hessian)
        return self._delta_pos

    @property
    def mw_delta_pos_AH(self):
        """
        Returns an (M, N) array of mass‑weighted displacements,
        one row for each of the M gradient vectors.
        """

        trust_radius = 1e-22
        S_diag = 1.0 / (trust_radius**2)

        G = self.mass_weighted_gradients
        H = self.mass_weighted_hessian
        M, n = G.shape

        Δ = np.zeros_like(G)

        for i in range(M):
            g = G[i]

            A = np.zeros((n + 1, n + 1))
            B = np.zeros_like(A)

            A[:n, :n] = H
            A[:n, n] = g
            A[n, :n] = g

            B[np.diag_indices(n)] = S_diag

            eigvals, eigvecs = scipy.linalg.eigh(A, B, subset_by_index=(0, 0))

            v = eigvecs[:, 0]

            Δ[i] = v[:n] / v[n]

        return Δ

    @property
    def mw_delta_pos_QN(self):
        """Process mw delta pos QN."""
        return -1.0 * np.matmul(
            self.mass_weighted_gradients_rot, self.mass_weighted_hessian_inverted
        )

    @property
    def mw_delta_pos(self):
        """Process mw delta pos."""
        return self.mw_delta_pos_QN

    @property
    def k_vectors(self) -> np.ndarray:
        """Process k vectors."""
        return self.mw_delta_pos @ self.normal_modes_reduced

    @cached_property
    def lambdas(self) -> np.ndarray:
        """Process lambdas."""
        lam = get_lambdas(self.k_vectors, self.omegas_reduced)
        lam = np.clip(lam / self.phonon_energies, 0, self.clip) * self.phonon_energies
        return lam

    @property
    def reorganization_energies(self):
        """Process reorganization energies."""
        return self.lambdas

    @property
    def l_over_hw(self):
        """Process l over hw."""
        return self.reorganization_energies / (self.phonon_energies[None, :])

    @property
    def kBT(self):
        """Process kBT."""
        return self.temperature * Units.constants["k_B"]

    @property
    def lclkBT(self):
        """Process lclkBT."""
        return np.sqrt(self.lambda_cl * self.kBT)

    @property
    def lclkbT(self):
        """Process lclkbT."""
        return self.lclkBT

    @property
    def n(self):
        """Process n."""
        return np.sqrt(self.eps_r)

    @property
    def energies(self):
        """Process energies."""
        return self._energies

    @energies.setter
    def energies(self, value):
        """Process energies."""
        if value is not None and not isinstance(value, np.ndarray):
            value = np.array(value)
        self._energies = value

    def wavelength_to_E(self, wavelength):
        """Process wavelength to E."""
        E = 2 * np.pi * self.hbar * self.c / (wavelength)
        return E

    @property
    def fluor_probabilities(self):
        """Process fluor probabilities."""
        renormE = self.energies - np.min(self.energies)
        probs = np.exp(-renormE / self.kBT)
        return probs / np.sum(probs)

    def correct_energies(self, energies):
        """Process correct energies."""
        energies = deepcopy(energies - np.sum(self.reorganization_energies, axis=-1))
        return energies

    def setup_spectrum(
        self,
        name="readied",
        prepare=False,
        fkwargs={},
        **kwargs,
    ):
        """Process setup spectrum."""
        nonrad = False
        if "read" in name.lower():

            prefactor = partial(constant_prefactor, val=1.0)
        elif "totj" in name.lower():
            prefval = deepcopy(2 * np.pi / self.hbar)
            prefactor = partial(constant_prefactor, val=prefval)
        elif "fluor" in name.lower():
            self.acceptor = False
            self.weight_fluor = True

            denom = (self.n) ** 3
            nom = 3 * np.pi * self.eps_0 * self.eps_r
            nom *= (self.c**3) * (self.hbar**4)
            prefval = deepcopy(denom / nom)
            prefactor = partial(fluor_prefactor, val=prefval)
        elif "absor" in name.lower():
            self.acceptor = True
            self.weight_fluor = False

            denom = self.n * np.pi * self.N_A
            nom = 3.0 * self.eps_0 * self.eps_r * self.hbar * self.c * np.log(self.b_bl)
            prefval = deepcopy(denom / nom)
            prefactor = partial(absor_prefactor, val=prefval)
        elif "nonrad" in name.lower():
            print("Non radiative decay is still under development.")
            self.acceptor = False
            self.constants = np.abs(self.mass_weighted_nacmes_rot) ** 2
            prefval = deepcopy(2 * np.pi * self.hbar * self.hbar / self.hbar)
            prefactor = partial(constant_prefactor, val=prefval)
            nonrad = True
        else:
            raise ValueError(f"Spectrum {name} not implemented.")

        if "abs" in name.lower():
            self.acceptor = True
            self.weight_fluor = False
        elif "emis" in name.lower():
            self.acceptor = False
            self.weight_fluor = True

        if self.weight_fluor:
            constants = self.constants * self.fluor_probabilities
        else:
            constants = self.constants

        energies = self.energies

        intensities = []
        if prepare:
            raise NotImplementedError("This was removed.")

        f = make_spectrum_function(
            energies,
            deepcopy(self.phonon_energies),
            deepcopy(self.lambdas),
            deepcopy(self.lambda_cl),
            deepcopy(self.kBT),
            deepcopy(constants),
            deepcopy(self.acceptor),
            prefactor=prefactor,
            nonrad=nonrad,
            **fkwargs,
        )
        return f

    def adap_spectrum(
        self,
        name="readied",
        prepare=False,
        E_interval=Units.convert([0.0, 4.0], "eV", "J"),
        rkwargs={"duration_goal": 10.0, "executor": loky.get_reusable_executor()},
        fkwargs={},
        loss_func=curvature_loss_function(1.0, 0.1, 0.1),
        **kwargs,
    ):
        """Process adap spectrum."""
        E_interval = np.asarray(E_interval)
        assert E_interval.ndim == 1
        if E_interval.shape[0] == 1:
            E_interval = [0, E_interval[0]]

        f = self.setup_spectrum(
            name=name,
            prepare=prepare,
            fkwargs=fkwargs,
            **kwargs,
        )
        learner = adaptive.Learner1D(
            f, bounds=(E_interval[0], E_interval[1]), loss_per_interval=loss_func
        )
        runner = adaptive.BlockingRunner(learner, **rkwargs)
        data = learner.to_numpy()
        self.last_energies = deepcopy(data[:, 0])
        self.last_intensities = deepcopy(data[:, 1])
        return self.last_energies, self.last_intensities

    def calc_rate(
        self,
        name="nonrad",
        prepare=False,
        energy=Units.convert(0.0, "eV", "J"),
        fkwargs={},
        **kwargs,
    ):
        """Process calc rate."""
        f = self.setup_spectrum(
            name=name,
            prepare=prepare,
            fkwargs=fkwargs,
            **kwargs,
        )
        out = f(energy)
        return out

    def loop_spectrum(
        self, E_interval, dE=0.1, name="readied", prepare=False, fkwargs={}, **kwargs
    ):
        """Process loop spectrum."""
        E_interval = np.asarray(E_interval)
        if E_interval.shape[0] == 1:
            N_points = int(E_interval[0] / dE) + 1
            E_interval = np.linspace(0, E_interval[0], N_points)
        elif E_interval.shape[0] == 2:
            N_points = int((E_interval[1] - E_interval[0]) / dE) + 1
            E_interval = np.linspace(E_interval[0], E_interval[1], N_points)

        f = self.setup_spectrum(
            name=name,
            prepare=prepare,
            fkwargs=fkwargs,
            **kwargs,
        )
        intensities = []
        for energy in (pbar := tqdm.tqdm(E_interval)):
            inten = f(Units.convert(energy, "eV", "J"))
            intensities.append(inten)

        self.last_energies = E_interval
        self.last_intensities = np.asarray(intensities)
        return self.last_intensities
