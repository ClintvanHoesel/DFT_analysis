import os
import os.path
import types
import warnings
from functools import cache, cached_property

import numpy as np
import tqdm

from .._plams import require_plams
from ..multipole_calculators import (
    list_to_symmetric_matrix,
    list_to_symmetric_rank3_tensor,
)

try:
    import dill as pickle
except ImportError:
    import pickle

import itertools

MAX_GRADIENT = 1e0


def add_to_instance(instance):
    """Return a decorator that binds a function as a method on *instance*."""

    def decorator(function):
        """Process decorator."""
        setattr(instance, function.__name__, types.MethodType(function, instance))
        return function

    return decorator


def Jeff(jab, sab, energy_a, energy_b):
    """Return the overlap-corrected effective transfer integral."""
    return jab - 0.5 * (energy_a + energy_b) * sab


def non_decreasing_combinations(elements, n):
    """Process non decreasing combinations."""
    return itertools.combinations_with_replacement(elements, n)


def pass_molecule(source, target, charge):
    """Process pass molecule."""
    target.depend.append(source)

    @add_to_instance(target)
    def prerun(self):
        """Process prerun."""
        self.molecule = source.results.get_main_molecule()

        self.molecule.properties.charge = charge


def to_input_order(kffile, data):
    """to_input_order(self, data)
    Reorder any iterable *data* from the internal atom order to the input atom order. The length of *data* must be equal to the number of atoms, otherwise an exception is raised. Returned value is a container of the same type as *data*.
    """
    mapping = _int2inp(kffile)
    if len(mapping) != len(data):
        raise Exception(
            "to_input_order() got an argument with incorrect length. Length must be equal to the number of atoms"
        )
    t = np.array if isinstance(data, np.ndarray) else type(data)
    return t([data[mapping[i] - 1] for i in range(len(mapping))])


def _atomic_numbers_input_order(kffile):
    """_atomic_numbers_input_order()
    Return a list of atomic numbers, in the input order.
    """
    PT = require_plams()[4]
    n = kffile.read("Geometry", "nr of atoms")
    tmp = kffile.read("Geometry", "atomtype").split()
    atomtypes = {i + 1: PT.get_atomic_number(tmp[i]) for i in range(len(tmp))}
    atomtype_idx = kffile.read("Geometry", "fragment and atomtype index")[-n:]
    atnums = [atomtypes[i] for i in atomtype_idx]
    return to_input_order(kffile, atnums)


def _int2inp(kffile):
    """_int2inp()
    Get mapping from the internal atom order to the input atom order.
    """
    aoi = kffile.read("Geometry", "atom order index")
    n = len(aoi) // 2
    return aoi[:n]


def compute_transition_dipole(C, dipole_integrals):
    """
    Computes transition dipole moment from MO coefficients and dipole integrals.

    Parameters:
    C (ndarray): TD-DFT excitation coefficients (N_occ, N_virt).
    dipole_integrals (ndarray): MO dipole integrals (N_occ, N_virt, 3) for (x,y,z).

    Returns:
    ndarray: Transition dipole moment vector (x, y, z).
    """
    transition_dipole = np.einsum("ij,ijx->x", C, dipole_integrals)
    return transition_dipole


def get_molecule(
    kffile,
    section="Geometry",
    variable="xyz InputOrder",
    unit="bohr",
    internal=False,
    n=1,
):
    """get_molecule(section, variable, unit='bohr', internal=False, n=1)
    Read molecule coordinates from *section*/*variable* of the main KF file.

    Returned |Molecule| instance is created by copying a molecule from associated |SCMJob| instance and updating atomic coordinates with values read from *section*/*variable*. The format in which coordinates are stored is not consistent for all programs or even for different sections of the same KF file. Sometimes coordinates are stored in bohr, sometimes in angstrom. The order of atoms can be either input order or internal order. These settings can be adjusted with *unit* and *internal* parameters. Some variables store more than one geometry, in those cases *n* can be used to choose the preferred one.
    """
    Atom, Molecule, _, _, _ = require_plams()
    atnums = _atomic_numbers_input_order(kffile)
    natoms = len(atnums)
    coords = kffile.read(section, variable)
    coords = [coords[i : i + 3] for i in range(0, len(coords), 3)]
    if len(coords) > natoms:
        if len(coords) < n * natoms:
            raise Exception(
                "get_molecule() failed. Not enough data in {}%{} to extract geometry no {}".format(
                    section, variable, n
                )
            )
        coords = coords[(n - 1) * natoms : n * natoms]
    if internal:
        mapping = _int2inp(kffile)
        coords = [coords[mapping[i] - 1] for i in range(len(coords))]
    ret = Molecule()
    for z, crd in zip(atnums, coords):
        ret.add_atom(Atom(atnum=z, coords=crd, unit=unit))
    ret.guess_bonds()
    return ret


class Single_RKF_Reader:
    def __init__(self, rkfpath, load=False, preload=False, name: str = ""):
        """Initialize the object."""
        self.rkfpath = rkfpath
        self.name = name
        self.pickle_path = rkfpath + f"{name}" + ".pick"

        assert os.path.isfile(self.rkfpath), f"RKF file {self.rkfpath} does not exist"
        if load:
            try:
                self.load_pickle()
            except:
                load = False

        KFFile = require_plams()[3]
        self.rkffile = KFFile(rkfpath)

        self.name = os.path.basename(os.path.dirname(rkfpath))

        if preload:
            self.__obtain_results__()
            self.save_pickle()

    @cached_property
    def n_atoms(self):
        """Process n atoms."""
        return int(self.read("Geometry", "nr of atoms"))

    @cached_property
    def bond_energy(self):
        """Process bond energy."""
        return self.rkffile.read("Energy", "Bond Energy")

    @cached_property
    def unrestricted(self):
        """Process unrestricted."""
        try:
            self.nmos_B = self.rkffile.read("A", "nmo_B")
            return True
        except:
            return False

    @cached_property
    def n_exc_S(self):
        """Process n exc S."""
        return int(self.rkffile.read("Excitations SS A", "nr of excenergies"))

    @cached_property
    def exc_en_S(self):
        """Process exc en S."""
        return np.array(self.rkffile.read("Excitations SS A", "excenergies"))

    @cached_property
    def tdms_S(self):
        """Process tdms S."""
        return np.array(
            self.rkffile.read("Excitations SS A", "transition dipole moments")
        ).reshape((self.n_exc_S, 3))

    def load_pickle(self):
        """Process load pickle."""
        with open(self.pickle_path, "rb") as f:
            self.__dict__.update(pickle.load(f))

    def save_pickle(self):
        """Process save pickle."""
        with open(self.pickle_path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def __obtain_results__(self):
        """Handle obtain results internally."""
        for func in sorted(
            [method for method in dir(self) if method.startswith("__get_")]
        ):
            try:
                getattr(self, func)()
            except Exception as e:
                warnings.warn(
                    f"Did not succeed with function {func}.\r\nError obtained was {e}",
                    stacklevel=2,
                )

        for func in sorted(
            [method for method in dir(self) if method.startswith("__pre_process_")]
        ):
            try:
                getattr(self, func)()
            except Exception as e:
                warnings.warn(
                    f"Did not succeed with function {func}.\r\nError obtained was {e}",
                    stacklevel=2,
                )

        for func in sorted(
            [method for method in dir(self) if method.startswith("__post_process_")]
        ):
            try:
                getattr(self, func)()
            except Exception as e:
                warnings.warn(
                    f"Did not succeed with function {func}.\r\nError obtained was {e}",
                    stacklevel=2,
                )

    def get_orbital_energy(self, iorb):
        """Return orbital energy."""
        if not self.unrestricted:
            assert iorb < len(self.orb_ens_A) * 2
            iorb = iorb % len(self.orb_ens_A)
        if iorb >= len(self.orb_ens_A):
            return self.orb_ens_B[iorb - len(self.orb_ens_A)]
        else:
            return self.orb_ens_A[iorb]

    @cached_property
    def GoWo_electron_energies(self):
        """Process GoWo electron energies."""
        if not self.unrestricted:
            return self.read("GW", "G0W0_QP_hole_ener")
        else:
            return np.stack(
                (
                    self.read("GW", "G0W0_QP_hole_ener_sp_A"),
                    self.read("GW", "G0W0_QP_hole_ener_sp_B"),
                ),
            )

    @cached_property
    def GoWo_hole_energies(self):
        """Process GoWo hole energies."""
        if not self.unrestricted:
            return self.read("GW", "G0W0_QP_part_ener")
        else:
            return np.stack(
                (
                    self.read("GW", "G0W0_QP_part_ener_sp_A"),
                    self.read("GW", "G0W0_QP_part_ener_sp_B"),
                ),
            )

    @property
    def GoWo_energies(self):
        """Process GoWo energies."""
        return np.concatenate((self.GoWo_electron_energies, self.GoWo_hole_energies))

    @cached_property
    def HOMO_energy(self):
        """Process HOMO energy."""
        return float(np.max(self.rkffile.read("AMSResults", "HOMOEnergy")))

    @cached_property
    def LUMO_energy(self):
        """Process LUMO energy."""
        return float(np.min(self.rkffile.read("AMSResults", "LUMOEnergy")))

    @cached_property
    def hessian(self):
        """Process hessian."""
        return np.array(self.rkffile.read("AMSResults", "Hessian")).reshape(
            (3 * self.n_atoms, 3 * self.n_atoms)
        )

    @cached_property
    def electrons_tot(self):
        """Process electrons tot."""
        return int(self.rkffile.read("General", "electrons"))

    @cached_property
    def occs_A(self):
        """Process occs A."""
        return self.rkffile.read("A", "froc_A")

    @property
    def electrons_A(self):
        """Process electrons A."""
        return int(sum(self.occs_A))

    @cached_property
    def orb_ens_A(self):
        """Process orb ens A."""
        return self.rkffile.read("A", "eps_A")

    @cached_property
    def orb_ens(self):
        """Process orb ens."""
        if self.unrestricted:
            return np.array(zip(self.orb_ens_A, self.orb_ens_B))
        else:
            return self.orb_ens_A

    @cached_property
    def nmos_A(self):
        """Process nmos A."""
        return self.rkffile.read("A", "nmo_A")

    @property
    def noccorbs_A(self):
        """Process noccorbs A."""
        return sum(1 if occ > 0.5 else 0 for occ in self.occs_A)

    @property
    def nvirtorbs_A(self):
        """Process nvirtorbs A."""
        return sum(1 if occ < 0.5 else 0 for occ in self.occs_A)

    @cached_property
    def overlap(self):
        """Process overlap."""
        return self.rkffile.read("A", "S-CoreSFO")

    @cached_property
    def atom_masses(self):
        """Process atom masses."""
        return self.rkffile.read("Molecule", "AtomMasses")

    @property
    def mol(self):
        """Process mol."""
        return get_molecule(self.rkffile)

    @property
    def translated_mol(self):
        """Process translated mol."""
        translated_mol = self.mol.copy()
        translated_mol.translate(
            -1 * np.array(self.translated_mol.get_center_of_mass())
        )
        return translated_mol

    @property
    def norbs_tot(self):
        """Process norbs tot."""
        if self.unrestricted:
            return self.nmos_B + self.nmos_A
        else:
            return 2 * self.nmos_A

    @property
    def bas_A(self):
        """Process bas A."""
        return np.array(self.rkffile.read("A", "Eig-CoreSFO_A")).reshape(
            (self.nmos_A, self.nmos_A)
        )

    @property
    def bas_B(self):
        """Process bas B."""
        if self.unrestricted:
            return np.array(self.rkffile.read("A", "Eig-CoreSFO_B")).reshape(
                (self.nmos_B, self.nmos_B)
            )
        else:
            return self.bas_A

    @property
    def bas(self):
        """Process bas."""
        tmp = np.zeros((self.nmos_A + self.nmos_B, self.nmos_A + self.nmos_B))
        tmp[: self.nmos_A, : self.nmos_A] = self.bas_A
        tmp[self.nmos_B :, self.nmos_B :] = self.bas_B
        return tmp

    @property
    def sfo_ovl_matrix_A(self):
        """Process sfo ovl matrix A."""
        try:
            sfo_ovl_matrix_A = np.zeros((self.nmos_A, self.nmos_A))
            ovl_list = self.rkffile.read("SFO_Overlap_A", "A")
            indices = np.tril_indices(self.nmos_A)
            sfo_ovl_matrix_A[indices] = ovl_list
            dia = np.diag(sfo_ovl_matrix_A)
            return sfo_ovl_matrix_A + sfo_ovl_matrix_A.T - np.diag(dia)
        except:
            sfo_ovl_matrix_A = np.zeros((self.nmos_A, self.nmos_A))
            ovl_list = self.rkffile.read("SFO_Overlap", "A")
            indices = np.tril_indices(self.nmos_A)
            sfo_ovl_matrix_A[indices] = ovl_list
            dia = np.diag(sfo_ovl_matrix_A)
            return sfo_ovl_matrix_A + sfo_ovl_matrix_A.T - np.diag(dia)

    @property
    def sfo_ovl_matrix_B(self):
        """Process sfo ovl matrix B."""
        try:
            sfo_ovl_matrix_B = np.zeros((self.nmos_B, self.nmos_B))
            ovl_list = self.rkffile.read("SFO_Overlap_B", "A")
            indices = np.tril_indices(self.nmos_B)
            sfo_ovl_matrix_B[indices] = ovl_list
            dia = np.diag(sfo_ovl_matrix_B)
            return sfo_ovl_matrix_B + sfo_ovl_matrix_B.T - np.diag(dia)
        except:
            return self.sfo_ovl_matrix_A

    @property
    def sfo_ovl_matrix(self):
        """Process sfo ovl matrix."""
        sfo_ovl_matrix = np.zeros(
            (self.nmos_A + self.nmos_B, self.nmos_A + self.nmos_B)
        )
        sfo_ovl_matrix[: self.nmos_A, : self.nmos_A] = self.sfo_ovl_matrix_A
        sfo_ovl_matrix[self.nmos_B :, self.nmos_B :] = self.sfo_ovl_matrix_B
        return sfo_ovl_matrix

    @property
    def ovl_matrix(self):
        """Process ovl matrix."""
        return (self.bas) @ (self.sfo_ovl_matrix) @ (self.bas.T)

    def __pre_process_ovlp_fock_matrix__(self):
        """Handle pre process ovlp fock matrix internally."""
        print("Preprocessing")

        self.fock_matrix = (self.bas) @ (self.sfo_fock_matrix) @ (self.bas.T)

    def read(self, sec, sub):
        """Process read."""
        return self.rkffile.read(sec, sub)

    @cached_property
    def charge(self):
        """Process charge."""
        return self.read("Molecule", "Charge")

    @cached_property
    def occs_B(self):
        """Process occs B."""
        try:
            return self.rkffile.read("A", "froc_B")
        except:
            return self.occs_A

    @property
    def electrons_B(self):
        """Process electrons B."""
        return int(sum(self.occs_B))

    @cached_property
    def orb_ens_B(self):
        """Process orb ens B."""
        try:
            return self.rkffile.read("A", "eps_B")
        except:
            return self.orb_ens_A

    @cached_property
    def nmos_B(self):
        """Process nmos B."""
        try:
            return self.rkffile.read("A", "nmo_B")
        except:
            return self.nmos_A

    @property
    def noccorbs_B(self):
        """Process noccorbs B."""
        try:
            return sum(1 if occ > 0.5 else 0 for occ in self.occs_B)
        except:
            return self.noccorbs_A

    @property
    def nvirtorbs_B(self):
        """Process nvirtorbs B."""
        try:
            return sum(1 if occ < 0.5 else 0 for occ in self.occs_B)
        except:
            return self.nvirtorbs_A

    @property
    def sfo_fock_matrix_A(self):
        """Process sfo fock matrix A."""
        try:
            sfo_fock_matrix_A = np.zeros((self.nmos_A, self.nmos_A))
            ovl_list = self.rkffile.read("SFO_Fock_A", "A")
            indices = np.tril_indices(self.nmos_A)
            sfo_fock_matrix_A[indices] = ovl_list
            dia = np.diag(sfo_fock_matrix_A)
            return sfo_fock_matrix_A + sfo_fock_matrix_A.T - np.diag(dia)
        except:
            sfo_fock_matrix_A = np.zeros((self.nmos_A, self.nmos_A))
            ovl_list = self.rkffile.read("SFO_Fock", "A")
            indices = np.tril_indices(self.nmos_A)
            sfo_fock_matrix_A[indices] = ovl_list
            dia = np.diag(sfo_fock_matrix_A)
            return sfo_fock_matrix_A + sfo_fock_matrix_A.T - np.diag(dia)

    @property
    def sfo_fock_matrix_B(self):
        """Process sfo fock matrix B."""
        try:
            sfo_ovl_matrix_B = np.zeros((self.nmos_B, self.nmos_B))
            ovl_list = self.rkffile.read("SFO_Fock_B", "A")
            indices = np.tril_indices(self.nmos_B)
            sfo_ovl_matrix_B[indices] = ovl_list
            dia = np.diag(sfo_ovl_matrix_B)
            return sfo_ovl_matrix_B + sfo_ovl_matrix_B.T - np.diag(dia)
        except:
            return self.sfo_fock_matrix_A

    @property
    def sfo_fock_matrix(self):
        """Process sfo fock matrix."""
        sfo_ovl_matrix = np.zeros(
            (self.nmos_A + self.nmos_B, self.nmos_A + self.nmos_B)
        )
        sfo_ovl_matrix[: self.nmos_A, : self.nmos_A] = self.sfo_fock_matrix_A
        sfo_ovl_matrix[self.nmos_B :, self.nmos_B :] = self.sfo_fock_matrix_B
        return sfo_ovl_matrix

    @cached_property
    def GSESTDM(self):
        """Process GSESTDM."""
        return np.array(self.rkffile.read("All excitations", "GSES TDM")).reshape(
            (self.n_exc, 3)
        )

    @cached_property
    def ESESTDM(self):
        """Process ESESTDM."""
        return np.array(self.rkffile.read("All excitations", "ESES TDM")).reshape(
            (self.n_exc, self.n_exc, 3)
        )

    @cached_property
    def ETETTDM(self):
        """Process ETETTDM."""
        return np.array(self.rkffile.read("All excitations", "ETET TDM")).reshape(
            (self.n_exc_T, self.n_exc_T, 3)
        )

    @property
    def ETETTDMrep(self):
        """Process ETETTDMrep."""
        expanded = np.repeat(np.repeat(self.ETETTDM, 3, axis=0), 3, axis=1)
        i_indices, j_indices = np.meshgrid(
            np.arange(3 * self.n_exc_T), np.arange(3 * self.n_exc_T), indexing="ij"
        )
        mask = (i_indices % 3) == (j_indices % 3)
        expanded *= mask[..., None]
        return expanded

    @cached_property
    def n_exc(self):
        """Process n exc."""
        return int(self.rkffile.read("Excitations SS A", "nr of excenergies"))

    @cached_property
    def exc_en(self):
        """Process exc en."""
        return np.array(self.rkffile.read("Excitations SS A", "excenergies"))

    @cached_property
    def singlet_excited_gradients(self):
        """Process singlet excited gradients."""
        skel = self.rkffile.get_skeleton()
        skel = skel["Excitations SS A"]
        grads = {}
        last_dim = np.nan
        for grad in [elem for elem in skel if "gradient" in elem]:
            num = int(grad.split()[-1])
            grads[num - 1] = np.array(self.read("Excitations SS A", grad)).reshape(
                (-1, 3)
            )
            if last_dim is np.nan:
                last_dim = grads[num - 1].shape[0]
            else:
                assert (
                    last_dim == grads[num - 1].shape[0]
                ), "Inconsistent gradient dimensions"
        if last_dim is np.nan:
            raise Exception("No excited state gradients found.")
        out_array = np.zeros((self.n_exc, last_dim, 3))
        for grad, _val in grads.items():
            out_array[grad, :, :] = grads[grad]

        out_array = np.where(np.abs(out_array) > MAX_GRADIENT, 0.0, out_array)
        return out_array

    @cached_property
    def triplet_excited_gradients(self):
        """Process triplet excited gradients."""
        skel = self.rkffile.get_skeleton()
        skel = skel["Excitations ST A"]
        grads = {}
        last_dim = np.nan
        for grad in [elem for elem in skel if "gradient" in elem]:
            num = int(grad.split()[-1])
            grads[num - 1] = np.array(self.read("Excitations ST A", grad)).reshape(
                (-1, 3)
            )
            if last_dim is np.nan:
                last_dim = grads[num - 1].shape[0]
            else:
                assert (
                    last_dim == grads[num - 1].shape[0]
                ), "Inconsistent gradient dimensions"
        if last_dim is np.nan:
            raise Exception("No excited state gradients found.")
        out_array = np.zeros((self.n_exc, last_dim, 3))
        for grad, _val in grads.items():
            out_array[grad, :, :] = grads[grad]

        out_array = np.where(np.abs(out_array) > MAX_GRADIENT, 0.0, out_array)
        return out_array

    @cached_property
    def TDA(self):
        """Process TDA."""
        if (
            np.prod(
                np.array(
                    self.rkffile.read("Excitations SS A", f"eigenvector {1}")
                ).shape
            )
            == self.noccorbs_A * self.nvirtorbs_A
        ):
            return True
        else:
            return False

    @cached_property
    def XAS_tdm_length(self):
        """Process XAS tdm length."""
        return np.array(self.rkffile.read("XAS DATA", "Trans Dip Mom (l)")).reshape(
            (3, self.n_exc)
        )

    @cached_property
    def XAS_tdm_velocity(self):
        """Process XAS tdm velocity."""
        return (
            np.array(self.rkffile.read("XAS DATA", "Trans Dip Mom (v)"))
            .reshape((3, self.n_exc))
            .T
        )

    @cached_property
    def XAS_tqm_velocity(self):
        """Process XAS tqm velocity."""
        out = (
            np.array(self.rkffile.read("XAS DATA", "Trans Quad Mom"))
            .reshape((6, self.n_exc))
            .T
        )
        return list_to_symmetric_matrix(out)

    @cached_property
    def XAS_tom_velocity(self):
        """Process XAS tom velocity."""
        out = (
            np.array(self.rkffile.read("XAS DATA", "Trans Octu Mom"))
            .reshape((10, self.n_exc))
            .T
        )
        return list_to_symmetric_rank3_tensor(out)

    @cached_property
    def XAS_tmdm_velocity(self):
        """Process XAS tmdm velocity."""
        return (
            np.array(self.rkffile.read("XAS DATA", "Trans Mdip Mom"))
            .reshape((3, self.n_exc))
            .T
        )

    @cached_property
    def XAS_tmqm_velocity(self):
        """Process XAS tmqm velocity."""
        out = (
            np.array(self.rkffile.read("XAS DATA", "Trans Mquad Mom"))
            .reshape((6, self.n_exc))
            .T
        )
        return list_to_symmetric_matrix(out)

    @cached_property
    def transition_dipole_moments(self):
        """Process transition dipole moments."""
        return np.array(
            self.rkffile.read("Excitations SS A", "transition dipole moments")
        ).reshape((self.n_exc, 3))

    @cached_property
    def n_exc_T(self):
        """Process n exc T."""
        return int(self.rkffile.read("Excitations ST A", "nr of excenergies"))

    @cached_property
    def exc_en_T(self):
        """Process exc en T."""
        return np.array(self.rkffile.read("Excitations ST A", "excenergies"))

    @cached_property
    def TDA_T(self):
        """Process TDA T."""
        if (
            np.prod(
                np.array(
                    self.rkffile.read("Excitations ST A", f"eigenvector {1}")
                ).shape
            )
            == self.noccorbs_A * self.nvirtorbs_A
        ):
            return True
        else:
            return False

    @cached_property
    def transition_dipole_moments_T(self):
        """Process transition dipole moments T."""
        return np.array(
            self.rkffile.read("Excitations ST A", "transition dipole moments")
        ).reshape((self.n_exc_T, 3))

    @property
    def n_exc_so(self):
        """Process n exc so."""
        return self.n_exc + 3 * self.n_exc_T + 1

    @cached_property
    def exc_en_so(self):
        """Process exc en so."""
        return np.array(self.rkffile.read("Excitations SO A", "excenergies"))

    @cached_property
    def transition_dipole_moments_so(self):
        """Process transition dipole moments so."""
        return np.array(
            self.rkffile.read("Excitations SO A", "transition dipole moments")
        ).reshape((self.n_exc_so, 3))

    @cache
    def get_exc_eigenvec_raw_i(self, i):
        """Return exc eigenvec raw i."""
        return np.array(
            self.rkffile.read("Excitations SS A", "F Vectors" + f"{i+1}".rjust(5, " "))
        )

    @cache
    def get_exc_eigenvec_raw_mode_i(self, i, mode="S"):
        """Return exc eigenvec raw mode i."""
        return np.array(
            self.rkffile.read(
                f"Excitations S{mode} A", "F Vectors" + f"{i+1}".rjust(5, " ")
            )
        )

    def get_exc_eigenvec_i(self, i, mode="S"):
        """Return exc eigenvec i."""
        fvec = np.array(self.get_exc_eigenvec_raw_mode_i(i, mode))
        if not self.unrestricted:
            return fvec.reshape((self.noccorbs_A, self.nvirtorbs_A)), fvec.reshape(
                (self.noccorbs_A, self.nvirtorbs_A)
            )
        else:
            fvec_A = fvec[: self.noccorbs_A * self.nvirtorbs_A].reshape(
                (self.noccorbs_A, self.nvirtorbs_A)
            )
            fvec_B = fvec[self.noccorbs_A * self.nvirtorbs_A :].reshape(
                (self.noccorbs_B, self.nvirtorbs_B)
            )
            return fvec_A, fvec_B

    def get_exc_eigenvec_all(self, mode="S"):
        """Return exc eigenvec all."""
        if mode == "S":
            nexc = self.n_exc_S
        elif mode == "T":
            nexc = self.n_exc_T
        elif mode == "O":
            nexc = self.n_exc_so
        else:
            raise ValueError("Unknown mode {mode}.")

        out1 = np.array([self.get_exc_eigenvec_i(i, mode)[0] for i in range(nexc)])
        out2 = np.array([self.get_exc_eigenvec_i(i, mode)[1] for i in range(nexc)])
        return out1, out2

    def get_electric_multipole_moment(self, i, multipole, triplet=False):
        """Return electric multipole moment."""
        if not self.unrestricted:
            if triplet:
                tddft_wav = self.get_exc_eigenvec_raw_mode_i(i, "T")
            else:
                tddft_wav = self.get_exc_eigenvec_raw_mode_i(i, "S")
            tddft_wav = tddft_wav.reshape((self.noccorbs_A, self.nvirtorbs_A))
            dip = np.array(self.read("Elec multipole ints OCCVIR", multipole))
            dip = dip.reshape((self.nvirtorbs_A, self.noccorbs_A))
            return np.sum(dip * tddft_wav.T * np.sqrt(2))
        else:
            if triplet:
                tddft_wav = self.get_exc_eigenvec_raw_mode_i(i, "T")
            else:
                tddft_wav = self.get_exc_eigenvec_raw_mode_i(i, "S")

            tddft_wav_A = tddft_wav[: self.noccorbs_A * self.nvirtorbs_A].reshape(
                (self.noccorbs_A, self.nvirtorbs_A)
            )
            dip_A = np.array(self.read("Elec multipole ints OCCVIR", multipole + "_A"))
            dip_A = dip_A.reshape((self.nvirtorbs_A, self.noccorbs_A))
            tot_A = np.sum(dip_A * tddft_wav_A.T)

            tddft_wav_B = tddft_wav[self.noccorbs_A * self.nvirtorbs_A :].reshape(
                (self.noccorbs_B, self.nvirtorbs_B)
            )
            dip_B = np.array(self.read("Elec multipole ints OCCVIR", multipole + "_B"))
            dip_B = dip_B.reshape((self.nvirtorbs_B, self.noccorbs_B))
            tot_B = np.sum(dip_B * tddft_wav_B.T)
            return tot_A + tot_B

    def get_electric_dipole_moment(self, vel=False):
        """Return electric dipole moment."""
        if vel:
            vel_string = "dipole velocity "
        else:
            vel_string = "dipole "
        out = [
            [
                self.get_electric_multipole_moment(i, vel_string + x)
                for x in ["x", "y", "z"]
            ]
            for i in range(self.n_exc_S)
        ]
        out = np.array(out)
        return out

    def get_electric_quadrupole_moment(self, vel=False):
        """Return electric quadrupole moment."""
        if vel:
            vel_string = "quadrupole velocity "
        else:
            vel_string = "quadrupole "
        out = [
            [
                self.get_electric_multipole_moment(i, vel_string + "_".join(x))
                for x in non_decreasing_combinations(["x", "y", "z"], 2)
            ]
            for i in range(self.n_exc_S)
        ]
        out = np.array(out)
        out = list_to_symmetric_matrix(out)
        return out

    def get_electric_octupole_moment(self, vel=False):
        """Return electric octupole moment."""
        if vel:
            vel_string = "octupole velocity "
        else:
            vel_string = "octupole "
        out = [
            [
                self.get_electric_multipole_moment(i, vel_string + "_".join(x))
                for x in non_decreasing_combinations(["x", "y", "z"], 3)
            ]
            for i in range(self.n_exc_S)
        ]
        out = np.array(out)
        out = list_to_symmetric_rank3_tensor(out)
        return out

    @cached_property
    def exc_eigenvec(self):
        """Process exc eigenvec."""
        exc_eigenvec = np.zeros(
            (
                self.n_exc,
                self.noccorbs_A + self.noccorbs_B,
                self.nvirtorbs_A + self.nvirtorbs_B,
            )
        )
        for i in range(self.n_exc):
            exc_eigenvec[
                i, : self.noccorbs_A, : self.nvirtorbs_A
            ] = self.get_exc_eigenvec_raw_i(i)[
                : self.noccorbs_A * self.nvirtorbs_A
            ].reshape(
                (self.noccorbs_A, self.nvirtorbs_A)
            )

            try:
                exc_eigenvec[
                    i, self.noccorbs_A :, self.nvirtorbs_A :
                ] = self.get_exc_eigenvec_raw_i(i)[
                    self.noccorbs_A * self.nvirtorbs_A :
                ].reshape(
                    (self.noccorbs_B, self.nvirtorbs_B)
                )
            except Exception as e:
                print(e)
                exc_eigenvec[i, self.noccorbs_A :, self.nvirtorbs_A :] = exc_eigenvec[
                    i, : self.noccorbs_A, : self.nvirtorbs_A
                ]
        return exc_eigenvec

    @property
    def exc_eigenvec_A(self):
        """Process exc eigenvec A."""
        return self.exc_eigenvec[:, : self.noccorbs_A, : self.nvirtorbs_A]

    @property
    def exc_eigenvec_B(self):
        """Process exc eigenvec B."""
        return self.exc_eigenvec[:, : self.noccorbs_A, : self.nvirtorbs_A]

    def get_exc_eigenvec_T_raw_i(self, i):
        """Return exc eigenvec T raw i."""
        return np.array(
            self.rkffile.read("Excitations ST A", "F Vectors" + f"{i+1}".rjust(5, " "))
        )

    @cached_property
    def exc_eigenvec_T(self):
        """Process exc eigenvec T."""
        exc_eigenvec = np.zeros(
            (
                self.n_exc,
                self.noccorbs_A + self.noccorbs_B,
                self.nvirtorbs_A + self.nvirtorbs_B,
            )
        )
        for i in range(self.n_exc):
            exc_eigenvec[
                i, : self.noccorbs_A, : self.nvirtorbs_A
            ] = self.get_exc_eigenvec_T_raw_i(i)[
                : self.noccorbs_A * self.nvirtorbs_A
            ].reshape(
                (self.noccorbs_A, self.nvirtorbs_A)
            )

            try:
                exc_eigenvec[
                    i, self.noccorbs_A :, self.nvirtorbs_A :
                ] = self.get_exc_eigenvec_T_raw_i(i)[
                    self.noccorbs_A * self.nvirtorbs_A :
                ].reshape(
                    (self.noccorbs_B, self.nvirtorbs_B)
                )
            except Exception as e:
                print(e)
                exc_eigenvec[i, self.noccorbs_A :, self.nvirtorbs_A :] = exc_eigenvec[
                    i, : self.noccorbs_A, : self.nvirtorbs_A
                ]
        return exc_eigenvec

    @property
    def exc_eigenvec_T_A(self):
        """Process exc eigenvec T A."""
        return self.exc_eigenvec_T[:, : self.noccorbs_A, : self.nvirtorbs_A]

    @property
    def exc_eigenvec_T_B(self):
        """Process exc eigenvec T B."""
        return self.exc_eigenvec_T[:, : self.noccorbs_A, : self.nvirtorbs_A]

    def __get_singlet_excitonic_properties__(self):
        """Handle get singlet excitonic properties internally."""
        print("Getting singlet properties")

        if not self.unrestricted:
            self.nNTOs = min(
                self.noccorbs_A + self.noccorbs_B, self.nvirtorbs_A + self.nvirtorbs_B
            )

            self.exc_eigenvec_A = np.zeros(
                (self.n_exc, self.noccorbs_A, self.nvirtorbs_A)
            )

            self.smatrix_A = np.zeros(
                (self.n_exc, min(self.noccorbs_A, self.nvirtorbs_A))
            )
            self.umatrix_A = np.zeros((self.n_exc, self.noccorbs_A, self.noccorbs_A))
            self.vhmatrix_A = np.zeros((self.n_exc, self.nvirtorbs_A, self.nvirtorbs_A))
            self.NTOs_A = {}
            for i in range(self.n_exc):
                self.exc_eigenvec_A[i, :, :] = np.array(
                    self.rkffile.read("Excitations SS A", f"eigenvector {i+1}")
                ).reshape((self.noccorbs_A, self.nvirtorbs_A))
                u, s, vh = np.linalg.svd(self.exc_eigenvec_A[i, :, :])

                self.umatrix_A[i, :, :] = u
                self.smatrix_A[i, :] = s
                self.vhmatrix_A[i, :, :] = vh
                self.NTOs_A[i] = np.linalg.svd(self.exc_eigenvec_A[i, :, :])
            self.exc_eigenvec_B = self.exc_eigenvec_A
            self.umatrix_B = self.umatrix_A
            self.smatrix_B = self.smatrix_A
            self.vhmatrix_B = self.vhmatrix_A
            self.NTOs_B = self.NTOs_A
            self.exc_eigenvec = np.zeros(
                (
                    self.n_exc,
                    self.noccorbs_A + self.noccorbs_A,
                    self.nvirtorbs_A + self.nvirtorbs_A,
                )
            )
            self.smatrix = np.zeros(
                (
                    self.n_exc,
                    min(
                        self.noccorbs_A + self.noccorbs_A,
                        self.nvirtorbs_A + self.nvirtorbs_A,
                    ),
                )
            )
            self.umatrix = np.zeros(
                (
                    self.n_exc,
                    self.noccorbs_A + self.noccorbs_A,
                    self.noccorbs_A + self.noccorbs_A,
                )
            )
            self.vhmatrix = np.zeros(
                (
                    self.n_exc,
                    self.nvirtorbs_A + self.nvirtorbs_A,
                    self.nvirtorbs_A + self.nvirtorbs_A,
                )
            )
            self.NTOs = {}
            for i in range(self.n_exc):
                self.exc_eigenvec[i, : self.noccorbs_A, : self.nvirtorbs_A] = np.array(
                    self.rkffile.read(
                        "Excitations SS A", "F Vectors" + f"{i+1}".rjust(5, " ")
                    )
                ).reshape((self.noccorbs_A, self.nvirtorbs_A))
                self.exc_eigenvec[i, self.noccorbs_A :, self.nvirtorbs_A :] = np.array(
                    self.rkffile.read(
                        "Excitations SS A", "F Vectors" + f"{i+1}".rjust(5, " ")
                    )
                ).reshape((self.noccorbs_A, self.nvirtorbs_A))
                u, s, vh = np.linalg.svd(self.exc_eigenvec[i, :, :])

                self.umatrix[i, :, :] = u
                self.smatrix[i, :] = s
                self.vhmatrix[i, :, :] = vh
                self.NTOs[i] = np.linalg.svd(self.exc_eigenvec[i, :, :])
            self.exc_eigenvec = self.exc_eigenvec * np.sqrt(0.5)
        else:
            self.exc_eigenvec_A = np.zeros(
                (self.n_exc, self.noccorbs_A, self.nvirtorbs_A)
            )
            self.smatrix_A = np.zeros(
                (self.n_exc, min(self.noccorbs_A, self.nvirtorbs_A))
            )
            self.umatrix_A = np.zeros((self.n_exc, self.noccorbs_A, self.noccorbs_A))
            self.vhmatrix_A = np.zeros((self.n_exc, self.nvirtorbs_A, self.nvirtorbs_A))
            self.NTOs_A = {}
            for i in range(self.n_exc):
                self.exc_eigenvec_A[i, :, :] = np.array(
                    self.rkffile.read(
                        "Excitations SS A", "F Vectors" + f"{i+1}".rjust(5, " ")
                    )
                )[: self.noccorbs_A * self.nvirtorbs_A].reshape(
                    (self.noccorbs_A, self.nvirtorbs_A)
                )
                u, s, vh = np.linalg.svd(self.exc_eigenvec_A[i, :, :])

                self.umatrix_A[i, :, :] = u
                self.smatrix_A[i, :] = s
                self.vhmatrix_A[i, :, :] = vh
                self.NTOs_A[i] = np.linalg.svd(self.exc_eigenvec_A[i, :, :])
            self.exc_eigenvec_B = np.zeros(
                (self.n_exc, self.noccorbs_B, self.nvirtorbs_B)
            )
            self.smatrix_B = np.zeros(
                (self.n_exc, min(self.noccorbs_B, self.nvirtorbs_B))
            )
            self.umatrix_B = np.zeros((self.n_exc, self.noccorbs_B, self.noccorbs_B))
            self.vhmatrix_B = np.zeros((self.n_exc, self.nvirtorbs_B, self.nvirtorbs_B))
            self.NTOs_B = {}
            for i in range(self.n_exc):
                self.exc_eigenvec_B[i, :, :] = np.array(
                    self.rkffile.read(
                        "Excitations SS A", "F Vectors" + f"{i+1}".rjust(5, " ")
                    )
                )[self.noccorbs_A * self.nvirtorbs_A :].reshape(
                    (self.noccorbs_B, self.nvirtorbs_B)
                )
                u, s, vh = np.linalg.svd(self.exc_eigenvec_B[i, :, :])

                self.umatrix_B[i, :, :] = u
                self.smatrix_B[i, :] = s
                self.vhmatrix_B[i, :, :] = vh
                self.NTOs_B[i] = np.linalg.svd(self.exc_eigenvec_B[i, :, :])
            self.nNTOs = min(
                self.noccorbs_A + self.noccorbs_B, self.nvirtorbs_A + self.nvirtorbs_B
            )
            self.exc_eigenvec = np.zeros(
                (
                    self.n_exc,
                    self.noccorbs_A + self.noccorbs_B,
                    self.nvirtorbs_A + self.nvirtorbs_B,
                )
            )
            self.smatrix = np.zeros((self.n_exc, self.nNTOs))
            self.umatrix = np.zeros(
                (
                    self.n_exc,
                    self.noccorbs_A + self.noccorbs_B,
                    self.noccorbs_A + self.noccorbs_B,
                )
            )
            self.vhmatrix = np.zeros(
                (
                    self.n_exc,
                    self.nvirtorbs_A + self.nvirtorbs_B,
                    self.nvirtorbs_A + self.nvirtorbs_B,
                )
            )
            self.NTOs = {}
            for i in range(self.n_exc):
                self.exc_eigenvec[i, : self.noccorbs_A, : self.nvirtorbs_A] = np.array(
                    self.rkffile.read(
                        "Excitations SS A", "F Vectors" + f"{i+1}".rjust(5, " ")
                    )
                )[: self.noccorbs_A * self.nvirtorbs_A].reshape(
                    (self.noccorbs_A, self.nvirtorbs_A)
                )
                self.exc_eigenvec[i, self.noccorbs_A :, self.nvirtorbs_A :] = np.array(
                    self.rkffile.read(
                        "Excitations SS A", "F Vectors" + f"{i+1}".rjust(5, " ")
                    )
                )[self.noccorbs_A * self.nvirtorbs_A :].reshape(
                    (self.noccorbs_B, self.nvirtorbs_B)
                )
                u, s, vh = np.linalg.svd(self.exc_eigenvec[i, :, :])

                self.umatrix[i, :, :] = u
                self.smatrix[i, :] = s
                self.vhmatrix[i, :, :] = vh
                self.NTOs[i] = np.linalg.svd(self.exc_eigenvec[i, :, :])

    def __get_triplet_excitonic_properties__(self):
        """Handle get triplet excitonic properties internally."""
        print("Getting triplet properties")

        if not self.unrestricted:
            self.nNTOs_T = min(
                self.noccorbs_A + self.noccorbs_B, self.nvirtorbs_A + self.nvirtorbs_B
            )

            self.exc_eigenvec_A_T = np.zeros(
                (self.n_exc_T, self.noccorbs_A, self.nvirtorbs_A)
            )
            self.smatrix_A_T = np.zeros(
                (self.n_exc_T, min(self.noccorbs_A, self.nvirtorbs_A))
            )
            self.umatrix_A_T = np.zeros(
                (self.n_exc_T, self.noccorbs_A, self.noccorbs_A)
            )
            self.vhmatrix_A_T = np.zeros(
                (self.n_exc_T, self.nvirtorbs_A, self.nvirtorbs_A)
            )
            self.NTOs_A_T = {}
            for i in range(self.n_exc_T):
                self.exc_eigenvec_A_T[i, :, :] = np.array(
                    self.rkffile.read("Excitations ST A", f"eigenvector {i+1}")
                ).reshape((self.noccorbs_A, self.nvirtorbs_A))
                u, s, vh = np.linalg.svd(self.exc_eigenvec_A_T[i, :, :])

                self.umatrix_A_T[i, :, :] = u
                self.smatrix_A_T[i, :] = s
                self.vhmatrix_A_T[i, :, :] = vh
                self.NTOs_A_T[i] = np.linalg.svd(self.exc_eigenvec_A[i, :, :])
            self.exc_eigenvec_B_T = self.exc_eigenvec_A_T
            self.umatrix_B_T = self.umatrix_A_T
            self.smatrix_B_T = self.smatrix_A_T
            self.vhmatrix_B_T = self.vhmatrix_A_T
            self.NTOs_B_T = self.NTOs_A_T
            self.exc_eigenvec_T = np.zeros(
                (
                    self.n_exc_T,
                    self.noccorbs_A + self.noccorbs_A,
                    self.nvirtorbs_A + self.nvirtorbs_A,
                )
            )
            self.smatrix_T = np.zeros(
                (
                    self.n_exc_T,
                    min(
                        self.noccorbs_A + self.noccorbs_A,
                        self.nvirtorbs_A + self.nvirtorbs_A,
                    ),
                )
            )
            self.umatrix_T = np.zeros(
                (
                    self.n_exc_T,
                    self.noccorbs_A + self.noccorbs_A,
                    self.noccorbs_A + self.noccorbs_A,
                )
            )
            self.vhmatrix_T = np.zeros(
                (
                    self.n_exc_T,
                    self.nvirtorbs_A + self.nvirtorbs_A,
                    self.nvirtorbs_A + self.nvirtorbs_A,
                )
            )
            self.NTOs_T = {}
            for i in range(self.n_exc_T):
                self.exc_eigenvec_T[i, : self.noccorbs_A, : self.nvirtorbs_A] = (
                    np.array(
                        self.rkffile.read(
                            "Excitations ST A", "F Vectors" + f"{i+1}".rjust(5, " ")
                        )
                    ).reshape((self.noccorbs_A, self.nvirtorbs_A))
                )
                self.exc_eigenvec_T[i, self.noccorbs_A :, self.nvirtorbs_A :] = (
                    np.array(
                        self.rkffile.read(
                            "Excitations ST A", "F Vectors" + f"{i+1}".rjust(5, " ")
                        )
                    ).reshape((self.noccorbs_A, self.nvirtorbs_A))
                )
                u, s, vh = np.linalg.svd(self.exc_eigenvec_T[i, :, :])

                self.umatrix_T[i, :, :] = u
                self.smatrix_T[i, :] = s
                self.vhmatrix_T[i, :, :] = vh
                self.NTOs_T[i] = np.linalg.svd(self.exc_eigenvec_T[i, :, :])
            self.exc_eigenvec_T = self.exc_eigenvec_T * np.sqrt(0.5)
        else:
            self.exc_eigenvec_A_T = np.zeros(
                (self.n_exc_T, self.noccorbs_A, self.nvirtorbs_A)
            )
            self.smatrix_A_T = np.zeros(
                (self.n_exc_T, min(self.noccorbs_A, self.nvirtorbs_A))
            )
            self.umatrix_A_T = np.zeros(
                (self.n_exc_T, self.noccorbs_A, self.noccorbs_A)
            )
            self.vhmatrix_A_T = np.zeros(
                (self.n_exc_T, self.nvirtorbs_A, self.nvirtorbs_A)
            )
            self.NTOs_A_T = {}
            for i in range(self.n_exc_T):
                self.exc_eigenvec_A_T[i, :, :] = np.array(
                    self.rkffile.read(
                        "Excitations ST A", "F Vectors" + f"{i+1}".rjust(5, " ")
                    )
                )[: self.noccorbs_A * self.nvirtorbs_A].reshape(
                    (self.noccorbs_A, self.nvirtorbs_A)
                )
                u, s, vh = np.linalg.svd(self.exc_eigenvec_A_T[i, :, :])

                self.umatrix_A_T[i, :, :] = u
                self.smatrix_A_T[i, :] = s
                self.vhmatrix_A_T[i, :, :] = vh
                self.NTOs_A_T[i] = np.linalg.svd(self.exc_eigenvec_A_T[i, :, :])
            self.exc_eigenvec_B_T = np.zeros(
                (self.n_exc_T, self.noccorbs_B, self.nvirtorbs_B)
            )
            self.smatrix_B_T = np.zeros(
                (self.n_exc_T, min(self.noccorbs_B, self.nvirtorbs_B))
            )
            self.umatrix_B_T = np.zeros(
                (self.n_exc_T, self.noccorbs_B, self.noccorbs_B)
            )
            self.vhmatrix_B_T = np.zeros(
                (self.n_exc_T, self.nvirtorbs_B, self.nvirtorbs_B)
            )
            self.NTOs_B_T = {}
            for i in range(self.n_exc_T):
                self.exc_eigenvec_B_T[i, :, :] = np.array(
                    self.rkffile.read(
                        "Excitations ST A", "F Vectors" + f"{i+1}".rjust(5, " ")
                    )
                )[self.noccorbs_A * self.nvirtorbs_A :].reshape(
                    (self.noccorbs_B, self.nvirtorbs_B)
                )
                u, s, vh = np.linalg.svd(self.exc_eigenvec_B_T[i, :, :])

                self.umatrix_B_T[i, :, :] = u
                self.smatrix_B_T[i, :] = s
                self.vhmatrix_B_T[i, :, :] = vh
                self.NTOs_B_T[i] = np.linalg.svd(self.exc_eigenvec_B_T[i, :, :])
            self.nNTOs_T = min(
                self.noccorbs_A + self.noccorbs_B, self.nvirtorbs_A + self.nvirtorbs_B
            )
            self.exc_eigenvec_T = np.zeros(
                (
                    self.n_exc_T,
                    self.noccorbs_A + self.noccorbs_B,
                    self.nvirtorbs_A + self.nvirtorbs_B,
                )
            )
            self.smatrix_T = np.zeros((self.n_exc_T, self.nNTOs))
            self.umatrix_T = np.zeros(
                (
                    self.n_exc_T,
                    self.noccorbs_A + self.noccorbs_B,
                    self.noccorbs_A + self.noccorbs_B,
                )
            )
            self.vhmatrix_T = np.zeros(
                (
                    self.n_exc_T,
                    self.nvirtorbs_A + self.nvirtorbs_B,
                    self.nvirtorbs_A + self.nvirtorbs_B,
                )
            )
            self.NTOs_T = {}
            for i in range(self.n_exc_T):
                self.exc_eigenvec_T[
                    i, : self.noccorbs_A, : self.nvirtorbs_A
                ] = np.array(
                    self.rkffile.read(
                        "Excitations ST A", "F Vectors" + f"{i+1}".rjust(5, " ")
                    )
                )[
                    : self.noccorbs_A * self.nvirtorbs_A
                ].reshape(
                    (self.noccorbs_A, self.nvirtorbs_A)
                )
                self.exc_eigenvec_T[
                    i, self.noccorbs_A :, self.nvirtorbs_A :
                ] = np.array(
                    self.rkffile.read(
                        "Excitations ST A", "F Vectors" + f"{i+1}".rjust(5, " ")
                    )
                )[
                    self.noccorbs_A * self.nvirtorbs_A :
                ].reshape(
                    (self.noccorbs_B, self.nvirtorbs_B)
                )
                u, s, vh = np.linalg.svd(self.exc_eigenvec_T[i, :, :])

                self.umatrix_T[i, :, :] = u
                self.smatrix_T[i, :] = s
                self.vhmatrix_T[i, :, :] = vh
                self.NTOs_T[i] = np.linalg.svd(self.exc_eigenvec_T[i, :, :])

    @cached_property
    def SO_mat_R(self):
        """Process SO mat R."""
        return np.array(self.rkffile.read("Excitations SO A", "SOmat-R"))

    @cached_property
    def SO_mat_I(self):
        """Process SO mat I."""
        return np.array(self.rkffile.read("Excitations SO A", "SOmat-I"))

    @property
    def mat_SO_R(self):
        """Process mat SO R."""
        mat_SO_R = np.zeros((self.n_exc_so, self.n_exc_so), dtype=np.float64)
        tmpmat = np.asarray(self.SO_mat_R)
        ind = np.tril_indices(self.n_exc_so)
        mat_SO_R[ind] = tmpmat
        mat_SO_R = mat_SO_R + mat_SO_R.T - np.eye(self.n_exc_so) * mat_SO_R
        return mat_SO_R

    @property
    def mat_SO_I(self):
        """Process mat SO I."""
        mat_SO_I = np.zeros((self.n_exc_so, self.n_exc_so), dtype=np.float64)
        tmpmat = np.asarray(self.SO_mat_I)
        ind = np.tril_indices(self.n_exc_so)
        mat_SO_I[ind] = tmpmat
        mat_SO_I = mat_SO_I + mat_SO_I.T - np.eye(self.n_exc_so) * mat_SO_I
        return mat_SO_I

    @property
    def mat_SO(self):
        """Process mat SO."""
        return self.mat_SO_R + 1.0j * self.mat_SO_I

    @property
    def eigensystem_SO(self):
        """Process eigensystem SO."""
        return np.linalg.eigh(self.mat_SO)

    @property
    def eigvals_SO(self):
        """Process eigvals SO."""
        eigv = self.eigensystem_SO[0]
        ind = np.argsort(np.real(eigv))
        return eigv[ind]

    @property
    def eigvecs_SO(self):
        """Process eigvecs SO."""
        try:
            eigvals, eigvecs = self.eigensystem_SO
            ind = np.argsort(np.real(eigvals))
        except:
            warnings.warn("""Could not get eigenvectors of SO matrix.""", stacklevel=2)
            return np.eye(self.n_exc_so)
        return eigvecs[:, ind]

    @property
    def so_eigvec_singlet(self):
        """Process so eigvec singlet."""
        eigvecs = self.eigvecs_SO

        eigvecssinglet = eigvecs[: self.n_exc, :]
        return eigvecssinglet

    @property
    def so_eigvec_triplet(self):
        """Process so eigvec triplet."""
        eigvecs = self.eigvecs_SO

        eigvecstriplet = eigvecs[self.n_exc : self.n_exc + 3 * self.n_exc_T, :]
        return eigvecstriplet

    def transpose_singlet(self, feat, sq=False):
        """Process transpose singlet."""
        eigvecs = self.so_eigvec_singlet
        if sq:
            eigvecs = np.abs(eigvecs) ** 2

        out = np.tensordot(eigvecs, feat, axes=(0, 0))

        return out

    def transpose_triplet(self, feat, sq=False):
        """Process transpose triplet."""
        eigvecs = self.so_eigvec_triplet
        if sq:
            eigvecs = np.abs(eigvecs) ** 2
        if feat.shape[0] * 3 == eigvecs.shape[0]:
            feat = np.repeat(feat, 3, axis=0)
        assert (
            feat.shape[0] == eigvecs.shape[0]
        ), "Feature and eigenvector shapes are not equal"

        out = np.tensordot(eigvecs, feat, axes=(0, 0))

        return out

    @property
    def E_exc_SO(self):
        """Process E exc SO."""
        out = self.eigvals_SO
        out = out - np.min(out)
        return out[1:]

    def __post_process_an_excitation_initialization__(self):
        """Handle post process an excitation initialization internally."""
        self.excitation_overlaps = np.zeros((self.n_exc, self.n_exc))
        self.excitation_couplings = np.zeros((self.n_exc, self.n_exc))
        self.excitation_overlaps_T = np.zeros((self.n_exc, self.n_exc))
        self.excitation_couplings_T = np.zeros((self.n_exc, self.n_exc))

    def post_process_all_excitation_coupling_S(self):
        """Process post process all excitation coupling S."""
        for i_exc1 in tqdm.tqdm(
            range(self.n_exc), desc="Singlet excitiation 1", position=0
        ):
            for i_exc2 in tqdm.tqdm(
                range(self.n_exc), desc="Singlet excitiation 2", position=1, leave=True
            ):
                self.post_process_single_excitation_coupling_S(i_exc1, i_exc2)
        self.save_pickle()
        return self.excitation_overlaps, self.excitation_couplings

    def post_process_single_excitation_coupling_S(self, i_exc1, i_exc2, force=False):
        """Process post process single excitation coupling S."""
        if (self.excitation_overlaps[i_exc1, i_exc2] == 0.0) or (force):

            (
                self.excitation_overlaps[i_exc1, i_exc2],
                self.excitation_couplings[i_exc1, i_exc2],
            ) = NTO_for_loop_double_excitation(
                self.noccorbs_A,
                self.noccorbs_B,
                self.nvirtorbs_A,
                self.nvirtorbs_B,
                self.nmos_A,
                self.nmos_B,
                self.exc_eigenvec,
                self.fock_matrix,
                self.ovl_matrix,
                i_exc1,
                i_exc2,
            )
            (
                self.excitation_overlaps[i_exc2, i_exc1],
                self.excitation_couplings[i_exc2, i_exc1],
            ) = (
                self.excitation_overlaps[i_exc1, i_exc2],
                self.excitation_couplings[i_exc1, i_exc2],
            )
        return (
            self.excitation_overlaps[i_exc1, i_exc2],
            self.excitation_couplings[i_exc1, i_exc2],
        )

    def post_process_all_excitation_coupling_T_mp(self):
        """Process post process all excitation coupling T mp."""
        from mpire import WorkerPool

        results = {}
        with WorkerPool(n_jobs=4) as pool:
            for i_exc1 in tqdm.tqdm(
                range(self.n_exc_T), desc="Triplet excitiation 1", position=0
            ):
                for i_exc2 in tqdm.tqdm(
                    range(self.n_exc_T),
                    desc="Triplet excitiation 2",
                    position=1,
                    leave=True,
                ):
                    if self.excitation_overlaps_T[i_exc1, i_exc2] == 0.0:
                        results[(i_exc1, i_exc2)] = pool.apply_async(
                            NTO_for_loop_double_excitation,
                            args=(
                                self.noccorbs_A,
                                self.noccorbs_B,
                                self.nvirtorbs_A,
                                self.nvirtorbs_B,
                                self.nmos_A,
                                self.nmos_B,
                                self.exc_eigenvec_T,
                                self.fock_matrix,
                                self.ovl_matrix,
                                i_exc1,
                                i_exc2,
                            ),
                        )
            for k, v in tqdm.tqdm(
                results.items(), desc="Results triplets", position=0, total=len(results)
            ):
                i_exc1 = k[0]
                i_exc2 = k[1]
                (
                    self.excitation_overlaps_T[i_exc1, i_exc2],
                    self.excitation_couplings_T[i_exc1, i_exc2],
                ) = v.get(timeout=None)
                (
                    self.excitation_overlaps_T[i_exc2, i_exc1],
                    self.excitation_couplings_T[i_exc2, i_exc1],
                ) = (
                    self.excitation_overlaps_T[i_exc1, i_exc2],
                    self.excitation_couplings_T[i_exc1, i_exc2],
                )

            self.save_pickle()
        return self.excitation_overlaps_T, self.excitation_couplings_T

    def post_process_all_excitation_coupling_T(self):
        """Process post process all excitation coupling T."""
        for i_exc1 in tqdm.tqdm(
            range(self.n_exc_T), desc="Triplet excitiation 1", position=0
        ):
            for i_exc2 in tqdm.tqdm(
                range(self.n_exc_T),
                desc="Triplet excitiation 2",
                position=1,
                leave=True,
            ):
                self.post_process_single_excitation_coupling_T(i_exc1, i_exc2)
        self.save_pickle()
        return self.excitation_overlaps_T, self.excitation_couplings_T

    def post_process_single_excitation_coupling_T(self, i_exc1, i_exc2):
        """Process post process single excitation coupling T."""
        if self.excitation_overlaps_T[i_exc1, i_exc2] == 0.0:

            (
                self.excitation_overlaps_T[i_exc1, i_exc2],
                self.excitation_couplings_T[i_exc1, i_exc2],
            ) = NTO_for_loop_double_excitation(
                self.noccorbs_A,
                self.noccorbs_B,
                self.nvirtorbs_A,
                self.nvirtorbs_B,
                self.nmos_A,
                self.nmos_B,
                self.exc_eigenvec_T,
                self.fock_matrix,
                self.ovl_matrix,
                i_exc1,
                i_exc2,
            )
            (
                self.excitation_overlaps_T[i_exc2, i_exc1],
                self.excitation_couplings_T[i_exc2, i_exc1],
            ) = (
                self.excitation_overlaps_T[i_exc1, i_exc2],
                self.excitation_couplings_T[i_exc1, i_exc2],
            )
        return (
            self.excitation_overlaps_T[i_exc1, i_exc2],
            self.excitation_couplings_T[i_exc1, i_exc2],
        )


def NTO_for_loop_single(
    noccorbs_A,
    noccorbs_B,
    nvirtorbs_A,
    nvirtorbs_B,
    nmos_A,
    nmos_B,
    exc_eigenvec,
    sfo_fock_matrix,
    sfo_ovl_matrix,
):
    """Process NTO for loop single."""
    n_exc = exc_eigenvec.shape[0]
    excitation_overlaps = np.zeros((n_exc, n_exc))
    excitation_couplings = np.zeros((n_exc, n_exc))
    for i_exc1 in range(n_exc):
        for i_exc2 in tqdm.tqdm(range(n_exc)):
            (
                excitation_overlaps[i_exc1, i_exc2],
                excitation_couplings[i_exc1, i_exc2],
            ) = NTO_for_loop_double_excitation(
                noccorbs_A,
                noccorbs_B,
                nvirtorbs_A,
                nvirtorbs_B,
                nmos_A,
                nmos_B,
                exc_eigenvec,
                sfo_fock_matrix,
                sfo_ovl_matrix,
                i_exc1,
                i_exc2,
                excitation_overlaps,
                excitation_couplings,
            )
    return excitation_overlaps, excitation_couplings


def NTO_for_loop_double_excitation(
    noccorbs_A,
    noccorbs_B,
    nvirtorbs_A,
    nvirtorbs_B,
    nmos_A,
    nmos_B,
    exc_eigenvec,
    sfo_fock_matrix,
    sfo_ovl_matrix,
    i_exc1,
    i_exc2,
    *_unused,
):
    """Compute coupling for two excitations from shared-molecule data."""
    return NTO_for_loop_two_excitations(
        noccorbs_A,
        noccorbs_B,
        nvirtorbs_A,
        nvirtorbs_B,
        noccorbs_A,
        noccorbs_B,
        nvirtorbs_A,
        nvirtorbs_B,
        nmos_A,
        nmos_B,
        nmos_A,
        nmos_B,
        exc_eigenvec,
        exc_eigenvec,
        sfo_fock_matrix,
        sfo_ovl_matrix,
        i_exc1,
        i_exc2,
    )


class Dimer_RKF_Analyzer:
    def __init__(
        self,
        rkfpath: str,
        job1: Single_RKF_Reader,
        job2: Single_RKF_Reader,
        load: bool = True,
        m_CT: int = 0,
        name: str = "",
    ):
        """Initialize the object."""
        self.rkf1 = job1
        self.rkf2 = job2

        self.m_CT = m_CT
        self.rkfpath = rkfpath
        self.pickle_path = rkfpath + f"{name}" + ".pickle"

        assert os.path.isfile(self.rkfpath)
        loaded = False

        if load:
            try:
                self.load_pickle()
                loaded = True
            except:
                pass

        if not loaded:
            KFFile = require_plams()[3]
            self.rkffile = KFFile(rkfpath)

            self.name = os.path.basename(os.path.dirname(rkfpath))

            self.__obtain_results__()
            self.save_pickle()

    def load_pickle(self):
        """Process load pickle."""
        with open(self.pickle_path, "rb") as f:
            self.__dict__.update(pickle.load(f))

    def save_pickle(self):
        """Process save pickle."""
        with open(self.pickle_path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def __obtain_results__(self):
        """Handle obtain results internally."""
        for func in sorted(
            [method for method in dir(self) if method.startswith("__get_")]
        ):
            try:
                getattr(self, func)()
            except Exception as e:
                warnings.warn(
                    f"Did not succeed with function {func}.\r\nError obtained was {e}",
                    stacklevel=2,
                )

        for func in sorted(
            [method for method in dir(self) if method.startswith("__pre_process_")]
        ):
            try:
                getattr(self, func)()
            except Exception as e:
                warnings.warn(
                    f"Did not succeed with function {func}.\r\nError obtained was {e}",
                    stacklevel=2,
                )

        for func in sorted(
            [method for method in dir(self) if method.startswith("__post_process_")]
        ):
            try:
                getattr(self, func)()
            except Exception as e:
                warnings.warn(
                    f"Did not succeed with function {func}.\r\nError obtained was {e}",
                    stacklevel=2,
                )

    def __get_mol__(self):
        """Handle get mol internally."""
        self.mol = get_molecule(self.rkffile)
        self.individual_molecules = self.mol.separate()
        self.translated_mol = self.mol.copy()
        self.translated_mol.translate(
            -1 * np.array(self.translated_mol.get_center_of_mass())
        )

    def __get_basic_props__(self):
        """Handle get basic props internally."""
        print("Gettig basic properties")
        self.electrons_tot = int(self.rkffile.read("General", "electrons"))

        self.occs_A = self.rkffile.read("A", "froc_A")
        self.electrons_A = int(sum(self.occs_A))
        self.orb_ens_A = self.rkffile.read("A", "eps_A")
        self.nmos_A = self.rkffile.read("A", "nmo_A")

        self.noccorbs_A = sum(1 if occ > 0.5 else 0 for occ in self.occs_A)
        self.nvirtorbs_A = sum(1 if occ < 0.5 else 0 for occ in self.occs_A)

        self.overlap = self.rkffile.read("A", "S-CoreSFO")

        self.unrestricted = False

        self.occs_B = self.occs_A
        self.electrons_B = self.electrons_A
        self.orb_ens_B = self.orb_ens_A
        self.nmos_B = self.nmos_A
        self.noccorbs_B = self.noccorbs_A
        self.nvirtorbs_B = self.nvirtorbs_A

        try:
            self.occs_B = self.rkffile.read("A", "froc_B")
            self.electrons_B = int(sum(self.occs_B))
            self.orb_ens_B = self.rkffile.read("A", "eps_B")
            self.nmos_B = self.rkffile.read("A", "nmo_B")

            self.noccorbs_B = sum(1 if occ > 0.5 else 0 for occ in self.occs_B)
            self.nvirtorbs_B = sum(1 if occ < 0.5 else 0 for occ in self.occs_B)

            self.unrestricted = True
            print(f"{self.name} seems to be an unrestricted calculation.")
        except Exception as e:
            print(f"{self.name} seems to be a restricted calculation.")
            print(f"IF NOT, then the error was: {e}")

    @property
    def ndimer_orbs(self):
        """Process ndimer orbs."""
        return self.job1.norbs_tot + self.job2.norbs_tot

    def __get_SFOorb_energies__(self):
        """Handle get SFOorb energies internally."""
        self.orb_enSFO_dimer = self.rkffile.read("SFOs", "energy")

    def __get_sfo_basis__(self):
        """Handle get sfo basis internally."""
        self.bas_A = np.array(self.rkffile.read("A", "Eig-CoreSFO_A")).reshape(
            (self.nmos_A, self.nmos_A)
        )
        if self.unrestricted:
            self.bas_B = np.array(self.rkffile.read("A", "Eig-CoreSFO_B")).reshape(
                (self.nmos_B, self.nmos_B)
            )
        else:
            self.bas_B = self.bas_A
        self.bas = np.zeros((self.nmos_A + self.nmos_B, self.nmos_A + self.nmos_B))
        self.bas[: self.nmos_A, : self.nmos_A] = self.bas_A
        self.bas[self.nmos_B :, self.nmos_B :] = self.bas_B

    def __get_ovl_matrix__(self):
        """Handle get ovl matrix internally."""
        self.sfo_ovl_matrix_A = np.zeros((self.nmos_A, self.nmos_A))
        ovl_list = self.rkffile.read("SFO_Overlap", "A")
        indices = np.tril_indices(self.nmos_A)
        self.sfo_ovl_matrix_A[indices] = ovl_list
        dia = np.diag(self.sfo_ovl_matrix_A)
        self.sfo_ovl_matrix_A = (
            self.sfo_ovl_matrix_A + self.sfo_ovl_matrix_A.T - np.diag(dia)
        )

        try:

            self.sfo_ovl_matrix_B = np.zeros((self.nmos_B, self.nmos_B))
            ovl_list = self.rkffile.read("SFO_Overlap_B", "A")
            indices = np.tril_indices(self.nmos_B)

            self.sfo_ovl_matrix_B[indices] = ovl_list
            dia = np.diag(self.sfo_ovl_matrix_B)
            self.sfo_ovl_matrix_B = (
                self.sfo_ovl_matrix_B + self.sfo_ovl_matrix_B.T - np.diag(dia)
            )

        except:
            self.sfo_ovl_matrix_B = self.sfo_ovl_matrix_A
        self.sfo_ovl_matrix = np.zeros(
            (self.nmos_A + self.nmos_B, self.nmos_A + self.nmos_B)
        )
        self.sfo_ovl_matrix[: self.nmos_A, : self.nmos_A] = self.sfo_ovl_matrix_A
        self.sfo_ovl_matrix[self.nmos_B :, self.nmos_B :] = self.sfo_ovl_matrix_B

    def __get_fock_matrix__(self):
        """Handle get fock matrix internally."""
        if not self.unrestricted:
            self.sfo_fock_matrix_A = np.zeros((self.nmos_A, self.nmos_A))
            fock_list = self.rkffile.read("SFO_Fock", "A")
            indices = np.tril_indices(self.nmos_A)
            self.sfo_fock_matrix_A[indices] = fock_list
            dia = np.diag(self.sfo_fock_matrix_A)
            self.sfo_fock_matrix_A = (
                self.sfo_fock_matrix_A + self.sfo_fock_matrix_A.T - np.diag(dia)
            )

            self.sfo_fock_matrix_B = self.sfo_fock_matrix_A
            self.sfo_fock_matrix = np.zeros(
                (self.nmos_A + self.nmos_B, self.nmos_A + self.nmos_B)
            )
            self.sfo_fock_matrix[: self.nmos_A, : self.nmos_A] = self.sfo_fock_matrix_A
            self.sfo_fock_matrix[self.nmos_B :, self.nmos_B :] = self.sfo_fock_matrix_B
        else:
            self.sfo_fock_matrix = np.zeros(
                (self.nmos_A + self.nmos_B, self.nmos_A + self.nmos_B)
            )
            tmp_fock_matrix = np.zeros((self.nmos_A, self.nmos_A))
            fock_list = self.rkffile.read("SFO_Fock_A", "A")
            indices = np.tril_indices(self.nmos_A)
            tmp_fock_matrix[indices] = fock_list
            dia = np.diag(tmp_fock_matrix)
            self.sfo_fock_matrix[: self.nmos_A, : self.nmos_A] = (
                tmp_fock_matrix + tmp_fock_matrix.T - np.diag(dia)
            )
            self.sfo_fock_matrix_A = tmp_fock_matrix + tmp_fock_matrix.T - np.diag(dia)

            tmp_fock_matrix = np.zeros((self.nmos_B, self.nmos_B))
            fock_list = self.rkffile.read("SFO_Fock_B", "A")
            indices = np.tril_indices(self.nmos_B)

            tmp_fock_matrix[indices] = fock_list
            dia = np.diag(tmp_fock_matrix)
            self.sfo_fock_matrix[self.nmos_A :, self.nmos_A :] = (
                tmp_fock_matrix + tmp_fock_matrix.T - np.diag(dia)
            )
            self.sfo_fock_matrix_B = tmp_fock_matrix + tmp_fock_matrix.T - np.diag(dia)

    def __post_process_NTOs__(self):
        """Handle post process NTOs internally."""
        warnings.warn("Processing NTO's not implemented", stacklevel=2)

    def __get_transfer_integral_data__(self):
        """Handle get transfer integral data internally."""
        self.TI_data = {}
        if not self.unrestricted:
            for k in self.rkffile.get_skeleton()["TransferIntegrals"]:
                self.TI_data[k] = self.rkffile.read("TransferIntegrals", k)
        else:
            for k in self.rkffile.get_skeleton()["TransferIntegrals_A"]:
                self.TI_data[k + "_A"] = self.rkffile.read("TransferIntegrals_A", k)
            for k in self.rkffile.get_skeleton()["TransferIntegrals_B"]:
                self.TI_data[k + "_B"] = self.rkffile.read("TransferIntegrals_B", k)

    def __post_process_charge_transport__(self):
        """Handle post process charge transport internally."""
        warnings.warn(
            "Post processing of charge transport might go wrong for combined unrestriced and restricted calculation",
            stacklevel=2,
        )
        self.jeff_matrix = np.empty(
            (self.nmos_A + self.nmos_B, self.nmos_A + self.nmos_B)
        )
        for iorb1 in range(self.nmos_A):
            for iorb2 in range(self.nmos_A):
                SDA = self.sfo_ovl_matrix[iorb1, iorb2]
                JDA = self.sfo_fock_matrix[iorb1, iorb2]
                E1 = self.rkf1.get_orbital_energy(iorb1)
                E2 = self.rkf2.get_orbital_energy(iorb2)

                self.jeff_matrix[iorb1, iorb2] = Jeff(JDA, SDA, E1, E2)

        for iorb1 in range(self.nmos_A, self.nmos_B):
            for iorb2 in range(self.nmos_A, self.nmos_B):
                SDA = self.sfo_ovl_matrix[iorb1, iorb2]
                JDA = self.sfo_fock_matrix[iorb1, iorb2]
                E1 = self.rkf1.get_orbital_energy(iorb1)
                E2 = self.rkf2.get_orbital_energy(iorb2)

                self.jeff_matrix[iorb1, iorb2] = Jeff(JDA, SDA, E1, E2)

    def __post_process_an_excitation_initialization__(self):
        """Handle post process an excitation initialization internally."""
        self.excitation_overlaps = np.zeros((self.rkf1.n_exc, self.rkf2.n_exc))
        self.excitation_couplings = np.zeros((self.rkf1.n_exc, self.rkf2.n_exc))
        self.excitation_overlaps_Baumeier = np.zeros(
            (self.rkf1.n_exc, self.rkf2.n_exc, 2 + 2 * self.m_CT, 2 + 2 * self.m_CT)
        )
        self.excitation_couplings_Baumeier = np.zeros(
            (self.rkf1.n_exc, self.rkf2.n_exc, 2 + 2 * self.m_CT, 2 + 2 * self.m_CT)
        )
        self.excitation_overlaps_T = np.zeros((self.rkf1.n_exc_T, self.rkf2.n_exc_T))
        self.excitation_couplings_T = np.zeros((self.rkf1.n_exc_T, self.rkf2.n_exc_T))
        self.excitation_overlaps_Baumeier_T = np.zeros(
            (self.rkf1.n_exc, self.rkf2.n_exc, 2 + 2 * self.m_CT, 2 + 2 * self.m_CT)
        )
        self.excitation_couplings_Baumeier_T = np.zeros(
            (self.rkf1.n_exc, self.rkf2.n_exc, 2 + 2 * self.m_CT, 2 + 2 * self.m_CT)
        )

    def post_process_all_excitation_coupling_SS(self):
        """Process post process all excitation coupling SS."""
        for i_exc1 in tqdm.tqdm(
            range(self.rkf1.n_exc), desc="Singlet excitiation 1", position=0
        ):
            for i_exc2 in tqdm.tqdm(
                range(self.rkf2.n_exc),
                desc="Singlet excitiation 2",
                position=1,
                leave=False,
            ):
                self.post_process_single_excitation_coupling_SS(i_exc1, i_exc2)
        self.save_pickle()
        return self.excitation_overlaps, self.excitation_couplings

    def post_process_single_excitation_coupling_SS(self, i_exc1, i_exc2):
        """Process post process single excitation coupling SS."""
        if self.excitation_overlaps[i_exc1, i_exc2] == 0.0:

            (
                self.excitation_overlaps[i_exc1, i_exc2],
                self.excitation_couplings[i_exc1, i_exc2],
            ) = NTO_for_loop_two_excitations(
                self.rkf1.noccorbs_A,
                self.rkf1.noccorbs_B,
                self.rkf1.nvirtorbs_A,
                self.rkf1.nvirtorbs_B,
                self.rkf2.noccorbs_A,
                self.rkf2.noccorbs_B,
                self.rkf2.nvirtorbs_A,
                self.rkf2.nvirtorbs_B,
                self.rkf1.nmos_A,
                self.rkf1.nmos_B,
                self.rkf2.nmos_A,
                self.rkf2.nmos_B,
                self.rkf1.exc_eigenvec,
                self.rkf2.exc_eigenvec,
                self.sfo_fock_matrix,
                self.sfo_ovl_matrix,
                i_exc1,
                i_exc2,
            )
        return (
            self.excitation_overlaps[i_exc1, i_exc2],
            self.excitation_couplings[i_exc1, i_exc2],
        )

    def post_process_all_excitation_coupling_TT(self):
        """Process post process all excitation coupling TT."""
        for i_exc1 in tqdm.tqdm(
            range(self.rkf1.n_exc_T), desc="Triplet excitiation 1", position=0
        ):
            for i_exc2 in tqdm.tqdm(
                range(self.rkf2.n_exc_T),
                desc="Triplet excitiation 2",
                position=1,
                leave=False,
            ):
                self.post_process_single_excitation_coupling_TT(i_exc1, i_exc2)
        self.save_pickle()
        return self.excitation_overlaps_T, self.excitation_couplings_T

    def post_process_single_excitation_coupling_Baumeier(self, i_exc1, i_exc2):
        """Process post process single excitation coupling Baumeier."""
        if self.excitation_overlaps_Baumeier[i_exc1, i_exc2, 0, 0] == 0.0:

            (
                self.excitation_overlaps_Baumeier[i_exc1, i_exc2, :, :],
                self.excitation_couplings_Baumeier[i_exc1, i_exc2, :, :],
            ) = Setup_for_loop_DIPRO(
                self.rkf1.noccorbs_A,
                self.rkf1.noccorbs_B,
                self.rkf1.nvirtorbs_A,
                self.rkf1.nvirtorbs_B,
                self.rkf2.noccorbs_A,
                self.rkf2.noccorbs_B,
                self.rkf2.nvirtorbs_A,
                self.rkf2.nvirtorbs_B,
                self.rkf1.nmos_A,
                self.rkf1.nmos_B,
                self.rkf2.nmos_A,
                self.rkf2.nmos_B,
                self.rkf1.exc_eigenvec,
                self.rkf2.exc_eigenvec,
                self.sfo_fock_matrix,
                self.sfo_ovl_matrix,
                i_exc1,
                i_exc2,
                self.m_CT,
            )
            print(
                self.excitation_overlaps_Baumeier[i_exc1, i_exc2],
                self.excitation_couplings_Baumeier[i_exc1, i_exc2],
            )

        return (
            self.excitation_overlaps_Baumeier[i_exc1, i_exc2],
            self.excitation_couplings_Baumeier[i_exc1, i_exc2],
        )

    def post_process_single_excitation_coupling_Baumeier_T(self, i_exc1, i_exc2):
        """Process post process single excitation coupling Baumeier T."""
        if self.excitation_overlaps_Baumeier_T[i_exc1, i_exc2, 0, 0] == 0.0:

            (
                self.excitation_overlaps_Baumeier_T[i_exc1, i_exc2, :, :],
                self.excitation_couplings_Baumeier_T[i_exc1, i_exc2, :, :],
            ) = Setup_for_loop_DIPRO(
                self.rkf1.noccorbs_A,
                self.rkf1.noccorbs_B,
                self.rkf1.nvirtorbs_A,
                self.rkf1.nvirtorbs_B,
                self.rkf2.noccorbs_A,
                self.rkf2.noccorbs_B,
                self.rkf2.nvirtorbs_A,
                self.rkf2.nvirtorbs_B,
                self.rkf1.nmos_A,
                self.rkf1.nmos_B,
                self.rkf2.nmos_A,
                self.rkf2.nmos_B,
                self.rkf1.exc_eigenvec_T,
                self.rkf2.exc_eigenvec_T,
                self.sfo_fock_matrix,
                self.sfo_ovl_matrix,
                i_exc1,
                i_exc2,
                self.m_CT,
            )
            print(
                self.excitation_overlaps_Baumeier_T[i_exc1, i_exc2],
                self.excitation_couplings_Baumeier_T[i_exc1, i_exc2],
            )

        return (
            self.excitation_overlaps_Baumeier_T[i_exc1, i_exc2],
            self.excitation_couplings_Baumeier_T[i_exc1, i_exc2],
        )

    def post_process_single_excitation_coupling_TT(self, i_exc1, i_exc2):
        """Process post process single excitation coupling TT."""
        if self.excitation_overlaps_T[i_exc1, i_exc2] == 0.0:

            (
                self.excitation_overlaps_T[i_exc1, i_exc2],
                self.excitation_couplings_T[i_exc1, i_exc2],
            ) = NTO_for_loop_two_excitations(
                self.rkf1.noccorbs_A,
                self.rkf1.noccorbs_B,
                self.rkf1.nvirtorbs_A,
                self.rkf1.nvirtorbs_B,
                self.rkf2.noccorbs_A,
                self.rkf2.noccorbs_B,
                self.rkf2.nvirtorbs_A,
                self.rkf2.nvirtorbs_B,
                self.rkf1.nmos_A,
                self.rkf1.nmos_B,
                self.rkf2.nmos_A,
                self.rkf2.nmos_B,
                self.rkf1.exc_eigenvec_T,
                self.rkf2.exc_eigenvec_T,
                self.sfo_fock_matrix,
                self.sfo_ovl_matrix,
                i_exc1,
                i_exc2,
            )

        return (
            self.excitation_overlaps_T[i_exc1, i_exc2],
            self.excitation_couplings_T[i_exc1, i_exc2],
        )

    def post_process_all_excitation_coupling_SO(self, rvec):
        """Process post process all excitation coupling SO."""
        assert rvec.shape == tuple([3])
        norm_rvec = rvec / (np.sum(rvec * rvec, axis=0))
        norm_tdm1 = self.rkf1.transition_dipole_moments / (
            np.sum(
                self.rkf1.transition_dipole_moments
                * self.rkf1.transition_dipole_moments,
                axis=1,
                keepdims=True,
            )
        )
        norm_tdm2 = self.rkf2.transition_dipole_moments / (
            np.sum(
                self.rkf2.transition_dipole_moments
                * self.rkf2.transition_dipole_moments,
                axis=1,
                keepdims=True,
            )
        )

        self.post_process_all_excitation_coupling_SS()
        self.post_process_all_excitation_coupling_TT()

        try:
            exc_en_so1, exc_eigenvec1 = self.rkf1.eigvals_SO, self.rkf1.eigvecs_SO
        except:
            try:
                exc_eigenvec1 = np.eye(self.rkf1.n_exc + 3 * self.rkf1.n_exc_T + 1)
            except:
                exc_eigenvec1 = np.eye(self.rkf1.n_exc + 3 * self.rkf1.n_exc + 1)
        n_exc1 = exc_eigenvec1.shape[1]

        try:
            exc_en_so2, exc_eigenvec2 = self.rkf2.eigvals_SO, self.rkf2.eigvecs_SO
        except:
            try:
                exc_eigenvec2 = np.eye(self.rkf2.n_exc + 3 * self.rkf2.n_exc_T + 1)
            except:
                exc_eigenvec2 = np.eye(self.rkf2.n_exc + 3 * self.rkf2.n_exc + 1)
        n_exc2 = exc_eigenvec2.shape[1]

        self.excitation_overlaps_SO, self.excitation_couplings_SO = np.zeros(
            (n_exc1, n_exc2)
        ), np.zeros((n_exc1, n_exc2))
        self.excitation_tdm_SO, self.excitation_coeffs_SO = np.zeros(
            (n_exc1, n_exc2)
        ), np.zeros((n_exc1, n_exc2))

        for i_exc1 in tqdm.tqdm(range(n_exc1), desc="SO excitiation 1", position=0):
            for i_exc2 in tqdm.tqdm(
                range(n_exc2), desc="SO excitiation 2", position=1, leave=False
            ):
                for i_singlet1 in range(self.rkf1.n_exc):
                    for i_singlet2 in range(self.rkf2.n_exc):
                        coeff = exc_eigenvec1[i_singlet1, i_exc1] * np.conjugate(
                            exc_eigenvec2[i_singlet2, i_exc2]
                        )
                        overlap = self.excitation_overlaps[i_singlet1, i_singlet2]
                        coupling = self.excitation_couplings[i_singlet1, i_singlet2]
                        tdm = np.sqrt(
                            np.sum(
                                self.rkf1.transition_dipole_moments[i_singlet1, :]
                                * self.rkf1.transition_dipole_moments[i_singlet1, :]
                            )
                        ) * np.sqrt(
                            np.sum(
                                self.rkf2.transition_dipole_moments[i_singlet2, :]
                                * self.rkf2.transition_dipole_moments[i_singlet2, :]
                            )
                        )
                        kappa = np.dot(
                            norm_tdm1[i_singlet1], norm_tdm2[i_singlet2]
                        ) - 3 * np.dot(norm_tdm1[i_singlet1], norm_rvec) * np.dot(
                            norm_rvec, norm_tdm2[i_singlet2]
                        )

                        self.excitation_overlaps_SO[i_exc1, i_exc2] += coeff * overlap
                        self.excitation_couplings_SO[i_exc1, i_exc2] += coeff * coupling
                        self.excitation_tdm_SO[i_exc1, i_exc2] += coeff * tdm * kappa
                        self.excitation_coeffs_SO[i_exc1, i_exc2] += coeff

                for i_triplet1 in range(3 * self.rkf1.n_exc):
                    for i_triplet2 in range(3 * self.rkf2.n_exc):
                        i_level1 = i_triplet1 // 3
                        i_level2 = i_triplet2 // 3

                        coeff = exc_eigenvec1[
                            self.rkf1.n_exc + i_triplet1, i_exc1
                        ] * np.conjugate(
                            exc_eigenvec2[self.rkf2.n_exc + i_triplet2, i_exc2]
                        )
                        overlap = self.excitation_overlaps_T[i_level1, i_level2]
                        coupling = self.excitation_couplings_T[i_level1, i_level2]

                        self.excitation_overlaps_SO[i_exc1, i_exc2] += coeff * overlap
                        self.excitation_couplings_SO[i_exc1, i_exc2] += coeff * coupling
                        self.excitation_coeffs_SO[i_exc1, i_exc2] += coeff

        return (
            self.excitation_overlaps_SO,
            self.excitation_couplings_SO,
            self.excitation_tdm_SO,
        )


def Setup_for_loop_DIPRO(
    noccorbs_A1,
    noccorbs_B1,
    nvirtorbs_A1,
    nvirtorbs_B1,
    noccorbs_A2,
    noccorbs_B2,
    nvirtorbs_A2,
    nvirtorbs_B2,
    nmos_A1,
    nmos_B1,
    nmos_A2,
    nmos_B2,
    exc_eigenvec1,
    exc_eigenvec2,
    sfo_fock_matrix,
    sfo_ovl_matrix,
    i_exc1,
    i_exc2,
    m_CT,
):
    """Process Setup for loop DIPRO."""
    n_CT = 2 * m_CT * m_CT
    hamilt = np.zeros((2 + n_CT, 2 + n_CT))
    ovlps = np.zeros((2 + n_CT, 2 + n_CT))
    for i_orbh1 in range(noccorbs_A1 + noccorbs_B1):
        for i_orbe1 in range(nvirtorbs_A1 + nvirtorbs_B1):
            if i_orbe1 > nvirtorbs_A1:
                j_orbe1proc = i_orbe1 + nmos_A1 + nmos_A2 + noccorbs_B1 - nvirtorbs_A1
            else:
                j_orbe1proc = i_orbe1 + noccorbs_A1
            if i_orbh1 > noccorbs_A1:
                j_orbh1proc = i_orbh1 + nmos_A1 + nmos_A2 - noccorbs_A1
            else:
                j_orbh1proc = i_orbh1
            coeffs = (
                exc_eigenvec1[i_exc1, i_orbh1, i_orbe1]
                * exc_eigenvec1[i_exc1, i_orbh1, i_orbe1]
            )
            overlap = sfo_ovl_matrix[j_orbh1proc, j_orbh1proc]
            overlap *= sfo_ovl_matrix[j_orbe1proc, j_orbe1proc]
            coupling = (
                (-1.0)
                * sfo_fock_matrix[j_orbh1proc, j_orbh1proc]
                * sfo_ovl_matrix[j_orbe1proc, j_orbe1proc]
            )
            coupling += (
                sfo_fock_matrix[j_orbe1proc, j_orbe1proc]
                * sfo_ovl_matrix[j_orbh1proc, j_orbh1proc]
            )
            hamilt[0, 0] += coupling * coeffs
            ovlps[0, 0] += overlap * coeffs

    for i_orbh2 in range(noccorbs_A2 + noccorbs_B2):
        for i_orbe2 in range(nvirtorbs_A2 + nvirtorbs_B2):
            if i_orbe2 > nvirtorbs_A2:
                j_orbe2proc = (
                    i_orbe2 + nmos_A1 + nmos_B1 + nmos_A2 + noccorbs_B2 - nvirtorbs_A2
                )
            else:
                j_orbe2proc = i_orbe2 + nmos_A1 + noccorbs_A2
            if i_orbh2 > noccorbs_A2:
                j_orbh2proc = i_orbh2 + nmos_A1 + nmos_B1 + nmos_A2 - noccorbs_A2
            else:
                j_orbh2proc = i_orbh2 + nmos_A1
            coeffs = (
                exc_eigenvec2[i_exc2, i_orbh2, i_orbe2]
                * exc_eigenvec2[i_exc2, i_orbh2, i_orbe2]
            )
            overlap = sfo_ovl_matrix[j_orbh2proc, j_orbh2proc]
            overlap *= sfo_ovl_matrix[j_orbe2proc, j_orbe2proc]
            coupling = (
                (-1.0)
                * sfo_fock_matrix[j_orbh2proc, j_orbh2proc]
                * sfo_ovl_matrix[j_orbe2proc, j_orbe2proc]
            )
            coupling += (
                sfo_fock_matrix[j_orbe2proc, j_orbe2proc]
                * sfo_ovl_matrix[j_orbh2proc, j_orbh2proc]
            )
            hamilt[1, 1] += coupling * coeffs
            ovlps[1, 1] += overlap * coeffs

    for i_orbh1 in range(noccorbs_A1 + noccorbs_B1):
        for i_orbh2 in range(noccorbs_A2 + noccorbs_B2):
            for i_orbe1 in range(nvirtorbs_A1 + nvirtorbs_B1):
                for i_orbe2 in range(nvirtorbs_A2 + nvirtorbs_B2):
                    coeffs = (
                        exc_eigenvec1[i_exc1, i_orbh1, i_orbe1]
                        * exc_eigenvec2[i_exc2, i_orbh2, i_orbe2]
                    )
                    if i_orbe1 > nvirtorbs_A1:
                        j_orbe1proc = (
                            i_orbe1 + nmos_A1 + nmos_A2 + noccorbs_B1 - nvirtorbs_A1
                        )
                    else:
                        j_orbe1proc = i_orbe1 + noccorbs_A1
                    if i_orbe2 > nvirtorbs_A2:
                        j_orbe2proc = (
                            i_orbe2
                            + nmos_A1
                            + nmos_B1
                            + nmos_A2
                            + noccorbs_B2
                            - nvirtorbs_A2
                        )
                    else:
                        j_orbe2proc = i_orbe2 + nmos_A1 + noccorbs_A2
                    if i_orbh1 > noccorbs_A1:
                        j_orbh1proc = i_orbh1 + nmos_A1 + nmos_A2 - noccorbs_A1
                    else:
                        j_orbh1proc = i_orbh1
                    if i_orbh2 > noccorbs_A2:
                        j_orbh2proc = (
                            i_orbh2 + nmos_A1 + nmos_B1 + nmos_A2 - noccorbs_A2
                        )
                    else:
                        j_orbh2proc = i_orbh2 + nmos_A1

                    overlap = sfo_ovl_matrix[j_orbh1proc, j_orbh2proc]
                    overlap *= sfo_ovl_matrix[j_orbe1proc, j_orbe2proc]

                    coupling = (
                        (-1.0)
                        * sfo_fock_matrix[j_orbh1proc, j_orbh2proc]
                        * sfo_ovl_matrix[j_orbe1proc, j_orbe2proc]
                    )
                    coupling += (
                        sfo_fock_matrix[j_orbe1proc, j_orbe2proc]
                        * sfo_ovl_matrix[j_orbh1proc, j_orbh2proc]
                    )

                    hamilt[0, 1] += coupling * coeffs
                    hamilt[1, 0] += coupling * coeffs

                    ovlps[0, 1] += overlap * coeffs
                    ovlps[1, 0] += overlap * coeffs
    return ovlps, hamilt


def NTO_for_loop_two_excitations(
    noccorbs_A1,
    noccorbs_B1,
    nvirtorbs_A1,
    nvirtorbs_B1,
    noccorbs_A2,
    noccorbs_B2,
    nvirtorbs_A2,
    nvirtorbs_B2,
    nmos_A1,
    nmos_B1,
    nmos_A2,
    nmos_B2,
    exc_eigenvec1,
    exc_eigenvec2,
    sfo_fock_matrix,
    sfo_ovl_matrix,
    i_exc1,
    i_exc2,
):
    """Process NTO for loop two excitations."""
    c = 0.0
    o = 0.0
    for i_orbh1 in range(noccorbs_A1 + noccorbs_B1):
        for i_orbh2 in range(noccorbs_A2 + noccorbs_B2):
            for i_orbe1 in range(nvirtorbs_A1 + nvirtorbs_B1):
                for i_orbe2 in range(nvirtorbs_A2 + nvirtorbs_B2):
                    coeffs = (
                        exc_eigenvec1[i_exc1, i_orbh1, i_orbe1]
                        * exc_eigenvec2[i_exc2, i_orbh2, i_orbe2]
                    )
                    if i_orbe1 >= nvirtorbs_A1:
                        j_orbe1 = (
                            i_orbe1 + nmos_A1 + nmos_A2 + noccorbs_B1 - nvirtorbs_A1
                        )
                    else:
                        j_orbe1 = i_orbe1 + noccorbs_A1
                    if i_orbe2 >= nvirtorbs_A2:
                        j_orbe2 = (
                            i_orbe2
                            + nmos_A1
                            + nmos_B1
                            + nmos_A2
                            + noccorbs_B2
                            - nvirtorbs_A2
                        )
                    else:
                        j_orbe2 = i_orbe2 + nmos_A1 + noccorbs_A2
                    if i_orbh1 >= noccorbs_A1:
                        j_orbh1 = i_orbh1 + nmos_A1 + nmos_A2 - noccorbs_A1
                    else:
                        j_orbh1 = i_orbh1
                    if i_orbh2 >= noccorbs_A2:
                        j_orbh2 = i_orbh2 + nmos_A1 + nmos_B1 + nmos_A2 - noccorbs_A2
                    else:
                        j_orbh2 = i_orbh2 + nmos_A1

                    overlap = sfo_ovl_matrix[j_orbh1, j_orbh2]
                    overlap *= sfo_ovl_matrix[j_orbe1, j_orbe2]

                    coupling = (
                        (-1.0)
                        * sfo_fock_matrix[j_orbh1, j_orbh2]
                        * sfo_ovl_matrix[j_orbe1, j_orbe2]
                    )
                    coupling += (
                        sfo_fock_matrix[j_orbe1, j_orbe2]
                        * sfo_ovl_matrix[j_orbh1, j_orbh2]
                    )

                    o += overlap * coeffs
                    c += coupling * coeffs
    return o, c
