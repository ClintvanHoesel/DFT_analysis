import re
import zlib
from collections import defaultdict
from functools import cache, cached_property

import numpy as np

from dft_utils.multipole_calculators import (
    list_to_symmetric_matrix_ORCA,
)
from dft_utils.units import Units

from .FileParser import FileParser


def _parse_text_to_nparr_tokens(text: str, n_r, n_c):
    """Handle parse text to nparr tokens internally."""
    A = np.zeros((n_r, n_c), float)

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        tokens = line.split()
        if not tokens:
            i += 1
            continue

        if all(tok.isdigit() for tok in tokens):
            cols = list(map(int, tokens))
            i += 1
            if not lines[i].strip():
                i += 1

            j = 0
            while i < len(lines) and lines[i].strip() and j < n_r:
                row_tokens = lines[i].split()
                row = int(row_tokens[0])
                values = row_tokens[1:]
                for m, val in enumerate(values):
                    A[row, cols[m]] = float(val)
                i += 1
                j += 1
            continue

        i += 1

    return A


def append_duplicates(lst):
    """Process append duplicates."""
    counts = defaultdict(int)
    result = []

    for item in lst:
        counts[item] += 1
        if counts[item] == 1:
            result.append(item)
        else:
            result.append(f"{item}_{counts[item]}")

    return result


class OutParser(FileParser):
    def __init__(self, path: str, level: int = 7):
        """Initialize the object."""
        super().__init__(path)

        self.level = level

    def parse(self):
        """Parse the supplied input."""
        return open(self.path).read()

    def _parse_text(self) -> str:
        """Handle parse text internally."""
        text = self.path.read_text()
        return text

    @cached_property
    def compressed_text(self) -> bytes:
        """Compress and cache the full text."""

        return zlib.compress(self._parse_text().encode("utf-8"), level=self.level)

    @property
    def text(self) -> str:
        """Access the original text, decompressing on the fly."""
        comp = getattr(self, "compressed_text", None)
        return zlib.decompress(comp).decode("utf-8")

    @property
    def ZPVE(self) -> float:
        """Process ZPVE."""
        pattern = re.compile(r"Non-thermal \(ZPE\) correction\s+([0-9]+\.[0-9]+)\s+Eh")
        m = pattern.search(self.text)
        return float(m.group(1))

    @property
    def ZPTE(self) -> float:
        """Process ZPTE."""
        pattern = re.compile(r"Total correction\s+([0-9]+\.[0-9]+)\s+Eh")
        m = pattern.search(self.text)
        return float(m.group(1))

    @property
    def output_dipole_moment(self) -> float:
        """Process output dipole moment."""
        pattern = re.compile(
            r"Total Dipole Moment *\: *(?P<dipx>[\-\d\.]+) *(?P<dipy>[\-\d\.]+) *(?P<dipz>[\-\d\.]+)"
        )
        m = pattern.search(self.text)
        return np.array(
            [float(m.group("dipx")), float(m.group("dipy")), float(m.group("dipz"))]
        )

    @property
    def n_excitons(self) -> str:
        """Access the number of excitons."""

        pattern = re.compile(
            r"Number of roots to be determined\s+\.\.\.\s+(?P<content>\d+)"
        )
        match = pattern.search(self.text)
        return int(match.group("content")) if match else 0

    @property
    def n_singlets(self):
        """Process n singlets."""
        return self.n_excitons

    @property
    def n_triplets(self):
        """Process n triplets."""
        return 3 * self.n_excitons

    @property
    def n_socexcitons(self):
        """Process n socexcitons."""
        return 4 * self.n_excitons + 1

    @cached_property
    def mat_SO(self) -> str:
        """Access the full SOC matrix text."""

        pattern = re.compile(
            r"The full SOC matrix: \n\s*Real part:\s*(?P<repart>[\s\deE\-\+\.]*)Image part:\s*(?P<impart>[\s\deE\-\+\.]*)",
            re.MULTILINE,
        )
        match = pattern.search(self.text)
        repart = match.group("repart")
        repart = _parse_text_to_nparr_tokens(
            repart, self.n_socexcitons, self.n_socexcitons
        )
        impart = match.group("impart")
        impart = _parse_text_to_nparr_tokens(
            impart, self.n_socexcitons, self.n_socexcitons
        )
        return repart + 1j * impart

    @cache
    def get_multipole_data(self, typ: str, n_c: int) -> np.ndarray:
        """Get excitonic data from the text via explicit loops."""

        pattern = re.compile(
            r"Transition " + typ + r" Moments *\n" r"(?P<content>[\d\sMult\>\.\-]*)",
            re.MULTILINE,
        )

        m = pattern.search(self.text)
        if not m:
            raise ValueError(f"No block for type={typ!r}")
        text = m.group("content")

        out = {}

        if "Mult" in text:
            pattern = re.compile(
                r"Mult (?P<m1>\d+) \> Mult (?P<m2>\d+)\s*" r"(?P<content>[\d\s\-\.]*)",
                re.MULTILINE,
            )
            for match in pattern.finditer(text):

                m1 = int(match.group("m1"))
                m2 = int(match.group("m2"))
                res = _parse_text_to_nparr_tokens(
                    match.group("content"), self.n_excitons, n_c
                )
                out[(m1, m2)] = res
        else:

            pattern = re.compile(
                r"(?P<content>(?:\d+ +)+(?:\n {2,9}\d+ +[ \-\.\deE]+)+)",
                re.MULTILINE,
            )
            i = 0

            for match in pattern.finditer(text):

                res = _parse_text_to_nparr_tokens(
                    match.group("content"), self.n_excitons, n_c
                )
                out[i] = res
                i += 1
        return out

    def get_excitonic_data(self, typ: str) -> np.ndarray:
        """Get excitonic data from the text via explicit loops."""

        pattern = re.compile(
            r"\-{4,}\s*\n *" + typ + r" *\n *-{4,} *\n"
            r"(?P<columns>[ \w\(\)\-\*]*)\n(?P<units>[ \w\(\)\-\*]*)"
            r" *\n *-{4,} *\n"
            r"(?P<content>(?:[\w\s\.\>\+Ee]|(?:\-(?!\-\-)))+)"
            r"\n *-{4,}",
            re.MULTILINE,
        )

        m = pattern.search(self.text)
        if not m:
            raise ValueError(f"No block for type={typ!r}")
        cols = m.group("columns").split()
        assert cols[0] == "Transition"
        cols = append_duplicates(cols)

        names = ["start", "end"] + cols[1:]
        ncols = len(names)
        storage = {name: [] for name in names}

        conv_int = True
        lbl_re = re.compile(r"(\d+)-(\d+)[A-Za-z]*\s*->\s*(\d+)-(\d+)[A-Za-z]*")

        for line in m.group("content").splitlines():
            tok = line.split()

            lm = lbl_re.match(tok[0] + " -> " + tok[2])
            if not lm:
                lbl_re = re.compile(
                    r"(\d+)-(\d+\.\d*)[A-Za-z]*\s*->\s*(\d+)-(\d+\.\d*)[A-Za-z]*"
                )
                lm = lbl_re.match(tok[0] + " -> " + tok[2])
                if not lm:
                    raise ValueError(f"Bad label in line: {line!r}")
                else:
                    conv_int = False

            if conv_int:
                s1, s2, e1, e2 = map(int, lm.groups())
            else:
                s1, s2, e1, e2 = map(float, lm.groups())
            storage["start"].append((s1, s2))
            storage["end"].append((e1, e2))

            floats = list(map(float, tok[3:]))
            if len(floats) != ncols - 2:
                raise ValueError(f"Column count mismatch in line: {line!r}")
            for name, val in zip(names[2:], floats):
                storage[name].append(val)

        dtype = []
        for name in names:
            if name in ("start", "end"):
                if conv_int:
                    dtype.append((name, "i4", (2,)))
                else:
                    dtype.append((name, "f8", (2,)))
            else:
                dtype.append((name, "f8"))

        arr = np.zeros(len(storage[names[0]]), dtype=dtype)
        for name in names:
            arr[name] = storage[name]

        return arr

    @cached_property
    def fluorescence_data(self):
        """Process fluorescence data."""
        return self.get_excitonic_data(
            "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS"
        )

    @property
    def GSES_data(self):
        """Process GSES data."""
        data = self.fluorescence_data
        mask = (data["start"] == [0, 1]).all(axis=1)
        data = data[mask]
        mask = data["end"][:, 1] == 1
        data = data[mask]
        return data

    @property
    def exc_en_S(self):
        """Process exc en S."""
        return self.GSES_data["Energy"]

    @property
    def tdms_S_direct(self):
        """Process tdms S direct."""
        data = np.array(self.GSES_data[["DX", "DY", "DZ"]])
        return np.column_stack([data["DX"], data["DY"], data["DZ"]])

    @property
    def tdms_S(self):
        """Process tdms S."""
        return self.get_multipole_data("Dipole", 3)[(1, 1)]

    @property
    def tqms_S(self):
        """Process tqms S."""
        out = self.get_multipole_data("Electric Quadrupole", 6)[0]
        out = list_to_symmetric_matrix_ORCA(out)
        return out

    @property
    def toms_S(self):
        """Process toms S."""
        return self.get_multipole_data("Electric Octupole", 9)[(1, 1)]

    @property
    def tmdms_S(self):
        """Process tmdms S."""
        return self.get_multipole_data("Angular momentum", 3)[(1, 1)]

    @property
    def tmqms_S(self):
        """Process tmqms S."""
        out = self.get_multipole_data("Magnetic Quadrupole", 6)[(1, 1)]
        out = list_to_symmetric_matrix_ORCA(out)
        return out

    @property
    def GSET_data(self):
        """Process GSET data."""
        data = self.fluorescence_data
        mask = (data["start"] == [0, 1]).all(axis=1)
        data = data[mask]
        mask = data["end"][:, 1] == 3
        data = data[mask]
        return data

    @property
    def exc_en_T(self):
        """Process exc en T."""
        return self.GSET_data["Energy"]

    @cached_property
    def phosphorescence_data(self):
        """Process phosphorescence data."""
        return self.get_excitonic_data(
            r"SOC CORRECTED ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS \(\- Extended \-\)"
        )

    @property
    def exc_en_so(self):
        """Process exc en so."""
        return self.phosphorescence_data["Energy"]

    @property
    def _exc_en_matrix_sobasis(self):
        """Handle exc en matrix sobasis internally."""
        gsen = [0.0]
        esen = self.exc_en_S
        eten = self.exc_en_T
        eten = np.tile(eten, 3)
        out = Units.convert(np.hstack((gsen, esen, eten)), "eV", "au")
        return np.diag(out)

    @property
    def mat_pureSO(self):
        """Process mat pureSO."""
        return self.mat_SO - self._exc_en_matrix_sobasis

    @cached_property
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
        eigvals, eigvecs = self.eigensystem_SO
        ind = np.argsort(np.real(eigvals))

        return eigvecs[:, ind]

    @property
    def so_eigvec_singlet(self):
        """Process so eigvec singlet."""
        eigvecs = self.eigvecs_SO
        eigvecssinglet = eigvecs[1 : self.n_singlets + 1, :]
        return eigvecssinglet

    @property
    def so_eigvec_triplet(self):
        """Process so eigvec triplet."""
        eigvecs = self.eigvecs_SO
        eigvecstriplet = eigvecs[
            self.n_singlets + 1 : self.n_singlets + self.n_triplets + 1, :
        ]
        return eigvecstriplet

    def transpose_singlet(self, feat, sq=False, inv=False):
        """Process transpose singlet."""
        eigvecs = self.so_eigvec_singlet
        if inv:
            eigvecs = eigvecs.T
        else:
            eigvecs = eigvecs.conj()

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

    @property
    def tdms_so(self):
        """Process tdms so."""
        datare = np.array(self.phosphorescence_data[["DX(re)", "DY(re)", "DZ(re)"]])
        dataim = np.array(self.phosphorescence_data[["DX(im)", "DY(im)", "DZ(im)"]])
        datare = np.column_stack([datare["DX(re)"], datare["DY(re)"], datare["DZ(re)"]])
        dataim = np.column_stack([dataim["DX(im)"], dataim["DY(im)"], dataim["DZ(im)"]])
        return datare + 1j * dataim

    @property
    def orbwins(self):
        """Process orbwins."""
        pattern = re.compile(
            r"Orbital Range Operator *\d+: *(\d+) *- *(\d+) *to *(\d+) *- *(\d+)"
        )
        match = pattern.search(self.text)
        if match:
            a, b, c, d = map(int, match.groups())
            print(f"Extracted: {a}-{b} to {c}-{d}")
        return a, b, c, d

    @property
    def rocis_norbs(self):
        """Process rocis norbs."""
        a, b, c, d = self.orbwins
        nSOMO = 1
        nDOMO = b - (a - 1) - nSOMO
        nVMO = d - (c - 1)
        return nDOMO, nSOMO, nVMO

    @cache
    def get_exciton_NACMEs(self) -> dict:
        """
        Parse TD-DFT gradients for singlet roots from ORCA output.

        Returns:
            dict: Dictionary with singlet root numbers as keys and gradient data as values
                  Each entry contains: gradients, norm, rms, max_gradient, atom_labels
        """

        singlet_pattern = re.compile(
            r"(?P<type>\w+) ROOT (?P<root>\d+)\s*\n\s*-+\s*\n[\s\S]*?CARTESIAN GRADIENT\s*\n\s*-+\s*\n(?P<gradient_data>(?:\s*\d+\s+[a-zA-Z]+\s*:\s*[\d\.\-\s]+\n)+)Difference[\s\S]*?in ETFs\s*\-*\ *\n\ *\n(?P<nacme_data>(?:\s*\d+\s+[a-zA-Z]+\s*:\s*[\d\.\-\s]+\n)+)",
            re.MULTILINE | re.DOTALL,
        )

        results = {}

        for match in singlet_pattern.finditer(self.text):
            typ = match.group("type")
            root_num = int(match.group("root"))
            gradient_data = match.group("gradient_data")
            nacme_data = match.group("nacme_data")

            gradients, atom_labels = self._parse_gradient_lines(gradient_data)
            nacmes, atom_labels = self._parse_gradient_lines(nacme_data)

            if typ in results:
                results[typ][root_num] = {
                    "gradients": gradients,
                    "atom_labels": atom_labels,
                    "nacmes": nacmes,
                }
            else:
                results[typ] = {}
                results[typ][root_num] = {
                    "gradients": gradients,
                    "atom_labels": atom_labels,
                    "nacmes": nacmes,
                }

        return results

    @property
    def nacmes_per_mult(self):
        """Process nacmes per mult."""
        nacmes = self.get_exciton_NACMEs()
        outs = {}
        for typ, nacmes_typ in nacmes.items():
            amount_of_gradient = max(list(nacmes_typ.keys()))
            out = np.zeros(
                (
                    amount_of_gradient,
                    *nacmes_typ[amount_of_gradient]["nacmes"].shape,
                )
            )
            for k in range(amount_of_gradient):
                v = None
                ik = 0
                while v is None:
                    if ik >= 0:
                        try:
                            v = nacmes_typ[k + 1 - ik]
                        except:
                            ik += 1
                        if ik > k:
                            ik = -1
                    else:
                        try:
                            v = nacmes_typ[k + 1 - ik]
                        except:
                            ik -= 1
                        if ik < -100:
                            raise Exception("Found no gradients.")
                out[k, :, :] = v["nacmes"]

            outs[typ] = out
        return outs

    @property
    def singlet_nacmes(self):
        """Process singlet nacmes."""
        return self.nacmes_per_mult["SINGLET"]

    @cache
    def get_exciton_gradients(self) -> dict:
        """
        Parse TD-DFT gradients for singlet roots from ORCA output.

        Returns:
            dict: Dictionary with singlet root numbers as keys and gradient data as values
                  Each entry contains: gradients, norm, rms, max_gradient, atom_labels
        """

        singlet_pattern = re.compile(
            r"(?P<type>\w+) ROOT (?P<root>\d+)\s*\n\s*-+\s*\n[\-\+\sa-zA-Z0-9\.\(\)\*\/\,\_\=\>\:\|\&]*?CARTESIAN GRADIENT\s*\n\s*-+\s*\n(?P<gradient_data>(?:\s*\d+\s+[a-zA-Z]+\s*:\s*[\d\.\-\s]+\n)+)",
            re.MULTILINE | re.DOTALL,
        )

        results = {}

        for match in singlet_pattern.finditer(self.text):
            typ = match.group("type")
            root_num = int(match.group("root"))
            gradient_data = match.group("gradient_data")

            gradients, atom_labels = self._parse_gradient_lines(gradient_data)

            if typ in results:
                results[typ][root_num] = {
                    "gradients": gradients,
                    "atom_labels": atom_labels,
                }
            else:
                results[typ] = {}
                results[typ][root_num] = {
                    "gradients": gradients,
                    "atom_labels": atom_labels,
                }

        return results

    @property
    def excitonic_gradients_per_mult(self):
        """Process excitonic gradients per mult."""
        exc_grads = self.get_exciton_gradients()
        outs = {}
        for typ, exc_grads_typ in exc_grads.items():
            amount_of_gradient = max(list(exc_grads_typ.keys()))
            out = np.zeros(
                (
                    amount_of_gradient,
                    *exc_grads_typ[amount_of_gradient]["gradients"].shape,
                )
            )
            for k in range(amount_of_gradient):
                v = None
                ik = 0
                while v is None:
                    if ik >= 0:
                        try:
                            v = exc_grads_typ[k + 1 - ik]
                        except:
                            ik += 1
                        if ik > k:
                            ik = -1
                    else:
                        try:
                            v = exc_grads_typ[k + 1 - ik]
                        except:
                            ik -= 1
                        if ik < -100:
                            raise Exception("Found no gradients.")
                out[k, :, :] = v["gradients"]

            outs[typ] = out
        return outs

    @property
    def singlet_gradients(self):
        """Process singlet gradients."""
        return self.excitonic_gradients_per_mult["SINGLET"]

    @property
    def triplet_gradients(self):
        """Process triplet gradients."""
        return self.excitonic_gradients_per_mult["TRIPLET"]

    def _parse_gradient_lines(self, gradient_text: str) -> tuple:
        """
        Parse individual gradient lines from the gradient section.

        Args:
            gradient_text: Text containing the gradient data lines

        Returns:
            tuple: (gradients_array, atom_labels) where gradients_array is (n_atoms, 3)
                   and atom_labels is list of (atom_index, element_symbol)
        """

        line_pattern = re.compile(
            r"^\s*(?P<atom_idx>\d+)\s+(?P<element>[A-Z][a-z]?)\s*:\s*"
            r"(?P<grad_x>[\-\d\.]+)\s+(?P<grad_y>[\-\d\.]+)\s+(?P<grad_z>[\-\d\.]+)",
            re.MULTILINE,
        )

        gradients = []
        atom_labels = []

        for match in line_pattern.finditer(gradient_text):
            atom_idx = int(match.group("atom_idx"))
            element = match.group("element")
            grad_x = float(match.group("grad_x"))
            grad_y = float(match.group("grad_y"))
            grad_z = float(match.group("grad_z"))

            gradients.append([grad_x, grad_y, grad_z])
            atom_labels.append((atom_idx, element))

        return np.array(gradients), atom_labels

    @cache
    def get_gradient_by_root(self, root_number: int) -> dict:
        """
        Get gradient data for a specific singlet root.

        Args:
            root_number: The singlet root number (1, 2, etc.)

        Returns:
            dict: Gradient data for the specified root, or None if not found
        """
        all_gradients = self.get_singlet_gradients()
        return all_gradients.get(root_number, None)

    @property
    def available_singlet_roots(self) -> list:
        """Get list of available singlet root numbers."""
        return list(self.get_singlet_gradients().keys())
