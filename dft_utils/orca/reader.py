import glob
import os
import re
from collections import defaultdict
from functools import cached_property
from os.path import join
from pathlib import Path
from typing import Dict, Type

import dill
import numpy as np

from .._plams import require_plams
from ..units import Units
from .cisparser import CISParser, ROCISParser
from .FileParser import FileParser
from .gbwparser import GBWParser
from .hessparser import HessParser
from .jsonparser import UBJSONParser, iJSONParser
from .outparser import OutParser
from .propertyparser import PropertyParser
from .rootsparser import SimpleCoordParser
from .tarparser import TarGzParser
from .tddft_helpers import (
    _contract_multipole_restricted,
    _contract_multipole_restricted_double,
    _contract_multipole_restricted_double_MO,
    _contract_multipole_restricted_MO,
    _contract_multipole_unrestricted,
    _contract_multipole_unrestricted_MO,
    intsAO_to_intsMO_restricted,
    intsAO_to_intsMO_unrestricted,
    intsMO_cut_restricted,
    intsMO_cut_unrestricted,
    static_multipole,
    static_multipole_restricted_MO,
    static_multipole_unrestricted,
    static_multipole_unrestricted_MO,
    to_soc,
)

PARSER_REGISTRY: Dict[str, Type[FileParser]] = {}


def register_parser(ext: str):
    """Decorator to register a parser for a given extension."""

    def decorator(cls: Type[FileParser]):
        """Process decorator."""
        PARSER_REGISTRY[ext] = cls
        return cls

    return decorator


@register_parser(".dill")
class DillParser(FileParser):
    def parse(self):
        """Parse the supplied input."""
        with open(self.path, "rb") as f:
            return dill.load(f)


@register_parser(".run")
class RunParser(FileParser):
    def parse(self):
        """Parse the supplied input."""
        return open(self.path).read().splitlines()


@register_parser(".err")
class ErrParser(FileParser):
    def parse(self):
        """Parse the supplied input."""
        return open(self.path).read().splitlines()


@register_parser(".in")
class InParser(FileParser):
    def parse(self):
        """Parse the supplied input."""
        return open(self.path).read().splitlines()


CISParser = register_parser(".cis")(CISParser)
ROCISParser = register_parser(".Davidson.ci")(ROCISParser)
ROCISParser = register_parser(".Davidson.hm.ci")(ROCISParser)
ROCISParser = register_parser(".Davidson.lm.ci")(ROCISParser)


GBWParser = register_parser(".gbw")(GBWParser)

OutParser = register_parser(".out")(OutParser)
OutParser = register_parser(".out2")(OutParser)

PropertyParser = register_parser(".property.txt")(PropertyParser)

SimpleCoordParser = register_parser(".grad.txt")(SimpleCoordParser)

TarGzParser = register_parser(".tar.gz")(TarGzParser)

HessParser = register_parser(".hess")(HessParser)

HessParser = register_parser(".hess.root")(HessParser)


iJSONParser = register_parser(".json")(iJSONParser)

UBJSONParser = register_parser(".ubjson")(UBJSONParser)


class ParserFactory:
    @staticmethod
    def get_ext(path: str) -> str:
        """Return ext."""
        _, ext = os.path.splitext(path)

        if ext == ".gz" and path.endswith(".tar.gz"):
            ext = ".tar.gz"
        if ext == ".txt" and path.endswith(".property.txt"):
            ext = ".property.txt"
        if ext == ".txt" and path.endswith(".grad.txt"):
            ext = ".grad.txt"
        if ext == ".ci" and path.endswith(".Davidson.ci"):
            ext = ".Davidson.ci"
        if ext == ".ci" and path.endswith(".Davidson.hm.ci"):
            ext = ".Davidson.hm.ci"
        pattern = re.compile(
            r"\.ES\.hess\.root\d+",
            re.MULTILINE,
        )
        if pattern.search(path):
            ext = ".hess.root"
        return ext

    @staticmethod
    def get_parser(path: str) -> FileParser:
        """Return parser."""
        ext = ParserFactory.get_ext(path)
        parser_cls = PARSER_REGISTRY.get(ext)
        if not parser_cls:
            raise ValueError(f"No parser registered for extension '{ext}'")
        return parser_cls(path)


def load_all_from_dir(directory: str):
    """Scan directory and parse all recognized files."""
    parsers = {}
    for fname in os.listdir(directory):
        full = join(directory, fname)
        ext = ParserFactory.get_ext(full)
        try:
            parser = ParserFactory.get_parser(full)
        except ValueError:

            continue

        if ext in parsers:
            if not isinstance(parsers[ext], list):
                parsers[ext] = [parsers[ext]]
            parsers[ext].append(parser)
        else:
            parsers[ext] = parser
    return parsers


def load_all_hess_roots(directory: Path) -> Dict[int, any]:
    """
    Find all files named 'hess.root{index}' in `directory`,
    parse each with HessParser, and return a mapping index → parsed data.
    """
    pattern = str(directory / "hess.root*")
    root_files = glob.glob(pattern)
    parsed: Dict[int, any] = {}

    for filepath in root_files:

        m = re.match(r".*hess\.root(\d+)$", filepath)
        if not m:
            continue
        idx = int(m.group(1))

        parser_cls = PARSER_REGISTRY.get(".hess")
        if parser_cls is None:
            raise RuntimeError("No parser registered for '.hess' files")
        parser = parser_cls(Path(filepath))
        parsed[idx] = parser.parse()

    return parsed


pattern = r"(?P<jobname>.+?)_(?P<basis>x2c[\w\-]+)_(?P<func>[\w\-]+)_(?P<charge>(?:-?\d+)|(?:utriplet))_(?P<type>[a-z\-\_]*[a-z]+(?:_\-?\d+)*)(?:_COSMO_eps(?P<epsr>(?:[-]?[0-9]+(?:[.][0-9]*)?|[.][0-9]+)))?"
JOB_RE = re.compile(pattern)


def prepSO(GG, GS, SS, TT):
    """Process prepSO."""
    nstates = 1 + SS.shape[0] + 3 * TT.shape[0]
    out = np.zeros((nstates, nstates, *SS.shape[2:]))
    out[0, 0, ...] = GG
    out[1 : 1 + SS.shape[0], 0, ...] = GS
    out[0, 1 : 1 + SS.shape[0], ...] = GS.conj()
    out[1 : 1 + SS.shape[0], 1 : 1 + SS.shape[0], ...] = SS
    for i in range(3):
        out[
            1 + SS.shape[0] + i * TT.shape[0] : 1 + SS.shape[0] + (i + 1) * TT.shape[0],
            1 + SS.shape[0] + i * TT.shape[0] : 1 + SS.shape[0] + (i + 1) * TT.shape[0],
            ...,
        ] = TT
    return out


class SingleOrcaJob:
    def __init__(self, folder):
        """Initialize the object."""
        self.folder = folder
        assert os.path.isdir(self.folder), self.folder

    @property
    def match(self):
        """Process match."""
        match = JOB_RE.match(self.name)
        if match:
            return match
        else:
            return None

    @property
    def name(self):
        """Process name."""
        return os.path.basename(self.folder)

    @property
    def jobname(self):
        """Process jobname."""
        return self.match.group("jobname")

    @property
    def charge(self):
        """Process charge."""
        return self.match.group("charge")

    @property
    def type(self):
        """Process type."""
        return self.match.group("type")

    @property
    def epsr(self):
        """Process epsr."""
        return float(self.match.group("epsr"))

    @cached_property
    def parsers(self):
        """Parse the supplied input."""
        return load_all_from_dir(self.folder)

    @property
    def parser_types(self):
        """Parse the supplied input."""
        return set(k for k in self.parsers.keys())

    def get_parser(self, type: str, i=None):
        """Return parser."""
        parsers = self.parsers[type]

        if isinstance(parsers, list):
            if i is None:

                return max(parsers, key=lambda item: item.path.stat().st_size)
            else:
                return parsers[i]
        else:
            return parsers

    def get_parsed_data(self, type):
        """Return parsed data."""
        return self.get_parser(type).parse()

    @property
    def atom_symbols(self):
        """Process atom symbols."""
        return self.get_parser(".property.txt").atom_symbols

    @property
    def hessian(self):
        """Process hessian."""
        return self.get_parser(".hess").hessian

    @property
    def atom_xyzs_ang(self):
        """Process atom xyzs ang."""
        return Units.convert(
            self.get_parser(".property.txt").atom_xyzs, "au", "Angstrom"
        )

    @property
    def atom_numbers(self):
        """Process atom numbers."""
        PT = require_plams()[4]
        p = self.atom_symbols
        return [PT.get_atomic_number(symbol) for symbol in p]

    @property
    def atom_masses(self):
        """Process atom masses."""
        PT = require_plams()[4]
        p = self.atom_symbols
        return [PT.get_mass(symbol) for symbol in p]

    @property
    def centre_of_mass(self):
        """Process centre of mass."""
        Mtot = 1.0 / (np.sum(self.atom_masses))
        out = np.array(self.atom_masses)[:, None] * np.array(self.atom_xyzs)
        return np.sum(out, axis=0) * Mtot

    @property
    def nuclear_dipole(self):
        """Process nuclear dipole."""
        out = np.array(self.atom_xyzs) * np.array(self.atom_numbers)[:, None]
        return np.sum(out, axis=0)

    @property
    def mol(self):
        """Process mol."""
        Molecule = require_plams()[1]
        return Molecule(numbers=self.atom_numbers, positions=self.atom_xyzs_ang)

    @property
    def init_coords(self):
        """Process init coords."""
        return np.asarray(self.mol)

    @property
    def exc_coords_dict(self):
        """Process exc coords dict."""
        res = {}
        for parser in self.parsers[".hess.root"]:
            match = re.search(r"\.root(\d+)$", parser.path.name)
            if match:
                root = int(match.group(1))
                coords = np.asarray(parser.mol)

                res[root] = coords
        return res

    @property
    def exc_grads_dict(self):
        """Process exc grads dict."""
        res = {}
        for parser in self.parsers[".grad.txt"]:
            match = re.search(r"\.(\w+)\.root(\d+)\.", parser.path.name)
            if match:
                root = int(match.group(2))
                group = str(match.group(1))
                grad = np.asarray(parser.parse())

                if group in res:
                    res[group][root] = grad
                else:
                    res[group] = {root: grad}
        return res

    @property
    def exc_coords(self):
        """Process exc coords."""
        d = self.exc_coords_dict
        shape = d[1].shape
        n = len(d)
        res = np.empty((n, *shape))
        for k, v in d.items():
            res[k - 1] = v
        return res

    @property
    def exc_grads(self):
        """Process exc grads."""
        d = self.exc_grads_dict
        groups = list(d.keys())

        out = {}
        for k1, v1 in d.items():
            shape = v1[1].shape
            n = len(v1)
            res = np.empty((n, *shape))
            for k2, v2 in v1.items():
                res[k2 - 1] = v2
            out[k1] = res
        return out

    @property
    def mo_occ(self):
        """Process mo occ."""
        mo_occ = np.array(self.get_parser(".json").mo_occ)
        if not self.restricted:
            mo_occ = mo_occ.reshape(2, -1)
        return mo_occ

    @property
    def mo_coeff(self):
        """Process mo coeff."""
        mo_coeff = np.array(self.get_parser(".json").mo_coeff)
        if not self.restricted:

            mo_coeff = mo_coeff.reshape(mo_coeff.shape[0], 2, mo_coeff.shape[0])
            mo_coeff = np.moveaxis(mo_coeff, 1, 0)
        return mo_coeff

    @property
    def delta_pos_exc(self):
        """Process delta pos exc."""
        exc_coords = self.exc_coords
        exc_coords = np.reshape(exc_coords, (exc_coords.shape[0], -1))
        init_coords = self.init_coords.flatten()[None, :]
        return exc_coords - init_coords

    def get_intsAO(self, name):
        """Return intsAO."""
        return getattr(self, name)

    def convert_intsAO(self, intsAO, name):
        """Process convert intsAO."""
        if not isinstance(intsAO, np.ndarray):
            if isinstance(intsAO, str):
                intsAO = self.get_intsAO(intsAO)
            elif name is not None:
                intsAO = self.get_intsAO(name)
            else:
                raise ValueError("Did not get intsAO")
        return intsAO

    def get_intsMO(self, name):
        """Return intsMO."""
        intsAO = self.convert_intsAO(name, None)
        if self.restricted:
            return intsAO_to_intsMO_restricted(intsAO, self.mo_coeff)
        else:
            return intsAO_to_intsMO_unrestricted(intsAO, self.mo_coeff)

    def multipole_staticMO_self(self, intsAO=None):
        """Process multipole staticMO self."""
        intsMO = self.get_intsMO(intsAO)
        if self.restricted:
            mo_occ = self.mo_occ
            mask = mo_occ == 2
            intsMO = intsMO[..., mask, :][..., mask]
            mo_occ = mo_occ[mask]
            return static_multipole_restricted_MO(intsMO, mo_occ)
        else:
            return static_multipole_unrestricted_MO(intsMO, self.mo_occ)

    def multipole_GSMO_self(self, intsAO=None):
        """Process multipole GSMO self."""
        intsMO = self.get_intsMO(intsAO)
        if self.restricted:
            __, intsMOov, __ = intsMO_cut_restricted(intsMO, self.mo_occ)
            return _contract_multipole_restricted_MO(intsMOov, self.singlet_xy)
        else:
            __, intsMOaov, __, __, intsMObov, __ = intsMO_cut_unrestricted(
                intsMO, self.mo_occ
            )
            return _contract_multipole_unrestricted_MO(
                intsMOaov, intsMObov, self.all_xy
            )

    def multipole_multipole_SSMO_self(self, intsAO=None):
        """Process multipole multipole SSMO self."""
        intsMO = self.get_intsMO(intsAO)
        if self.restricted:
            intsMOoo, intsMOov, intsMOvv = intsMO_cut_restricted(intsMO, self.mo_occ)
            static = intsMO_cut_restricted(intsMOoo, self.mo_occ[self.mo_occ == 2])
            return _contract_multipole_restricted_double_MO(
                intsMOoo, intsMOvv, static, self.singlet_xy, self.singlet_xy
            )
        else:
            raise NotImplementedError()

    def multipole_multipole_TTMO_self(self, intsAO=None):
        """Process multipole multipole TTMO self."""
        intsMO = self.get_intsMO(intsAO)
        if self.restricted:
            intsMOoo, intsMOov, intsMOvv = intsMO_cut_restricted(intsMO, self.mo_occ)
            static = intsMO_cut_restricted(intsMOoo, self.mo_occ[self.mo_occ == 2])
            return _contract_multipole_restricted_double_MO(
                intsMOoo, intsMOvv, static, self.triplet_xy, self.triplet_xy
            )
        else:
            raise NotImplementedError()

    def multipole_multipole_self_prepSOMO(self, intsAO=None, return_all=False):
        """Process multipole multipole self prepSOMO."""
        if intsAO.lower() != "spin":
            intsMO = self.get_intsMO(intsAO)
            if self.restricted:
                intsMOoo, intsMOov, intsMOvv = intsMO_cut_restricted(
                    intsMO, self.mo_occ
                )
                GG = static_multipole_restricted_MO(
                    intsMOoo, self.mo_occ[self.mo_occ == 2]
                )
                GS = _contract_multipole_restricted_MO(intsMOov, self.singlet_xy)
                SS = _contract_multipole_restricted_double_MO(
                    intsMOoo, intsMOvv, GG, self.singlet_xy, self.singlet_xy
                )
                TT = _contract_multipole_restricted_double_MO(
                    intsMOoo, intsMOvv, GG, self.triplet_xy, self.triplet_xy
                )

                out = prepSO(GG, GS, SS, TT)
                if return_all:
                    return out, (GG, GS, SS, TT)
                else:
                    return out
            else:
                raise NotImplementedError()
        else:
            return self.multipole_multipole_self_prepSPIN(return_all=return_all)

    def multipole_multipole_self_prepSPIN(self, return_all=False):
        """Process multipole multipole self prepSPIN."""
        n_S = len(self.singlet_xy)
        n_T = len(self.triplet_xy)
        out = np.zeros((1 + n_S + 3 * n_T, 1 + n_S + 3 * n_T, 3), dtype=complex)

        m_ST = [0, -1, 1]
        invsqrt2 = 0.5 * np.sqrt(2.0)
        for iT1, m_S1 in enumerate(m_ST):
            for iT2, m_S2 in enumerate(m_ST):

                if m_S1 == m_S2:
                    out[
                        1 + n_S + iT1 * n_T : 1 + n_S + (iT1 + 1) * n_T,
                        1 + n_S + iT2 * n_T : 1 + n_S + (iT2 + 1) * n_T,
                        2,
                    ] = (
                        np.eye(n_T, dtype=complex) * m_S1
                    )

                elif abs(m_S1 - m_S2) == 1:
                    if m_S1 > m_S2:
                        out[
                            1 + n_S + iT1 * n_T : 1 + n_S + (iT1 + 1) * n_T,
                            1 + n_S + iT2 * n_T : 1 + n_S + (iT2 + 1) * n_T,
                            0,
                        ] = (
                            np.eye(n_T, dtype=complex) * invsqrt2
                        )
                        out[
                            1 + n_S + iT1 * n_T : 1 + n_S + (iT1 + 1) * n_T,
                            1 + n_S + iT2 * n_T : 1 + n_S + (iT2 + 1) * n_T,
                            1,
                        ] = (
                            -1.0j * np.eye(n_T, dtype=complex) * invsqrt2
                        )
                    elif m_S1 < m_S2:
                        out[
                            1 + n_S + iT1 * n_T : 1 + n_S + (iT1 + 1) * n_T,
                            1 + n_S + iT2 * n_T : 1 + n_S + (iT2 + 1) * n_T,
                            0,
                        ] = (
                            np.eye(n_T, dtype=complex) * invsqrt2
                        )
                        out[
                            1 + n_S + iT1 * n_T : 1 + n_S + (iT1 + 1) * n_T,
                            1 + n_S + iT2 * n_T : 1 + n_S + (iT2 + 1) * n_T,
                            1,
                        ] = (
                            1.0j * np.eye(n_T, dtype=complex) * invsqrt2
                        )
                    else:
                        raise Exception("This should be impossible.")
        if return_all:
            return out, (
                out[0, 0],
                out[1, :],
                out[1 : 1 + n_S, 1 : 1 + n_S],
                out[1 + n_S :, 1 + n_S :],
            )
        else:
            return out

    def multipole_multipole_SOSO_MO_self(self, name, return_all=False):
        """Process multipole multipole SOSO MO self."""
        intsEXC = self.multipole_multipole_self_prepSOMO(name, return_all=return_all)
        if return_all:
            extra = (*intsEXC[1], self.eigvecs_SO)
            intsEXC = intsEXC[0]
            return to_soc(intsEXC, self.eigvecs_SO), extra
        else:
            return to_soc(intsEXC, self.eigvecs_SO)

    def multipole_static_self(self, intsAO=None, name=None):
        """Process multipole static self."""
        intsAO = self.convert_intsAO(intsAO, name)
        if self.restricted:
            return static_multipole(intsAO, self.mo_coeff, self.mo_occ)
        else:
            return static_multipole_unrestricted(intsAO, self.mo_coeff, self.mo_occ)

    def multipole_GS_self(self, intsAO=None, name=None):
        """Process multipole GS self."""
        intsAO = self.convert_intsAO(intsAO, name)
        if self.restricted:
            return _contract_multipole_restricted(
                intsAO, self.singlet_xy, self.mo_coeff, self.mo_occ
            )
        else:
            return _contract_multipole_unrestricted(
                intsAO, self.all_xy, self.mo_coeff, self.mo_occ
            )

    def multipole_multipole_SS_self(self, intsAO=None, name=None):
        """Process multipole multipole SS self."""
        intsAO = self.convert_intsAO(intsAO, name)
        if self.restricted:
            return _contract_multipole_restricted_double(
                intsAO, self.singlet_xy, self.singlet_xy, self.mo_coeff, self.mo_occ
            )
        else:
            raise NotImplementedError()

    def multipole_multipole_TT_self(self, intsAO=None, name=None):
        """Process multipole multipole TT self."""
        intsAO = self.convert_intsAO(intsAO, name)
        if self.restricted:
            return _contract_multipole_restricted_double(
                intsAO, self.triplet_xy, self.triplet_xy, self.mo_coeff, self.mo_occ
            )
        else:
            raise NotImplementedError()

    def multipole_multipole_self_prepSO(self, intsAO=None, name=None, return_all=False):
        """Process multipole multipole self prepSO."""
        intsAO = self.convert_intsAO(intsAO, name)
        GG = self.multipole_static_self(intsAO)
        GS = self.multipole_GS_self(intsAO)
        SS = self.multipole_multipole_SS_self(intsAO)
        TT = self.multipole_multipole_TT_self(intsAO)
        out = prepSO(GG, GS, SS, TT)

        if return_all:
            return out, (GG, GS, SS, TT)
        else:
            return out

    @property
    def get_rocis_cis(self):
        """Return rocis cis."""
        nDOMO, nSOMO, nVMO = self.rocis_norbs
        out = self.get_parser(".Davidson.ci").get_data(nDOMO, nSOMO, nVMO)
        return out

    @property
    def rocis_cisT(self):
        """Process rocis cisT."""
        out = []
        for i in range(6):
            out.append(np.array([x[i] for x in self.get_rocis_cis]))
        return out

    @property
    def preppedsoc_gradients(self):
        """Process preppedsoc gradients."""
        sgrads = self.singlet_gradients
        tgrads = self.triplet_gradients
        tgrads = np.tile(tgrads, (3, 1, 1))
        gsgrads = np.zeros((1, *sgrads.shape[1:]))
        grads = np.concatenate((gsgrads, sgrads, tgrads))
        return grads

    @property
    def soc_gradients(self):
        """Process soc gradients."""
        grads = self.preppedsoc_gradients

        grads = np.tensordot(
            np.real(self.eigvecs_SO * self.eigvecs_SO.conj()), grads, axes=(0, 0)
        )
        return grads

    def multipole_multipole_SOSO_self(self, name, return_all=False):
        """Process multipole multipole SOSO self."""
        intsEXC = self.multipole_multipole_self_prepSO(name, return_all=return_all)
        if return_all:
            extra = (*intsEXC[1], self.eigvecs_SO)
            intsEXC = intsEXC[0]
            return to_soc(intsEXC, self.eigvecs_SO), extra
        else:
            return to_soc(intsEXC, self.eigvecs_SO)

    def multipole_multipole_SOSO_ineffi_self(self, name, return_all=False):
        """Process multipole multipole SOSO ineffi self."""
        if return_all:
            raise ValueError("Return all cannot be true.")
        GG = self.multipole_static_self(name)
        GS = self.multipole_GS_self(name)
        SS = self.multipole_multipole_SS_self(name)
        TT = self.multipole_multipole_TT_self(name)
        nstates = 1 + SS.shape[0] + 3 * TT.shape[0]
        out = np.zeros((nstates, nstates, *SS.shape[2:]), dtype=complex)

        inS = np.zeros((1 + SS.shape[0], 1 + SS.shape[0], *SS.shape[2:]), dtype=complex)
        inS[0, 0, ...] = GG
        inS[1 : 1 + SS.shape[0], 0, ...] = GS
        inS[0, 1 : 1 + SS.shape[0], ...] = GS.conj()
        inS[1 : 1 + SS.shape[0], 1 : 1 + SS.shape[0], ...] = SS
        out += to_soc(inS, self.eigvecs_SO[: 1 + SS.shape[0], :])

        inT = np.zeros((TT.shape[0], TT.shape[0], *TT.shape[2:]), dtype=complex)

        inT[:, :] = TT
        for i in range(3):
            out += to_soc(
                inT,
                self.eigvecs_SO[
                    1
                    + SS.shape[0]
                    + i * TT.shape[0] : 1
                    + SS.shape[0]
                    + (i + 1) * TT.shape[0],
                    :,
                ],
            )
        return out

    def __getattr__(self, name):
        """
        Automatically look for attributes in parsers if they don't exist in this class.
        First tries to find a parser with the attribute name as its type,
        then checks if any parser has the requested attribute.
        """

        for parser_type in self.parser_types:
            parser = self.get_parser(parser_type)
            try:

                result = getattr(parser, name)
                if callable(result):

                    return lambda *args, **kwargs: getattr(parser, name)(
                        *args, **kwargs
                    )
                return result
            except AttributeError:

                continue

        raise AttributeError(
            f"'{self.__class__.__name__}' object and its parsers have no attribute '{name}'"
        )


def find_all_jobs(folder):
    """Process find all jobs."""
    matches = []
    for f in sorted(os.listdir(folder)):
        match = JOB_RE.match(os.path.basename(f))
        if match:

            path = join(folder, f)
            if os.path.isdir(path):
                out = SingleOrcaJob(path)
                matches.append(out)

    return matches


class OrcaJobCollection:
    """Collection of SingleOrcaJobs grouped by jobname."""

    def __init__(self, jobs=None):
        """
        Initialize a collection with optional initial jobs.

        Args:
            jobs: List of SingleOrcaJob instances or None
        """
        self.jobs_by_name = defaultdict(list)
        if jobs:
            for job in jobs:
                self.add_job(job)

    def add_job(self, job):
        """Add a SingleOrcaJob to the collection."""
        if not isinstance(job, SingleOrcaJob):
            raise TypeError(f"Expected SingleOrcaJob, got {type(job).__name__}")

        if job.match is None:
            return

        self.jobs_by_name[job.jobname].append(job)

    def get_jobs(self, jobname):
        """Get all jobs with the specified jobname."""
        return self.jobs_by_name.get(jobname, [])

    @property
    def jobnames(self):
        """Get list of all unique jobnames in the collection."""
        return list(self.jobs_by_name.keys())

    @property
    def job_counts(self):
        """Get dictionary of jobname to count of jobs."""
        return {name: len(jobs) for name, jobs in self.jobs_by_name.items()}

    def __len__(self):
        """Get total number of jobs in the collection."""
        return sum(len(jobs) for jobs in self.jobs_by_name.values())

    def __getitem__(self, jobname):
        """Get jobs for a specific jobname using dictionary-like access."""
        return self.get_jobs(jobname)

    def __iter__(self):
        """Iterate through all jobnames."""
        return iter(self.jobnames)


class GroupedOrcaJob:
    """Class that combines all SingleOrcaJob instances with the same jobname."""

    @classmethod
    def create(cls, jobname, jobs):
        """
        Factory method to create either a GroupedOrcaJob or return the single job.

        Args:
            jobname: The common job name
            jobs: List of SingleOrcaJob instances with this jobname

        Returns:
            Either a GroupedOrcaJob instance or the single SingleOrcaJob
        """
        if not jobs:
            return None
        elif len(jobs) == 1:
            return jobs[0]
        else:
            return cls(jobname, jobs)

    def __init__(self, jobname, jobs):
        """
        Initialize with a jobname and related jobs.

        Args:
            jobname: The common job name
            jobs: List of SingleOrcaJob instances with this jobname
        """
        self.jobname = jobname
        self.jobs = jobs

    @property
    def charges(self):
        """Get all unique charge values across jobs."""
        return sorted(set(job.charge for job in self.jobs))

    @property
    def types(self):
        """Get all unique type values across jobs."""
        return sorted(set(job.type for job in self.jobs))

    @property
    def epsr_values(self):
        """Get all unique epsr values across jobs."""
        return sorted(set(job.epsr for job in self.jobs))

    @property
    def folders(self):
        """Get all folders for the jobs."""
        return [job.folder for job in self.jobs]

    def get_jobs_by_charge(self, charge):
        """Filter jobs by charge value."""
        return GroupedOrcaJob.create(
            self.jobname, [job for job in self.jobs if job.charge == charge]
        )

    def get_jobs_by_type(self, type_value):
        """Filter jobs by type value."""
        return GroupedOrcaJob.create(
            self.jobname, [job for job in self.jobs if job.type == type_value]
        )

    def get_jobs_by_epsr(self, epsr):
        """Filter jobs by epsr value."""
        return GroupedOrcaJob.create(
            self.jobname, [job for job in self.jobs if job.epsr == epsr]
        )

    def get_job(self, charge=None, type_value=None, epsr=None):
        """
        Get a specific job matching the criteria.

        Args:
            charge: Optional charge value to match
            type_value: Optional type value to match
            epsr: Optional epsr value to match

        Returns:
            SingleOrcaJob matching all specified criteria or None if not found
        """
        matches = self.jobs

        if charge is not None:
            matches = [job for job in matches if job.charge == charge]

        if type_value is not None:
            matches = [job for job in matches if job.type == type_value]

        if epsr is not None:
            matches = [job for job in matches if job.epsr == epsr]

        return matches[0] if matches else None

    def get_total_energy(self, charge, geom=None):
        """Return total energy."""
        if geom is None:
            geom = charge

        if geom == charge:
            job1 = "geoopt"
        else:
            job1 = f"reorg_{charge}"

        return self.get_jobs_by_charge(geom).get_jobs_by_type(job1).bond_energy

    def get_delta_energy(self, charge1, charge2, geom1=None, geom2=None):
        """Return delta energy."""
        return self.get_total_energy(charge2, geom2) - self.get_total_energy(
            charge1, geom1
        )

    @property
    def hessian(self):
        """Process hessian."""
        return self.get_jobs_by_type("hess").hessian

    @property
    def mol(self):
        """Process mol."""
        return self.get_jobs_by_type("geoopt").mol

    @property
    def mol_masses(self):
        """Process mol masses."""
        return [a.mass for a in self.mol.atoms]

    @property
    def mol_coords(self):
        """Process mol coords."""
        return np.asarray(self.mol)

    @property
    def nuc_gradients(self):
        """Process nuc gradients."""
        return self.get_jobs_by_type("exc").nuc_gradients

    @property
    def delta_pos_exc(self):
        """Process delta pos exc."""
        return self.get_jobs_by_type("exc").delta_pos_exc

    @property
    def tdms_S(self):
        """Process tdms S."""
        return self.get_jobs_by_type("exc").tdms_S

    def __len__(self):
        """Get number of jobs in this group."""
        return len(self.jobs)

    def __repr__(self):
        """Handle repr internally."""
        return f"GroupedOrcaJob('{self.jobname}', {len(self.jobs)} jobs)"

    def __getattr__(self, name):
        """
        Automatically look for attributes in parsers if they don't exist in this class.
        First tries to find a parser with the attribute name as its type,
        then checks if any parser has the requested attribute.
        """

        out = {}
        for job in self.jobs:
            try:
                result = getattr(job, name, None)
                if result is not None:
                    out[job] = result
            except AttributeError:
                continue
            except ValueError:
                continue
            except KeyError:
                continue
        if len(out) == 0:
            raise AttributeError(
                f"'{self.__class__.__name__}' object and its jobs have no attribute '{name}'"
            )
        if len(out) == 1:
            out = list(out.values())[0]
        return out


def find_all_grouped_jobs(folder):
    """
    Find all Orca jobs and group them by jobname.

    Args:
        folder: Directory to search for jobs

    Returns:
        Dictionary mapping jobnames to GroupedOrcaJob instances
    """

    all_jobs = find_all_jobs(folder)

    job_collection = OrcaJobCollection(all_jobs)

    grouped_jobs = {
        jobname: GroupedOrcaJob.create(jobname, jobs)
        for jobname, jobs in job_collection.jobs_by_name.items()
    }

    return grouped_jobs
