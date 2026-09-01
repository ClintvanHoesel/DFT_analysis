import struct
from functools import cache, cached_property

import numpy as np

from .FileParser import FileParser


def parse_cis(file):
    """Parse the supplied input."""
    n_orb = 205
    n_el = 26
    n_occ = n_el // 2
    n_virt = n_orb - n_occ
    extra = 0
    n = int(n_occ * (n_virt) + extra)
    res = []
    with open(file, "rb") as f:

        init = struct.unpack(f"<{9}i", f.read(4 * 9))
        n_excitons = init[0]

        n_occa = init[3]
        n_orba = init[4] + 1
        n_virta = n_orba - n_occa

        n_occb = init[7]
        n_orbb = init[8] + 1
        n_virtb = n_orbb - n_occb
        if int(n_orbb) == 0:
            restricted = True
        else:
            restricted = False

        nelema = n_virta * n_occa
        nelemb = n_virtb * n_occb
        if restricted:
            nelemtot = nelema
        else:
            nelemtot = nelema + nelemb
        for _i in range(n_excitons):
            tmp2 = struct.unpack(f"<{6}i", f.read(4 * 6))
            n_elem = tmp2[0]
            assert n_elem == nelemtot
            mult = tmp2[2]
            iroot = tmp2[4]
            en = struct.unpack(f"<{1}d", f.read(8 * 1))
            tmp = struct.unpack(f"<{1}d", f.read(8 * 1))
            out = struct.unpack(f"<{n_elem}d", f.read(8 * n_elem))
            if restricted:
                invsqrt2 = 1.0 / np.sqrt(2)
                outa = np.array(out).reshape((n_occa, n_virta)) * invsqrt2
                outb = outa
            else:
                outa = np.array(out[:nelema]).reshape((n_occa, n_virta))
                outb = np.array(out[nelema:]).reshape((n_occb, n_virtb))

            res.append(
                [en, tmp, tmp2, np.sum(np.array(out) ** 2), outa, outb, mult, iroot]
            )
    return restricted, res


class ROCISParser(FileParser):
    def parse(self):
        """Parse the supplied input."""
        with open(self.path, "rb") as f:
            n1, n2 = struct.unpack(f"<{2}i", f.read(4 * 2))

            n2list = struct.unpack(f"<{n2}d", f.read(8 * n2))

            cislist = []
            for _i in range(n2):
                cislist.append(struct.unpack(f"<{n1}d", f.read(8 * n1)))
        return n1, n2, n2list, cislist

    def get_data(self, nDOMO: int, nSOMO: int, nVMO: int):
        """Return data."""
        out = []
        inp = self.parse()
        for en, exc in zip(inp[2], inp[2]):
            phirest = np.array(exc)
            print(np.sum(phirest**2))
            nnext = 1
            phig = phirest[:nnext]
            phirest = phirest[nnext:]
            nnext = nDOMO * nSOMO
            phids = phirest[:nnext]
            phirest = phirest[nnext:]
            nnext = nVMO * nSOMO
            phisv = phirest[:nnext]
            phirest = phirest[nnext:]
            nnext = nVMO * nSOMO * nDOMO
            phidssv = phirest[:nnext].reshape((nDOMO, nSOMO, nVMO))
            phirest = phirest[nnext:]
            nnext = nVMO * nDOMO
            phidssfv = phirest[:nnext].reshape((nDOMO, nVMO))
            phirest = phirest[nnext:]
            out.append(en, phig, phids, phisv, phidssv, phidssfv)
        return out


class CISParser(FileParser):
    @cache
    def _parse_all(self):
        """Handle parse all internally."""
        return parse_cis(self.path)

    def parse(self):
        """Parse the supplied input."""
        return self._parse_all()[1]

    @property
    def restricted(self):
        """Process restricted."""
        return bool(self._parse_all()[0])

    @property
    def cis_data(self):
        """Process cis data."""
        data = self.parse()
        return [[v[0], v[6], v[4], v[5]] for v in data]

    def get_en_mult(self, mult=None):
        """Return en mult."""
        data = self.cis_data
        if mult is not None:
            data = [v for v in data if v[1] == mult]
        data = np.array([v[0][0] for v in data])
        return data

    @property
    def singlet_en(self):
        """Process singlet en."""
        return self.get_en_mult(mult=1)

    @property
    def triplet_en(self):
        """Process triplet en."""
        return self.get_en_mult(mult=3)

    @property
    def all_en(self):
        """Process all en."""
        return self.get_en_mult(mult=None)

    def get_x_mult(self, mult=None):
        """Return x mult."""
        data = self.cis_data
        if mult is not None:
            data = [v for v in data if v[1] == mult]
        if self.restricted:
            data = [v[2] for v in data]
        else:
            data = [[v[2], v[3]] for v in data]
        return data

    @cached_property
    def singlet_xy(self):
        """Process singlet xy."""
        return self.get_x_mult(mult=1)

    @cached_property
    def triplet_xy(self):
        """Process triplet xy."""
        return self.get_x_mult(mult=3)

    @property
    def all_xy(self):
        """Process all xy."""
        return self.get_x_mult(mult=None)
