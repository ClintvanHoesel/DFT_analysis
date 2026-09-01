from __future__ import annotations

from typing import TYPE_CHECKING

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap

from dft_utils.units import Units

from .._plams import require_plams

if TYPE_CHECKING:
    from scm.plams import Molecule

ATOM_COLORS = {
    0: (1.0, 0.5490196078431373, 0.0),
    1: (1.0, 1.0, 1.0),
    3: (0.6666666666666666, 0.6666666666666666, 0.6666666666666666),
    4: (0.6705882352941176, 0.6705882352941176, 0.6705882352941176),
    6: (0.3764705882352941, 0.3764705882352941, 0.3764705882352941),
    7: (0.0, 0.0, 1.0),
    8: (1.0, 0.0, 0.0),
    9: (0.0, 1.0, 0.0),
    11: (0.9450980392156862, 0.9450980392156862, 0.9450980392156862),
    12: (0.7019607843137254, 0.7019607843137254, 0.7019607843137254),
    13: (0.6, 0.6, 0.6),
    14: (0.8823529411764706, 0.6627450980392157, 0.37254901960784315),
    15: (1.0, 0.6470588235294118, 0.0),
    16: (1.0, 1.0, 0.0),
    17: (0.5647058823529412, 0.9333333333333333, 0.5647058823529412),
    19: (0.6666666666666666, 0.6666666666666666, 0.6666666666666666),
    22: (0.7137254901960784, 0.6862745098039216, 0.6627450980392157),
    24: (0.9098039215686274, 0.9450980392156862, 0.8313725490196079),
    25: (0.011764705882352941, 0.6588235294117647, 0.6196078431372549),
    26: (0.7176470588235294, 0.2549019607843137, 0.054901960784313725),
    27: (0.0, 0.2784313725490196, 0.6705882352941176),
    28: (0.5019607843137255, 0.5019607843137255, 0.5019607843137255),
    29: (0.8509803921568627, 0.5294117647058824, 0.09803921568627451),
    30: (0.9921568627450981, 0.9725490196078431, 1.0),
    31: (0.7568627450980392, 0.7568627450980392, 0.7568627450980392),
    33: (0.23137254901960785, 0.3254901960784314, 0.23529409803921569),
    34: (0.6980392156862745, 0.13333333333333333, 0.13333333333333333),
    35: (0.6470588235294118, 0.16470588235294117, 0.16470588235294117),
    44: (0.8509803921568627, 0.28627450980392155, 0.4470588235294118),
    47: (0.7529411764705882, 0.7529411764705882, 0.7529411764705882),
    48: (1.0, 0.8901960784313725, 0.011764705882352941),
    50: (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
    53: (0.6274509803921569, 0.12549019607843137, 0.9411764705882353),
    60: (1.0, 0.4117647058823529, 0.7058823529411765),
    74: (0.2, 0.2, 0.2),
    76: (0.0, 1.0, 1.0),
    78: (0.8980392156862745, 0.8941176470588236, 0.8862745098039215),
    79: (1.0, 0.8431372549019608, 0.0),
    80: (0.9019607843137255, 0.9019607843137255, 0.9019607843137255),
    82: (0.09803921568627451, 0.09803921568627451, 0.09803921568627451),
    92: (1.0, 0.0784313725490196, 0.5764705882352941),
    106: (0.19607843137254902, 0.803921568627451, 0.19607843137254902),
    107: (0.0196078431372549, 0.5647058823529412, 0.2),
    108: (0.2980392156862745, 0.7333333333333333, 0.09019607843137255),
}

ATOM_RADII = {
    "Xx": 0.22,
    "Gh": 0.22,
    "El": 0.22,
    "Eh": 0.22,
    "H": 0.30,
    "He": 0.99,
    "Li": 1.52,
    "Be": 1.12,
    "B": 0.88,
    "C": 0.77,
    "N": 0.70,
    "O": 0.66,
    "F": 0.64,
    "Ne": 1.60,
    "Na": 1.86,
    "Mg": 1.60,
    "Al": 1.43,
    "Si": 1.17,
    "P": 1.10,
    "S": 1.04,
    "Cl": 0.99,
    "Ar": 1.92,
    "K": 2.31,
    "Ca": 1.97,
    "Sc": 1.60,
    "Ti": 1.46,
    "V": 1.31,
    "Cr": 1.25,
    "Mn": 1.29,
    "Fe": 1.26,
    "Co": 1.25,
    "Ni": 1.24,
    "Cu": 1.28,
    "Zn": 1.33,
    "Ga": 1.41,
    "Ge": 1.22,
    "As": 1.21,
    "Se": 1.17,
    "Br": 1.14,
    "Kr": 1.97,
    "Rb": 2.44,
    "Sr": 2.15,
    "Y": 1.80,
    "Zr": 1.57,
    "Nb": 1.41,
    "Mo": 1.36,
    "Tc": 1.35,
    "Ru": 1.33,
    "Rh": 1.34,
    "Pd": 1.38,
    "Ag": 1.44,
    "Cd": 1.49,
    "In": 1.66,
    "Sn": 1.62,
    "Sb": 1.41,
    "Te": 1.37,
    "I": 1.33,
    "Xe": 2.17,
    "Cs": 2.62,
    "Ba": 2.17,
    "La": 1.88,
    "Ce": 1.818,
    "Pr": 1.824,
    "Nd": 1.814,
    "Pm": 1.834,
    "Sm": 1.804,
    "Eu": 2.084,
    "Gd": 1.804,
    "Tb": 1.773,
    "Dy": 1.781,
    "Ho": 1.762,
    "Er": 1.761,
    "Tm": 1.759,
    "Yb": 1.922,
    "Lu": 1.738,
    "Hf": 1.57,
    "Ta": 1.43,
    "W": 1.37,
    "Re": 1.37,
    "Os": 1.34,
    "Ir": 1.35,
    "Pt": 1.38,
    "Au": 1.44,
    "Hg": 1.52,
    "Tl": 1.71,
    "Pb": 1.75,
    "Bi": 1.70,
    "Po": 1.40,
    "At": 1.40,
    "Rn": 2.40,
    "Fr": 2.70,
    "Ra": 2.20,
    "Ac": 2.00,
    "Th": 1.79,
    "Pa": 1.63,
    "U": 1.56,
    "Np": 1.55,
    "Pu": 1.59,
    "Am": 1.73,
    "Cm": 1.74,
    "Bk": 1.70,
    "Cf": 1.86,
    "Es": 1.86,
    "Fm": 2.00,
    "Md": 2.00,
    "No": 2.00,
    "Lr": 2.00,
    "Rf": 2.00,
    "Db": 2.00,
    "Sg": 2.00,
    "Bh": 2.00,
    "Hs": 2.00,
    "Mt": 2.00,
    "Ds": 2.00,
    "Rg": 2.00,
    "Cn": 2.00,
    "Nh": 2.00,
    "Fl": 2.00,
    "Mc": 2.00,
    "Lv": 2.00,
    "Ts": 2.00,
    "Og": 2.00,
    "Uue": 2.00,
    "Ubn": 2.00,
    "default": 0.20,
}


class CubeParser:
    def __init__(self, path):
        """Initialize the object."""
        self.path = path
        self.parse()

    def parse(self):
        """Parse the supplied input."""
        Atom, Molecule, _, _, _ = require_plams()
        self._mol = Molecule()
        with open(self.path, "r") as f:
            self._comment1 = next(f).strip()
            self._comment2 = next(f).strip()

            l = next(f).strip().split()
            self._num_atoms = abs(int(l[0]))
            self._min_xyz = np.array([float(x) for x in l[1:]])

            l = next(f).strip().split()
            self._nx = int(l[0])
            self._lvecx = np.array([float(x) for x in l[1:]])

            l = next(f).strip().split()
            self._ny = int(l[0])
            self._lvecy = np.array([float(x) for x in l[1:]])

            l = next(f).strip().split()
            self._nz = int(l[0])
            self._lvecz = np.array([float(x) for x in l[1:]])

            self._boxvecs = np.array([self._lvecx[0], self._lvecy[1], self._lvecz[2]])

            self._atoms = []
            for _ in range(self._num_atoms):
                l = next(f).strip().split()
                atnum = int(l[0])
                coords = np.array([float(x) for x in l[2:]])
                coords = Units.convert(coords, "au", "Angstrom")
                atom = Atom(atnum=atnum, coords=coords)
                self._mol.add_atom(atom)
                self._atoms.append(atom)
        self._mol.guess_bonds()

    @property
    def grid(self):
        """Process grid."""
        spacing = self.boxvecs
        dimensions = np.array([self.nx, self.ny, self.nz])
        origin = self.min_xyz

        grid = pv.read(str(self.path))
        grid.dimensions = dimensions
        grid.origin = origin
        grid.spacing = spacing

        return grid

    @property
    def molecule(self):
        """Process molecule."""
        return self._mol

    @property
    def num_atoms(self):
        """Process num atoms."""
        return self._num_atoms

    @property
    def min_xyz(self):
        """Process min xyz."""
        return self._min_xyz

    @property
    def nx(self):
        """Process nx."""
        return self._nx

    @property
    def ny(self):
        """Process ny."""
        return self._ny

    @property
    def nz(self):
        """Process nz."""
        return self._nz

    @property
    def lvecx(self):
        """Process lvecx."""
        return self._lvecx

    @property
    def lvecy(self):
        """Process lvecy."""
        return self._lvecy

    @property
    def lvecz(self):
        """Process lvecz."""
        return self._lvecz

    @property
    def boxvecs(self):
        """Process boxvecs."""
        return self._boxvecs

    @property
    def atoms(self):
        """Process atoms."""
        return self._atoms

    @property
    def comments(self):
        """Process comments."""
        return (self._comment1, self._comment2)

    def plot_cube_as_volume(self, **kwargs):
        """Process plot cube as volume."""
        return plot_cube_volume(self.path, **kwargs)


class Plotter:
    def __init__(self, **kwargs):
        """Initialize the object."""
        self.pk = kwargs

        self.pl = pv.Plotter(**self.pk)
        self.pl.set_background("white")
        self.pl.camera_position = "xy"
        self.vols = []
        self.meshes = []
        self.atoms = {}
        self.bonds = {}

    def add_mesh(self, mesh, **kwargs):
        """Process add mesh."""
        mesh = self.pl.add_mesh(mesh, **kwargs)
        self.meshes.append(mesh)
        return mesh

    def plot_volume(self, grid, **kwargs):
        """Process plot volume."""
        vol = self.pl.add_volume(grid, **kwargs)
        vol.prop.interpolation_type = "linear"
        self.vols.append(vol)
        return vol

    def plot_contour(self, grid, isovalue, **kwargs):
        """Process plot contour."""
        surf = grid.contour(isovalue)
        self.add_mesh(surf, **kwargs)
        return surf

    def plot_molecule(
        self,
        molecule: Molecule,
        bond_radius: float = 0.1,
        atom_colors={},
        atom_radii={},
        atom_kwargs={},
        bond_kwargs={},
    ):
        """
        Plot atoms as spheres and bonds as tubes using ATOM_COLORS mapping.
        """

        atom_colors = ATOM_COLORS | atom_colors
        atom_radii = ATOM_RADII | atom_radii
        for atom in molecule.atoms:
            atnum = atom.atnum
            center = atom.coords
            center = Units.convert(center, "Angstrom", "au")
            color = atom_colors.get(atnum, atom_colors[0])
            symbol = atom.symbol
            radius = atom_radii.get(symbol, atom_radii["default"])

            sphere = pv.Sphere(center=center, radius=radius)
            self.atoms[atom] = sphere
            self.add_mesh(sphere, color=color, **atom_kwargs)

        for bond in molecule.bonds:
            a1 = bond.atom1.coords
            a2 = bond.atom2.coords
            a1 = Units.convert(a1, "Angstrom", "au")
            a2 = Units.convert(a2, "Angstrom", "au")
            line = pv.Line(a1, a2)
            tube = line.tube(radius=bond_radius)
            self.bonds[bond] = tube
            self.add_mesh(tube, color="white", **bond_kwargs)

    def show(self, **kwargs):
        """Process show."""
        self.pl.show(**kwargs)


def truncate_cmap(cmap, minval=0.25, maxval=0.75, n=256):
    """Return a copy of `cmap` restricted to [minval, maxval] in its 0..1 domain."""
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_colors = cmap(np.linspace(minval, maxval, n))
    return LinearSegmentedColormap.from_list(
        f"{getattr(cmap, 'name', 'cmap')}_trunc_{minval:.2f}_{maxval:.2f}", new_colors
    )


def plot_cube_volume(
    cube_path,
    windows_size=[1200 * 2, 800 * 2],
    max_color=1.0,
    threshold=0.0,
    maxval=0.03,
    tcmap=cmc.vik,
    alpha=0.6,
    power=2.0,
    silhouette=dict(
        color="black",
        line_width=8.0,
        opacity=1.0,
    ),
    zoom=2.0,
    elevation=-10,
):
    """Process plot cube volume."""
    cp = CubeParser(cube_path)
    pl = Plotter(off_screen=True, window_size=windows_size)

    tcmap = truncate_cmap(tcmap, 0.5 * (1.0 - max_color), 0.5 * (1.0 + max_color))

    lut = pv.LookupTable(
        cmap=tcmap,
        scalar_range=(-maxval, maxval),
        above_range_color=tcmap(1.0),
        below_range_color=tcmap(0.0),
    )
    xopacs = np.linspace(-maxval, maxval, 251)
    normalized_x = np.abs(xopacs / maxval)

    opacs = np.where(
        normalized_x < threshold, 0, np.abs(normalized_x - threshold) ** power
    )
    opacs = np.clip(opacs, 0, 1.0) * alpha
    lut.apply_opacity(opacs, kind="linear")

    molactor = pl.plot_molecule(cp.molecule)

    volactor = pl.plot_volume(
        cp.grid,
        cmap=lut,
        show_scalar_bar=False,
    )

    pl.pl.camera_position = "xy"
    pl.pl.reset_camera()
    pl.pl.camera.zoom(zoom)
    pl.pl.camera.elevation = elevation

    pl.show(screenshot=cube_path + ".png")
    return pl
