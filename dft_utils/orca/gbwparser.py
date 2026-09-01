import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .._plams import require_plams
from .FileParser import FileParser


def _is_settings(value):
    """Return whether *value* is a PLAMS ``Settings`` instance."""
    try:
        settings_type = require_plams()[2]
    except ImportError:
        return False
    return isinstance(value, settings_type)


def get_end(s):
    """Return end."""
    if (not _is_settings(s)) or ("_end" not in s):
        return s
    else:
        return "{} end".format(s["_end"])


def pretty_print_inner(s, indent):
    """Process pretty print inner."""
    inp = ""
    for i, (key, value) in enumerate(s.items()):
        end = get_end(value)
        if i == 0:
            inp += " {} {}\n".format(key, end)
        else:
            inp += "{}{} {}\n".format(indent, key, end)
    return inp


def pretty_print_orca(s, indent="", print_main=False):
    """Set print_main to true for initial call to have main section
    at the top
    """
    inp = ""
    if print_main:
        inp += "! {}\n\n".format(pretty_print_orca(s.main))
        pretty_print_orca(s, indent)
    if _is_settings(s):
        for k, v in s.items():
            if k in ("main", "molecule"):

                continue
            else:
                indent2 = (len(k) + 2) * " "
                if not _is_settings(v):
                    inp += "%{} {}\n\n".format(k, pretty_print_orca(v))
                else:
                    block = pretty_print_inner(v, indent2)
                    inp += "%{}{}{}end\n\n".format(k, block, indent)
    elif isinstance(s, list):
        inp += "{}{}".format(indent, " ".join(s))
    else:
        inp += "{}{}".format(indent, s)
    return inp


def print_molecule(mol):
    """Print a molecule in the ORCA format using the xyz notation."""
    if "charge" in mol.properties and isinstance(mol.properties.charge, int):
        charge = mol.properties.charge
    else:
        charge = 0
    if "multiplicity" in mol.properties and isinstance(
        mol.properties.multiplicity, int
    ):
        multi = mol.properties.multiplicity
    else:
        multi = 1

    xyz = "\n".join(at.str(symbol=True, space=21, decimal=14) for at in mol.atoms)
    return "* xyz {} {}\n{}\n*\n\n".format(charge, multi, xyz)


class GBWParser(FileParser):
    def _perform_rescue_run(self, mol):
        """Handle perform rescue run internally."""
        gbw_name = self.path.name
        gbw_pathparent = self.path.parent
        gbw_path = os.path.abspath(self.path)
        basename = os.path.splitext(gbw_path)[0]
        bbasename = os.path.basename(basename)
        new_inp_path = basename + ".inp"
        xyz_path = basename + ".xyz"

        with open(os.path.join(gbw_pathparent, xyz_path), "w") as f:

            mol.writexyz(f)

        if "_0_" in gbw_name:
            charge = 0
            spin = 1
        elif "_1_" in gbw_name:
            charge = 1
            spin = 2
        elif "_-1_" in gbw_name:
            charge = -1
            spin = 2
        else:
            raise ValueError("GBW could not find charge.")
        mol.properties.charge = charge
        mol.properties.multiplicity = spin
        text = f'! RESCUE NOITER\r\n%moinp "{gbw_name}" \r\n{print_molecule(mol)}'
        with open(new_inp_path, "w") as f:
            f.write(text)
        orca = shutil.which("orca")
        if not orca:
            raise RuntimeError("orca not in PATH")
        cmd = [orca, f"{bbasename}.inp"]
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
            cwd=gbw_pathparent,
        )

    def _convert_molden(self):
        """Handle convert molden internally."""
        gbw_path = os.path.abspath(self.path)
        basename = os.path.splitext(gbw_path)[0]

        cmd = """
        $(dirname "$(which orca_2mkl)")/orca_2mkl {base} -molden
        """.format(base=basename)
        completed = subprocess.run(
            ["bash", "-lc", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        return completed

    def to_molden(self, remove=False):
        """Process to molden."""
        from pyscf import gto, scf
        from pyscf.tools import molden

        self._convert_molden()

        mol = gto.Mole()
        mol.verbose = 0
        mol.output = None
        mol.build()

        molden.from_molden(mol, self.molden_file)
        mf = scf.RHF(mol)

        if remove:
            gbw_path = os.path.abspath(self.path)
            basename = os.path.splitext(gbw_path)[0]
            molden_file = basename + ".molden.input"
            os.remove(self.molden_file)

        return mol, mf

    def _convert_json(self, cheap=False):
        """Handle convert json internally."""
        import json

        gbw_pathparent = self.path.parent
        gbw_path = os.path.abspath(self.path)
        basename = os.path.splitext(gbw_path)[0]
        bbasename = os.path.basename(basename)

        if "_0_" in bbasename:
            dosoc = True
        elif "_1_" in bbasename:
            dosoc = False
        elif "_-1_" in bbasename:
            dosoc = False
        else:
            dosoc = False
            print("Could not resolve charge")

        if cheap:
            D = {
                "MOCoefficients": True,
                "Basisset": False,
                "1elIntegrals": ["S"],
                "1elPropertyIntegrals": [
                    "dipole",
                    "quadrupole",
                ],
                "1elPropertyRelIntegrals": [
                    "dipole",
                    "quadrupole",
                ],
                "LoewdinCharge": False,
                "CIS": False,
                "CISNRoots": False,
                "JSONFormats": ["json"],
            }
        else:
            D = {
                "MOCoefficients": True,
                "Basisset": False,
                "1elIntegrals": ["H", "HMO", "S", "T", "V"],
                "1elPropertyIntegrals": [
                    "dipole",
                    "quadrupole",
                    "velocity",
                    "angular_momentum",
                    "higherMoment",
                ],
                "1elPropertyRelIntegrals": [
                    "dipole",
                    "quadrupole",
                    "velocity",
                ],
                "ori_el": 1,
                "LoewdinCharge": False,
                "CIS": False,
                "CISNRoots": False,
                "JSONFormats": ["json"],
            }
        if dosoc:
            D["1elPropertyIntegrals"].append("soc")

        output_path = os.path.join(gbw_pathparent, "orca.json.conf")
        with open(output_path, "w") as f:
            json.dump(D, f, indent=4)

        orca2json = shutil.which("orca_2json")
        if not orca2json:
            raise RuntimeError("orca_2json not in PATH")
        cmd = [orca2json, f"{bbasename}.gbw"]
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
            cwd=gbw_pathparent,
        )

    def parse(self):
        """Parse the supplied input."""
        raise NotImplementedError()
        return self.parse_gbw2(self.path)

        with open(self.path) as f:
            return f.read()

    def parse_gbw2(self, file_name):
        """Parse the supplied input."""
        out = dict()
        with open(file_name, "rb") as f:
            f.seek(24)
            offset = struct.unpack("<q", f.read(8))[0]
            f.seek(offset)
            operators = struct.unpack("<i", f.read(4))[0]
            dimension = struct.unpack("<i", f.read(4))[0]

            out["offset"] = offset
            out["operators"] = operators
            out["dimension"] = dimension

            print("Offset: {}".format(offset))
            print("Number of Operators: {}".format(operators))
            print("Basis Dimension: {}".format(dimension))

            for i in range(operators):
                op_out = dict()

                print("\nOperator: {}".format(i))
                coefficients = np.array(
                    list(struct.iter_unpack("<d", f.read(8 * dimension**2)))
                )
                occupations = np.array(
                    list(struct.iter_unpack("<d", f.read(8 * dimension)))
                )
                energies = np.array(
                    list(struct.iter_unpack("<d", f.read(8 * dimension)))
                )
                irreps = np.array(list(struct.iter_unpack("<i", f.read(4 * dimension))))
                cores = np.array(list(struct.iter_unpack("<i", f.read(4 * dimension))))

                op_out["coefficients"] = coefficients.reshape(
                    (
                        int(np.sqrt(coefficients.shape[0])),
                        int(np.sqrt(coefficients.shape[0])),
                    )
                )
                op_out["occupations"] = occupations[:, 0]
                op_out["energies"] = energies[:, 0]
                op_out["irreps"] = irreps
                op_out["cores"] = cores

                out[f"Op{i}"] = op_out

        with open(file_name, "rb") as f:
            f.seek(8)
            offset = struct.unpack("<q", f.read(8))[0]
            f.seek(offset)
            atoms = struct.unpack("<i", f.read(4))[0]
            n_atoms = atoms

            atom_data_floats = []
            weird_data = []
            for i in range(atoms):
                float_data = struct.unpack("<dddddd", f.read(8 * 6))
                weird = list(struct.iter_unpack("<b", f.read(20)))
                atom_data_floats.append(float_data)
                weird_data.append(weird)

        out["geometry"] = np.asarray(atom_data_floats).reshape((n_atoms, -1))
        out["x"] = out["geometry"][:, 0]
        out["y"] = out["geometry"][:, 1]
        out["z"] = out["geometry"][:, 2]
        out["charge"] = out["geometry"][:, 3]
        out["mass"] = out["geometry"][:, 5]

        out["weird"] = np.asarray(weird_data).reshape((n_atoms, -1))
        out["atom_type"] = out["weird"][:, 0]
        return out

    def run_orca_plot_file(self, sett):
        """Process run orca plot file."""
        gbw_pathparent = self.path.parent
        output_path = os.path.join(gbw_pathparent, "plot.inp")
        text = pretty_print_orca(sett)
        with open(output_path, "w") as f:
            f.write(text)
        subprocess.run([""])
        subprocess.run(
            ["$(which orca_plot)", str(self.path), str(output_path)],
            text=True,
            check=True,
        )

    def run_orca_plot_interactive(self, commands, cwd=None, cores=None):
        """Process run orca plot interactive."""
        if cwd is None:
            cwd = self.path.parent

        if cores is None or cores <= 1:
            executable = shutil.which("orca_plot")
            if not executable:
                raise FileNotFoundError(
                    "The 'orca_plot' executable was not found in your PATH."
                )
            full_command = [executable, str(self.path), "-i"]
            return subprocess.run(
                full_command,
                input=commands,
                text=True,
                check=True,
                cwd=cwd,
            )
        else:

            assert isinstance(cores, int)
            executable = shutil.which("orca_plot_mpi")
            if not executable:
                raise FileNotFoundError(
                    "The 'orca_plot_mpi' executable was not found in your PATH."
                )

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".inp", dir=cwd, delete=False
            ) as tmp:
                tmp.writelines(commands)
                tmp_path = tmp.name

            full_command = [
                "mpirun",
                "-np",
                str(cores),
                executable,
                str(self.path),
                "-i",
                tmp_path,
            ]
            out = None
            try:
                out = subprocess.run(
                    full_command,
                    input=commands,
                    text=True,
                    check=True,
                    cwd=cwd,
                )
            finally:
                Path(tmp_path).unlink(missing_ok=True)
            return out

    def run_orca_plotmo(self, typ=1, mo_list=[0], op=0, intv=40, ff=7, **kwargs):
        """Process run orca plotmo."""
        if len(mo_list) == 0:
            return None
        commands_before = [
            "1",
            str(typ),
        ]
        commands_after = [
            "3",
            str(op),
            "4",
            str(intv),
            "5",
            str(ff),
            "11",
        ]
        commands_end = ["12", ""]
        commands_list = commands_before
        for mo in mo_list:
            commands_list += [
                "2",
                str(mo),
            ] + commands_after
        commands_list += commands_end
        commands = "\n".join(commands_list)
        return self.run_orca_plot_interactive(commands=commands, **kwargs)

    def run_orca_plotexc(self, exc_list=[1], intv=40, ff=7, diff=True, **kwargs):
        """Process run orca plotexc."""
        if len(exc_list) == 0:
            return None
        states = " ".join(list(map(str, exc_list)))
        if diff:
            mode = 6
        else:
            mode = 7
        commands_list = [
            "4",
            str(intv),
            "5",
            str(ff),
            str(mode),
            "y",
            str(states),
            "12",
            "",
        ]
        commands = "\n".join(commands_list)
        return self.run_orca_plot_interactive(commands=commands, **kwargs)

    def plot_orbital(cube_file, isovalue, color, screenshot_file):
        """
        Loads a cube file in PyVista, extracts isosurface, and renders to an image.
        """
        try:
            import pyvista as pv
        except ModuleNotFoundError:
            print(
                "PyVista is required for plotting. Install it with: pip install pyvista"
            )
            """
            Runs orca_plot to generate a Gaussian cube file for the specified orbital.
            """
        grid = pv.read(str(cube_file))
        surf = grid.contour([abs(isovalue)])
        p = pv.Plotter(off_screen=True, window_size=[1200, 1200])
        p.add_mesh(surf, color=color, opacity=0.8, specular=0.5, smooth_shading=True)
        p.set_background("white")
        p.camera_position = "xy"
        p.show(screenshot=screenshot_file)
        p.close()
