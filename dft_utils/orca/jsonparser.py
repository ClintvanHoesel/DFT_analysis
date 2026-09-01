import json
import pickle
import zlib
from decimal import Decimal
from functools import cached_property, wraps
from pathlib import Path
from typing import Any, Iterator, Union

import ijson
import numpy as np

from .FileParser import FileParser

try:
    from pyscf.lib import einsum
except ImportError:
    print("Could not find PySCF einsum. Falling back to numpy.")
    from numpy import einsum


def _convert_decimal_number(obj: Any) -> Union[int, float]:
    """Fast conversion for numeric values only"""
    if isinstance(obj, Decimal):
        if obj == obj.to_integral_value():
            return int(obj)
        return float(obj)
    return obj


def _convert_decimals(obj: Any) -> Any:
    """Optimized recursive conversion with type-specific handling"""
    obj_type = type(obj)

    if obj_type is Decimal:
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    elif obj_type is dict:
        return {k: _convert_decimals(v) for k, v in obj.items()}
    elif obj_type is list:
        return [_convert_decimals(v) for v in obj]
    elif obj_type is tuple:
        return tuple(_convert_decimals(v) for v in obj)
    else:
        return obj


def _stream_array(path: Path, prefix: str):
    """
    Helper generator: streams a JSON array at the given prefix.
    Yields each element in the array with decimals converted.
    """
    with open(path, "rb") as f:
        for item in ijson.items(f, prefix + ".item"):
            yield _convert_decimals(item)


def _stream_to_list(path: Path, prefix: str):
    """Handle stream to list internally."""
    return [x for x in _stream_array(path, prefix)]


def _load_scalar(path: Path, dotted_key: str) -> Any:
    """Handle load scalar internally."""
    with open(path, "rb") as f:
        parser = ijson.parse(f)
        parts = dotted_key.split(".")
        stack = []
        for prefix, event, value in parser:
            if event in ("start_map", "end_map", "start_array", "end_array"):
                continue
            keys = prefix.split(".")
            if keys == parts:
                return _convert_decimals(value)
    return None


def stream_specific_field(
    path: Path, prefix: str, field: str = None, buffer_size: int = 64 * 1024
) -> Iterator[Any]:
    """
    Stream only a specific field from array items - much more efficient.
    """
    if field:
        full_prefix = f"{prefix}.item.{field}"
    else:
        full_prefix = f"{prefix}.item"
    with open(path, "rb", buffering=buffer_size) as f:
        for value in ijson.items(f, full_prefix):
            yield _convert_decimals(value)


def stream_to_numpy(
    path: Path,
    prefix: str,
    field: str = None,
    dtype=np.float64,
    buffer_size: int = 64 * 1024,
) -> np.ndarray:
    """
    Stream directly to numpy array for better memory efficiency.
    """
    values = list(stream_specific_field(path, prefix, field, buffer_size))
    return np.array(values, dtype=dtype)


def compressed_cached_property(func, level=1):
    """Process compressed cached property."""
    attr = f"_{func.__name__}_compressed"

    @property
    @wraps(func)
    def wrapper(self):
        """Process wrapper."""
        if hasattr(self, attr):
            comp = getattr(self, attr)
        else:
            val = func(self)
            ser = pickle.dumps(val, protocol=pickle.HIGHEST_PROTOCOL)
            comp = zlib.compress(ser, level=level)
            setattr(self, attr, comp)
        return pickle.loads(zlib.decompress(comp))

    return wrapper


class iJSONParser(FileParser):
    """
    Streaming JSON parser using ijson. Properties directly stream and convert relevant sections.
    """

    def __init__(self, path: Union[str, Path]):
        """Initialize the object."""
        super().__init__(path)
        self._path = Path(path)

    def parse(self):
        """Parse the supplied input."""
        return None

    @cached_property
    def mo_coeff(self) -> np.ndarray:
        """Process mo coeff."""
        group = "Molecule.MolecularOrbitals.MOs"
        field = "MOCoefficients"
        return stream_to_numpy(self.path, group, field, dtype=np.float64).T

    @cached_property
    def mo_occ(self) -> np.ndarray:
        """Process mo occ."""
        group = "Molecule.MolecularOrbitals.MOs"
        field = "Occupancy"
        return stream_to_numpy(self.path, group, field, dtype=np.float64)

    @cached_property
    def mo_energy(self) -> np.ndarray:
        """Process mo energy."""
        group = "Molecule.MolecularOrbitals.MOs"
        field = "OrbitalEnergy"
        return stream_to_numpy(self.path, group, field, dtype=np.float64)

    @cached_property
    def dipole_AO(self) -> np.ndarray:
        """Process dipole AO."""
        group = "Molecule.dipole"

        return stream_to_numpy(self.path, group, dtype=np.float64)

    @property
    def dipoleRel_AO(self) -> np.ndarray:
        """Process dipoleRel AO."""
        group = "Molecule.dipoleRel"
        return stream_to_numpy(self.path, group, dtype=np.float64)

    @cached_property
    def quadrupole_AO(self) -> np.ndarray:
        """Process quadrupole AO."""
        group = "Molecule.quadrupole"
        return stream_to_numpy(self.path, group, dtype=np.float64)

    @property
    def quadrupoleRel_AO(self) -> np.ndarray:
        """Process quadrupoleRel AO."""
        group = "Molecule.quadrupoleRel"
        return stream_to_numpy(self.path, group, dtype=np.float64)

    @property
    def angular_momentum(self):
        """Process angular momentum."""
        group = "Molecule.angularMomentum"
        return stream_to_numpy(self.path, group, dtype=np.float64)

    @property
    def electric_octupole_length(self):
        """Process electric octupole length."""
        group = "Molecule.electricOctupoleLength"
        return stream_to_numpy(self.path, group, dtype=np.float64)

    @property
    def electric_octupole_velocity(self):
        """Process electric octupole velocity."""
        group = "Molecule.electricOctupoleVelocity"
        return stream_to_numpy(self.path, group, dtype=np.float64)

    @property
    def electric_quadrupole_velocity(self):
        """Process electric quadrupole velocity."""
        group = "Molecule.electricQuadrupoleVelocity"
        return stream_to_numpy(self.path, group, dtype=np.float64)

    @property
    def magnetic_quadrupole(self):
        """Process magnetic quadrupole."""
        group = "Molecule.magneticQuadrupole"
        return stream_to_numpy(self.path, group, dtype=np.float64)

    @property
    def velocity(self):
        """Process velocity."""
        group = "Molecule.velocity"
        return stream_to_numpy(self.path, group, dtype=np.float64)

    @property
    def soc(self):
        """Process soc."""
        group = "Molecule.soc"
        return stream_to_numpy(self.path, group, dtype=np.float64)

    @property
    def socRel(self):
        """Process socRel."""
        group = "Molecule.socRel"
        return stream_to_numpy(self.path, group, dtype=np.float64)

    @property
    def overlap_AO(self) -> np.ndarray:
        """Process overlap AO."""
        group = "Molecule.S-Matrix"
        return stream_to_numpy(self.path, group, dtype=np.float64)

    @property
    def atoms(self) -> Any:
        """Process atoms."""
        group = "Molecule.Atoms"
        return list(stream_specific_field(self.path, group))

    @cached_property
    def base_name(self) -> Any:
        """Process base name."""
        return _load_scalar(self._path, "Molecule.BaseName")

    @cached_property
    def charge(self) -> Any:
        """Process charge."""
        return _load_scalar(self._path, "Molecule.Charge")

    @cached_property
    def coordinate_units(self) -> Any:
        """Process coordinate units."""
        return _load_scalar(self._path, "Molecule.CoordinateUnits")

    @property
    def s_matrix(self) -> np.ndarray:
        """Process s matrix."""
        group = "Molecule.S-Matrix"
        return stream_to_numpy(self.path, group, dtype=np.float64)

    @property
    def h_matrix(self) -> np.ndarray:
        """Process h matrix."""
        group = "Molecule.H-Matrix"
        return stream_to_numpy(self.path, group, dtype=np.float64)

    @property
    def t_matrix(self) -> np.ndarray:
        """Process t matrix."""
        group = "Molecule.T-Matrix"
        return stream_to_numpy(self.path, group, dtype=np.float64)

    @property
    def v_matrix(self) -> np.ndarray:
        """Process v matrix."""
        group = "Molecule.V-Matrix"
        return stream_to_numpy(self.path, group, dtype=np.float64)

    @property
    def hmo(self) -> np.ndarray:
        """Process hmo."""
        group = "Molecule.HMO"
        return stream_to_numpy(self.path, group, dtype=np.float64)

    @property
    def f_matrix(self) -> np.ndarray:
        """Process f matrix."""
        group = "Molecule.F-Matrix"
        return stream_to_numpy(self.path, group, dtype=np.float64)

    @property
    def j_matrix(self) -> np.ndarray:
        """Process j matrix."""
        group = "Molecule.J-Matrix"
        return stream_to_numpy(self.path, group, dtype=np.float64)

    @property
    def k_matrix(self) -> np.ndarray:
        """Process k matrix."""
        group = "Molecule.K-Matrix"
        return stream_to_numpy(self.path, group, dtype=np.float64)

    @property
    def vxc_matrix(self) -> np.ndarray:
        """Process vxc matrix."""
        group = "Molecule.VXC-Matrix"
        return stream_to_numpy(self.path, group, dtype=np.float64)

    @property
    def vsol_matrix(self) -> np.ndarray:
        """Process vsol matrix."""
        group = "Molecule.Vsol-Matrix"
        return stream_to_numpy(self.path, group, dtype=np.float64)

    @cached_property
    def hf_type(self) -> Any:
        """Process hf type."""
        return _load_scalar(self._path, "Molecule.HFTyp")

    @cached_property
    def multiplicity(self) -> Any:
        """Process multiplicity."""
        return _load_scalar(self._path, "Molecule.Multiplicity")

    @cached_property
    def origin(self) -> np.ndarray:
        """Process origin."""
        group = "Molecule.Origin"
        return stream_to_numpy(self.path, group, dtype=np.float64)

    @cached_property
    def point_group(self) -> Any:
        """Process point group."""
        return _load_scalar(self._path, "Molecule.PointGroup")

    @property
    def static_electric_dipole_moment(self) -> np.ndarray:
        """Process static electric dipole moment."""
        return einsum(
            "i,ji,ki,xjk->x",
            self.mo_occ,
            self.mo_coeff.conj(),
            self.mo_coeff,
            self.dipole_AO,
        )

    def has_property(self, prop_name: str) -> bool:
        """Process has property."""
        val = _load_scalar(self._path, f"Molecule.{prop_name}")
        return val is not None

    def get_property(self, prop_name: str, default=None):
        """Return property."""
        val = _stream_to_list(self._path, f"Molecule.{prop_name}")
        if val is None:
            return default
        return val


class JSONParser(FileParser):
    def parse(self):
        """Parse the supplied input."""
        with open(self.path, "r") as f:
            data = json.load(f)

        return data

    @compressed_cached_property
    def data(self):
        """Process data."""
        return self.parse()

    @cached_property
    def mo_coeff(self):
        """Process mo coeff."""
        mo_coeff = [
            v["MOCoefficients"]
            for v in self.data["Molecule"]["MolecularOrbitals"]["MOs"]
        ]
        return np.array(mo_coeff).T

    @cached_property
    def mo_occ(self):
        """Process mo occ."""
        mo_coeff = [
            v["Occupancy"] for v in self.data["Molecule"]["MolecularOrbitals"]["MOs"]
        ]
        return np.array(mo_coeff)

    @cached_property
    def mo_energy(self):
        """Process mo energy."""
        mo_coeff = [
            v["OrbitalEnergy"]
            for v in self.data["Molecule"]["MolecularOrbitals"]["MOs"]
        ]
        return np.array(mo_coeff)

    @property
    def dipole_AO(self):
        """Process dipole AO."""
        return np.array(self.data["Molecule"]["dipole"])

    @property
    def dipoleRel_AO(self):
        """Process dipoleRel AO."""
        return np.array(self.data["Molecule"]["dipoleRel"])

    @property
    def quadrupole_AO(self):
        """Process quadrupole AO."""
        return np.array(self.data["Molecule"]["quadrupole"])

    @property
    def quadrupoleRel_AO(self):
        """Process quadrupoleRel AO."""
        return np.array(self.data["Molecule"]["quadrupoleRel"])

    @property
    def overlap_AO(self):
        """Process overlap AO."""
        return np.array(self.data["Molecule"]["S-Matrix"])

    @property
    def atoms(self):
        """Process atoms."""
        return self.data["Molecule"]["Atoms"]

    @cached_property
    def base_name(self):
        """Process base name."""
        return self.data["Molecule"]["BaseName"]

    @cached_property
    def charge(self):
        """Process charge."""
        return self.data["Molecule"]["Charge"]

    @property
    def coordinate_units(self):
        """Process coordinate units."""
        return self.data["Molecule"]["CoordinateUnits"]

    @property
    def h_matrix(self):
        """Process h matrix."""
        return np.array(self.data["Molecule"]["H-Matrix"])

    @property
    def hf_type(self):
        """Process hf type."""
        return self.data["Molecule"]["HFTyp"]

    @property
    def hmo(self):
        """Process hmo."""
        return self.data["Molecule"]["HMO"]

    @property
    def molecular_orbitals(self):
        """Process molecular orbitals."""
        return self.data["Molecule"]["MolecularOrbitals"]

    @cached_property
    def multiplicity(self):
        """Process multiplicity."""
        return self.data["Molecule"]["Multiplicity"]

    @cached_property
    def origin(self):
        """Process origin."""
        return np.array(self.data["Molecule"]["Origin"])

    @cached_property
    def point_group(self):
        """Process point group."""
        return self.data["Molecule"]["PointGroup"]

    @property
    def s_matrix(self):
        """Process s matrix."""
        return np.array(self.data["Molecule"]["S-Matrix"])

    @property
    def t_matrix(self):
        """Process t matrix."""
        return np.array(self.data["Molecule"]["T-Matrix"])

    @property
    def td_dft(self):
        """Process td dft."""
        return self.data["Molecule"]["TD-DFT"]

    @property
    def v_matrix(self):
        """Process v matrix."""
        return np.array(self.data["Molecule"]["V-Matrix"])

    @property
    def angular_momentum(self):
        """Process angular momentum."""
        return np.array(self.data["Molecule"]["angularMomentum"])

    @property
    def electric_octupole_length(self):
        """Process electric octupole length."""
        return np.array(self.data["Molecule"]["electricOctupoleLength"])

    @property
    def electric_octupole_velocity(self):
        """Process electric octupole velocity."""
        return np.array(self.data["Molecule"]["electricOctupoleVelocity"])

    @property
    def electric_quadrupole_velocity(self):
        """Process electric quadrupole velocity."""
        return np.array(self.data["Molecule"]["electricQuadrupoleVelocity"])

    @property
    def magnetic_quadrupole(self):
        """Process magnetic quadrupole."""
        return np.array(self.data["Molecule"]["magneticQuadrupole"])

    @property
    def velocity(self):
        """Process velocity."""
        return np.array(self.data["Molecule"]["velocity"])

    @property
    def static_electric_dipole_moment(self):
        """Process static electric dipole moment."""
        return einsum(
            "i,ji,ki,xjk->x",
            self.mo_occ,
            self.mo_coeff.conj(),
            self.mo_coeff,
            self.dipole_AO,
        )

    def has_property(self, prop_name):
        """Check if a property exists in the molecule data"""
        return prop_name in self.data.get("Molecule", {})

    def get_property(self, prop_name, default=None):
        """Safely get a property from molecule data with optional default"""
        try:
            value = self.data["Molecule"][prop_name]
            if isinstance(value, list):
                return np.array(value)
            return value
        except KeyError:
            return default


class UBJSONParser(JSONParser):
    def parse(self):
        """Parse the supplied input."""
        import ubjson

        with open(self.path, "rb") as f:
            data = ubjson.load(f)
        return data


class BSONParser(JSONParser):
    def parse(self):
        """Parse the supplied input."""
        import bson

        with open(self.path, "rb") as f:
            data = bson.loads(f.read())
        return data
