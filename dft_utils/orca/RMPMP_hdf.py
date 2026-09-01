import sqlite3
import warnings
from functools import cache, cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

from dft_utils.chkfile import load, save
from dft_utils.colour_utils import get_RGB_colour_from_spectrum
from dft_utils.radii_calc import MPMP_Radius_calculator
from dft_utils.spectrum_converter import (
    Multipole_Absorption_Spectrum,
    Multipole_Emission_Spectrum,
)
from dft_utils.units import Units

UNSET = object()


def find_hdf5_files(directory: str, file_pattern: str = "*.hdf5") -> List[Path]:
    """Find all HDF5 files in directory and subdirectories."""
    path = Path(directory)
    return list(path.rglob(file_pattern))


def load_hdf5_files_to_dataframe(
    root_folder: str,
    dataset_keys: Optional[List[str]] = None,
    file_pattern: str = "*.hdf5",
    add_file_info: bool = True,
    concat_axis: int = 0,
    handle_errors: str = "warn",
) -> pd.DataFrame:
    """
    Recursively find and load HDF5 files from a directory into a pandas DataFrame.

    Parameters:
    -----------
    root_folder : str
        Root directory to search for HDF5 files
    dataset_keys : List[str], optional
        Specific dataset keys to load from each HDF5 file. If None, attempts to load all datasets
    file_pattern : str, default "*.hdf5"
        File pattern to match (supports *.hdf5, *.h5, etc.)
    add_file_info : bool, default True
        Whether to add columns with file path and name information
    concat_axis : int, default 0
        Axis along which to concatenate DataFrames (0=rows, 1=columns)
    handle_errors : str, default 'warn'
        How to handle errors: 'warn', 'skip', or 'raise'

    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with data from all HDF5 files
    """

    def load_single_hdf5(file_path: Path) -> Optional[pd.DataFrame]:
        """Load data from a single HDF5 file."""
        try:
            with h5py.File(file_path, "r") as f:
                data_dict = {}

                if dataset_keys is None:

                    keys_to_load = list(f.keys())
                else:
                    keys_to_load = dataset_keys

                for key in keys_to_load:
                    if key in f:
                        dataset = f[key]

                        if dataset.ndim == 1:
                            data_dict[key] = dataset[:]
                        elif dataset.ndim == 2:

                            data_dict[key] = dataset[:].flatten()
                        else:

                            data_dict[key] = dataset[0] if dataset.size > 0 else None
                    elif dataset_keys is not None:
                        warning_msg = f"Dataset '{key}' not found in {file_path}"
                        if handle_errors == "warn":
                            warnings.warn(warning_msg, stacklevel=2)
                        elif handle_errors == "raise":
                            raise KeyError(warning_msg)

                if not data_dict:
                    if handle_errors != "skip":
                        warnings.warn(f"No datasets found in {file_path}", stacklevel=2)
                    return None

                df = pd.DataFrame(data_dict)

                if add_file_info:
                    df["file_path"] = str(file_path)
                    df["file_name"] = file_path.name
                    df["file_dir"] = str(file_path.parent)

                return df

        except Exception as e:
            error_msg = f"Error loading {file_path}: {str(e)}"
            if handle_errors == "raise":
                raise
            elif handle_errors == "warn":
                warnings.warn(error_msg, stacklevel=2)
            return None

    print(f"Searching for HDF5 files in: {root_folder}")
    hdf5_files = find_hdf5_files(root_folder, file_pattern=file_pattern)

    if not hdf5_files:
        print(f"No HDF5 files found matching pattern '{file_pattern}'")
        return pd.DataFrame()

    print(f"Found {len(hdf5_files)} HDF5 files")

    dataframes = []
    for file_path in hdf5_files:
        print(f"Loading: {file_path}")
        df = load_single_hdf5(file_path)
        if df is not None and not df.empty:
            dataframes.append(df)

    if not dataframes:
        print("No data was successfully loaded from any files")
        return pd.DataFrame()

    print(f"Combining {len(dataframes)} DataFrames...")
    try:
        combined_df = pd.concat(
            dataframes, axis=concat_axis, ignore_index=True, sort=False
        )
        print(f"Successfully created DataFrame with shape: {combined_df.shape}")
        return combined_df
    except Exception as e:
        print(f"Error combining DataFrames: {e}")
        if handle_errors == "raise":
            raise
        return pd.DataFrame()


def inspect_hdf5_structure(file_path: str) -> Dict[str, Any]:
    """
    Inspect the structure of an HDF5 file to see available datasets.

    Parameters:
    -----------
    file_path : str
        Path to the HDF5 file

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing information about datasets in the file
    """
    structure = {}

    try:
        with h5py.File(file_path, "r") as f:

            def visitor_func(name, obj):
                """Process visitor func."""
                if isinstance(obj, h5py.Dataset):
                    structure[name] = {
                        "shape": obj.shape,
                        "dtype": obj.dtype,
                        "size": obj.size,
                    }

            f.visititems(visitor_func)

    except Exception as e:
        print(f"Error inspecting {file_path}: {e}")

    return structure


def quick_inspect_folder(folder_path: str, max_files: int = 5):
    """Quickly inspect the first few HDF5 files to understand their structure."""
    hdf5_files = list(Path(folder_path).rglob("*.hdf5"))[:max_files]

    for file_path in hdf5_files:
        print(f"\n--- Structure of {file_path.name} ---")
        structure = inspect_hdf5_structure(str(file_path))
        for dataset_name, info in structure.items():
            print(f"  {dataset_name}: shape={info['shape']}, dtype={info['dtype']}")


class PairwiseDatabase:
    """
    Object-oriented interface for storing and retrieving two-molecule spectral data
    in a SQLite database.

    Key: (mol_i, mol_j,
          charge_i, spec_i, ph_i, vib_i, ord_i,
          charge_j, spec_j, ph_j, vib_j, ord_j)
    -> pickled numpy array
    Molecule identifiers (mol_i, mol_j) are stored as TEXT.
    Boolean flags (spec, ph, vib) are stored with SQLite BOOLEAN affinity.
    """

    _DDL = """
    CREATE TABLE IF NOT EXISTS pair_data (
        mol_i        TEXT    NOT NULL,
        mol_j        TEXT    NOT NULL,
        charge_i     INTEGER NOT NULL,
        spec_i       BOOLEAN NOT NULL,
        ph_i         BOOLEAN NOT NULL,
        vib_i        BOOLEAN NOT NULL,
        ord_i        INTEGER NOT NULL,
        charge_j     INTEGER NOT NULL,
        spec_j       BOOLEAN NOT NULL,
        ph_j         BOOLEAN NOT NULL,
        vib_j        BOOLEAN NOT NULL,
        ord_j        INTEGER NOT NULL,
        iexc_i       INTEGER         ,
        iexc_j       INTEGER         ,
        x_spec_blob  FLOAT           ,
        k_r          FLOAT           ,
        PRIMARY KEY (mol_i, mol_j,
                     charge_i, spec_i, ph_i, vib_i, ord_i,
                     charge_j, spec_j, ph_j, vib_j, ord_j, iexc_i, iexc_j)
    );
    """

    def __init__(self, db_path: str):
        """
        Initialize the database connection and ensure schema exists.
        """
        self.db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Handle ensure schema internally."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(self._DDL)

    def save(
        self,
        mol_i: str,
        mol_j: str,
        charge_i: int,
        spec_i: bool,
        ph_i: bool,
        vib_i: bool,
        ord_i: int,
        charge_j: int,
        spec_j: bool,
        ph_j: bool,
        vib_j: bool,
        ord_j: int,
        iexc_i: int,
        iexc_j: int,
        x_spec: float,
        kr: float,
    ) -> None:
        """
        Store the spectrum array for a molecule pair,
        where each molecule has its own settings and charge.
        Boolean flags are stored directly as True/False.
        """
        key = (
            mol_i,
            mol_j,
            charge_i,
            spec_i,
            ph_i,
            vib_i,
            ord_i,
            charge_j,
            spec_j,
            ph_j,
            vib_j,
            ord_j,
            iexc_i,
            iexc_j,
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pair_data
                (mol_i, mol_j,
                 charge_i, spec_i, ph_i, vib_i, ord_i,
                 charge_j, spec_j, ph_j, vib_j, ord_j,
                 iexc_i, iexc_j,
                 x_spec_blob, k_r)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (*key, x_spec, kr),
            )

    def load(
        self,
        mol_i: str,
        mol_j: str,
        charge_i: int,
        spec_i: bool,
        ph_i: bool,
        vib_i: bool,
        ord_i: int,
        charge_j: int,
        spec_j: bool,
        ph_j: bool,
        vib_j: bool,
        ord_j: int,
        iexc_i: int,
        iexc_j: int,
    ) -> float:
        """
        Retrieve the spectrum array for a molecule pair with individual settings and charge.
        Raises KeyError if no entry exists.
        """
        key = (
            mol_i,
            mol_j,
            charge_i,
            spec_i,
            ph_i,
            vib_i,
            ord_i,
            charge_j,
            spec_j,
            ph_j,
            vib_j,
            ord_j,
            iexc_i,
            iexc_j,
        )
        sql = """
            SELECT x_spec_blob FROM pair_data
            WHERE mol_i=? AND mol_j=?
              AND charge_i=? AND spec_i=? AND ph_i=? AND vib_i=? AND ord_i=?
              AND charge_j=? AND spec_j=? AND ph_j=? AND vib_j=? AND ord_j=?
              AND iexc_i=?   AND iexc_j=?;
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(sql, key)
            row = cur.fetchone()
        if row is None:
            raise KeyError(f"No entry for key {key}")
        return row[0]

    def exists(
        self,
        mol_i: str,
        mol_j: str,
        charge_i: int,
        spec_i: bool,
        ph_i: bool,
        vib_i: bool,
        ord_i: int,
        charge_j: int,
        spec_j: bool,
        ph_j: bool,
        vib_j: bool,
        ord_j: int,
        iexc_i: int,
        iexc_j: int,
    ) -> bool:
        """
        Check efficiently if a given key exists in the database.
        Returns True if present, False otherwise.
        """
        key = (
            mol_i,
            mol_j,
            charge_i,
            spec_i,
            ph_i,
            vib_i,
            ord_i,
            charge_j,
            spec_j,
            ph_j,
            vib_j,
            ord_j,
        )
        sql = """
            SELECT 1 FROM pair_data
            WHERE mol_i=? AND mol_j=?
              AND charge_i=? AND spec_i=? AND ph_i=? AND vib_i=? AND ord_i=?
              AND charge_j=? AND spec_j=? AND ph_j=? AND vib_j=? AND ord_j=?
              AND iexc_i=?   AND iexc_j=?;
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(sql, key)
            return cur.fetchone() is not None

    def get_all(self) -> List[Dict[str, Any]]:
        """
        Retrieve all entries in the database.
        Returns a list of dicts with keys for metadata and 'x_spec'.
        """
        sql = "SELECT * FROM pair_data;"
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
        return [self._row_to_dict(row) for row in rows]

    def query(
        self,
        mol_i: Optional[str] = UNSET,
        mol_j: Optional[str] = UNSET,
        charge_i: Optional[int] = UNSET,
        spec_i: Optional[bool] = UNSET,
        ph_i: Optional[bool] = UNSET,
        vib_i: Optional[bool] = UNSET,
        ord_i: Optional[int] = UNSET,
        charge_j: Optional[int] = UNSET,
        spec_j: Optional[bool] = UNSET,
        ph_j: Optional[bool] = UNSET,
        vib_j: Optional[bool] = UNSET,
        ord_j: Optional[int] = UNSET,
        iexc_i: Optional[int] = UNSET,
        iexc_j: Optional[int] = UNSET,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve entries matching provided filters.
        Use UNSET (default) to ignore a field.
        Use None to match NULL values in the database.
        """
        filters: List[str] = []
        params: List[Any] = []

        for field, value in [
            ("mol_i", mol_i),
            ("mol_j", mol_j),
            ("charge_i", charge_i),
            ("spec_i", spec_i),
            ("ph_i", ph_i),
            ("vib_i", vib_i),
            ("ord_i", ord_i),
            ("charge_j", charge_j),
            ("spec_j", spec_j),
            ("ph_j", ph_j),
            ("vib_j", vib_j),
            ("ord_j", ord_j),
            ("iexc_i", iexc_i),
            ("iexc_j", iexc_j),
        ]:
            if value is UNSET:
                continue
            elif value is None:
                filters.append(f"{field} IS NULL")
            else:
                filters.append(f"{field}=?")
                params.append(value)

        sql = "SELECT * FROM pair_data"
        if filters:
            sql += " WHERE " + " AND ".join(filters)
        sql += ";"

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(sql, params)
            rows = cur.fetchall()
        return [self._row_to_dict(row) for row in rows]

    def delete(
        self,
        mol_i: Optional[str] = None,
        mol_j: Optional[str] = None,
        charge_i: Optional[int] = None,
        spec_i: Optional[bool] = None,
        ph_i: Optional[bool] = None,
        vib_i: Optional[bool] = None,
        ord_i: Optional[int] = None,
        charge_j: Optional[int] = None,
        spec_j: Optional[bool] = None,
        ph_j: Optional[bool] = None,
        vib_j: Optional[bool] = None,
        ord_j: Optional[int] = None,
        iexc_i: Optional[int] = None,
        iexc_j: Optional[int] = None,
    ) -> int:
        """
        Delete entries matching provided filters. None means no filter on that field.
        Returns the number of entries deleted.
        """
        filters: List[str] = []
        params: List[Any] = []
        for field, value in [
            ("mol_i", mol_i),
            ("mol_j", mol_j),
            ("charge_i", charge_i),
            ("spec_i", spec_i),
            ("ph_i", ph_i),
            ("vib_i", vib_i),
            ("ord_i", ord_i),
            ("charge_j", charge_j),
            ("spec_j", spec_j),
            ("ph_j", ph_j),
            ("vib_j", vib_j),
            ("ord_j", ord_j),
            ("iexc_i", iexc_i),
            ("iexc_j", iexc_j),
        ]:
            if value is not None:
                filters.append(f"{field}=?")
                params.append(value)

        sql = "DELETE FROM pair_data"
        if filters:
            sql += " WHERE " + " AND ".join(filters)
        sql += ";"

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(sql, params)
            return cur.rowcount

    def _row_to_dict(self, row: Tuple) -> Dict[str, Any]:
        """Handle row to dict internally."""
        keys = [
            "mol_i",
            "mol_j",
            "charge_i",
            "spec_i",
            "ph_i",
            "vib_i",
            "ord_i",
            "charge_j",
            "spec_j",
            "ph_j",
            "vib_j",
            "ord_j",
            "iexc_i",
            "iexc_j",
            "x_spec_blob",
            "k_r",
        ]
        data = dict(zip(keys, row))

        return data


if __name__ == "__main__":
    db = PairwiseDatabase("pairwise.db")

    db.save(
        mol_i="mol_A",
        mol_j="mol_B",
        charge_i=0,
        spec_i=False,
        ph_i=False,
        vib_i=False,
        ord_i=0,
        charge_j=1,
        spec_j=True,
        ph_j=True,
        vib_j=True,
        ord_j=1,
        x_spec=np.pi,
    )
    loaded = db.load(
        mol_i="mol_A",
        mol_j="mol_B",
        charge_i=0,
        spec_i=False,
        ph_i=False,
        vib_i=False,
        ord_i=0,
        charge_j=1,
        spec_j=True,
        ph_j=True,
        vib_j=True,
        ord_j=1,
    )
    assert np.allclose(np.pi, loaded)


class HDF_Spectrum:
    def __init__(self, file_path):
        """Initialize the object."""
        self.file_path = file_path

    @property
    def fp(self):
        """Process fp."""
        return self.file_path

    @property
    def file_name(self):
        """Process file name."""
        return self.file_path.name

    @property
    def name(self):
        """Process name."""
        return self.file_path.stem

    @property
    def file_dir(self):
        """Process file dir."""
        return str(self.file_path.parent)

    def write(self, key, val):
        """Process write."""
        save(self.file_path, self.name + "/" + key, val)

    def get(self, key):
        """Process get."""
        return load(self.file_path, self.name + "/" + key)

    def spec_string(self, charge, ph, spec, vib, order, iexc=None, magn=False):
        """Process spec string."""
        specc = "emission_spectrum" if spec else "absorption_spectrum"
        phh = "phosp" if ph else "fluor"
        vibb = "vib" if vib else "novib"
        if not magn:
            if order == 1:
                order = "tdms"
            elif order == 2:
                order = "tqms"
            elif order == 3:
                order = "toms"
        else:
            if order == 1:
                order = "tmdms"
            elif order == 2:
                order = "tmqms"
            elif order == 3:
                order = "tmoms"
        if isinstance(iexc, int) and iexc is not None:
            iexcs = f"/{iexc}"
        else:
            iexcs = ""
        k = f"/{charge}/{phh}/{specc}/{vibb}/{order}{iexcs}"
        return k

    @cache
    def load_moms(self, charge, ph, spec, vib, order, iexc=None, magn=False):
        """Process load moms."""
        k = self.spec_string(charge, ph, spec, vib, order, iexc, magn)
        x = self.get(k + "/x")
        y = self.get(k + "/y")
        return x, y

    @cache
    def load_rgb(self, charge, ph, spec, vib, order, iexc=None, magn=False):
        """Process load rgb."""
        k = self.spec_string(charge, ph, spec, vib, order, iexc, magn)
        x = self.get(k + "/rgb")
        return x

    @cache
    def load_spec(self, charge, ph, spec, vib, order, iexc=None, magn=False):
        """Process load spec."""
        k = self.spec_string(charge, ph, spec, vib, order, iexc, magn)
        x = self.get(k + "/x_spec")
        y = self.get(k + "/y_spec")
        return x, y

    @cache
    def load_spec_plot(self, charge, ph, spec, vib, order, iexc=None, magn=False):
        """Process load spec plot."""
        x, y = self.load_spec(charge, ph, spec, vib, order, iexc, magn)

        x = Units.convert(x, "J", "eV")

        if spec:
            y = Units.nconvert(y, "J", "eV", -1)
        return x, y

    @cached_property
    def group(self):
        """Process group."""
        return self.get("group").decode()

    @cache
    def get_kr(self, ph, vib=True, charge=0, order=1, magn=False):
        """Return kr."""
        sett = {
            "spec": True,
            "order": order,
            "ph": ph,
            "vib": vib,
            "charge": charge,
            "magn": magn,
        }
        x, y = self.load_spec(**sett)
        spec = Multipole_Emission_Spectrum(sett["order"], x, y, magn=magn)
        return spec.norm

    def get_mom(self, charge, ph, spec, vib, order, iexc=None, magn=False, kr=None):
        """Return mom."""
        x, y = self.load_moms(charge, ph, spec, vib, order, iexc, magn)

        if spec:

            spec = Multipole_Emission_Spectrum(order, x, y, kr=kr, magn=magn)
        else:
            spec = Multipole_Absorption_Spectrum(order, x, y, magn=magn)
        return spec

    def get_spec(self, charge, ph, spec, vib, order, iexc=None, magn=False, kr=None):
        """Return spec."""
        x, y = self.load_spec(charge, ph, spec, vib, order, iexc, magn)

        if spec:

            spec = Multipole_Emission_Spectrum(
                order,
                x,
                y,
                kr=kr,
                magn=magn,
            )
        else:
            spec = Multipole_Absorption_Spectrum(
                order,
                x,
                y,
                magn=magn,
            )
        return spec

    def calc_rgb(self, charge, ph, spec, vib, order, iexc=None, magn=False):
        """Process calc rgb."""
        x, y = self.load_spec_plot(charge, ph, spec, vib, order, iexc, magn)
        x = 1240.0 / x
        if spec:
            y = y * 1240.0 / (x * x)
        return get_RGB_colour_from_spectrum(x, y)

    def get_calc_RMPMP(self, other, sett1: dict, sett2: dict, error=False, **kwargs):
        """Return calc RMPMP."""
        if "spec" in sett1:
            assert sett1["spec"]
        else:
            sett1["spec"] = True
        if "spec" in sett2:
            assert not sett2["spec"]
        else:
            sett2["spec"] = False
        spec1 = self.get_mom(**sett1)
        spec2 = other.get_mom(**sett2)
        calc = MPMP_Radius_calculator(spec1, spec2, **kwargs)
        kr = self.get_kr(sett1["ph"], sett1["vib"], sett1["charge"])
        calc.krr = kr
        return calc

    def calc_RMPMP(self, other, sett1: dict, sett2: dict, error=False, **kwargs):
        """Process calc RMPMP."""
        calc = self.get_calc_RMPMP(other, sett1, sett2, error=error)
        return calc.get_MPMP_radius(error=error, norm_kr=calc.krr)
