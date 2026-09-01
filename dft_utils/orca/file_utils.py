"""Shared filesystem and archive helpers for ORCA workflows."""

import os
import shutil
from pathlib import Path

import numpy as np

from .reader import JOB_RE
from .tarparser import TarGzParser


def remove_directory_pathlib(path):
    """Remove a directory and all its contents using ``pathlib``."""
    path = Path(path)
    if path.is_dir():
        shutil.rmtree(path)
        print(f"Removed directory: {path}")
    else:
        print(f"Path is not a directory: {path}")


def remove_directory_safe(path):
    """Remove a directory while reporting common filesystem errors."""
    path = Path(path)
    try:
        if path.is_dir():
            shutil.rmtree(path)
            print(f"Successfully removed directory: {path}")
        else:
            print(f"Path is not a directory: {path}")
    except PermissionError:
        print(f"Permission denied: Cannot remove {path}")
    except FileNotFoundError:
        print(f"Directory not found: {path}")
    except Exception as exc:
        print(f"Error removing directory {path}: {exc}")


def remove_directory_force(path):
    """Remove a directory while ignoring removal errors."""
    path = Path(path)
    if path.is_dir():
        try:
            shutil.rmtree(path, ignore_errors=True)
            print(f"Force removed directory: {path}")
        except Exception as exc:
            print(f"Could not remove directory {path}: {exc}")


def remove_directory_confirm(path):
    """Remove a directory after asking the user for confirmation."""
    path = Path(path)
    if path.is_dir():
        response = input(f"Remove directory '{path}' and all its contents? (y/N): ")
        if response.lower() in ["y", "yes"]:
            try:
                shutil.rmtree(path)
                print(f"Removed directory: {path}")
            except Exception as exc:
                print(f"Error removing directory: {exc}")
        else:
            print("Removal cancelled.")
    else:
        print(f"Path is not a directory: {path}")


def remove_directory_if_exists(path):
    """Remove a path only when it exists and is a directory."""
    path = Path(path)
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
        print(f"Removed existing directory: {path}")
    elif path.exists():
        print(f"Path exists but is not a directory: {path}")
    else:
        print(f"Path does not exist: {path}")


def remove_dir_oneliner(path):
    """Remove a directory using a compact best-effort implementation."""
    path = Path(path)
    path.is_dir() and shutil.rmtree(path)
    try:
        path.is_dir() and shutil.rmtree(path)
    except Exception:
        pass


def find_all_tar_gz(folder):
    """Return all ``.tar.gz`` files directly inside a folder."""
    out_files = []
    files = os.listdir(folder)
    for file in files:
        if file[-7:] == ".tar.gz":
            out_files.append(os.path.join(folder, file))
    return out_files


def get_all_tar_gz_parsers(folder):
    """Return parsers for all ``.tar.gz`` files in a folder."""
    paths = find_all_tar_gz(folder)
    return [TarGzParser(path) for path in paths]


def get_tar_gz_parsers_per_mol(folder):
    """Group tar archive parsers by the molecule name in their filenames."""
    out = {}
    for parser in get_all_tar_gz_parsers(folder):
        match = JOB_RE.match(parser.name)
        if not match:
            print("Could not parse ", parser.name)
            continue
        out.setdefault(match.group("jobname"), []).append(parser)
    return out


def safe_get_energy_or_nan(func, *args):
    """Return a function result, or ``numpy.nan`` when it raises an error."""
    try:
        return func(*args)
    except Exception as exc:
        print(exc)
        return np.nan
