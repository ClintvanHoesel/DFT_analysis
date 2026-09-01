"""Lazy access to the optional SCM PLAMS runtime."""

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def require_plams() -> tuple[Any, Any, Any, Any, Any]:
    """Return the PLAMS classes used by the package.

    PLAMS is an optional dependency because it is normally supplied by the
    external AMS installation.  Keeping this import behind a function allows
    parsers that do not need PLAMS to be imported in a regular Python
    environment.

    Raises:
        ImportError: If the AMS/PLAMS runtime is not importable.
    """
    try:
        from scm.plams import Atom, Settings
        from scm.plams.mol.molecule import Molecule
        from scm.plams.tools.kftools import KFFile
        from scm.plams.tools.periodic_table import PT
    except ImportError as exc:
        raise ImportError(
            "This operation requires the optional SCM PLAMS runtime. "
            "Install or activate the AMS/PLAMS environment so that "
            "`scm.plams` is importable."
        ) from exc

    return Atom, Molecule, Settings, KFFile, PT
