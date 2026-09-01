"""Utilities for analysing ADF and ORCA quantum-chemistry results.

The repository uses the import name :mod:`dft_utils`.  The package is kept
lightweight at import time; optional ORCA/ADF integrations are loaded only
when their modules are imported.
"""

from .periodic_table import PT, PeriodicTable
from .units import Units

__all__ = ["PT", "PeriodicTable", "Units"]

__version__ = "0.1.0"
