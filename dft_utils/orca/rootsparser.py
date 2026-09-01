from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, List

import numpy as np

from .._plams import require_plams
from ..units import Units
from .FileParser import FileParser

if TYPE_CHECKING:
    from scm.plams import Molecule


@dataclass
class SimpleCoordParser(FileParser):
    """
    Parser for a simple coordinate file of the form:

        12

        Z₁   x₁   y₁   z₁
        Z₂   x₂   y₂   z₂
        …
        Zₙ   xₙ   yₙ   zₙ

    where:
        • The first non‐empty line is an integer n = number of atoms.
        • Each of the next n lines has four columns:
            Zᵢ ∈ ℤ  (atomic number),
            xᵢ, yᵢ, zᵢ ∈ ℝ  (Cartesian coordinates).

    In “equation” form:
        Let L = [ℓ₀, ℓ₁, …, ℓₘ] be the list of non‐empty lines.
        Then
            n = int(ℓ₀),
        and for each i∈{1,…,n}:
            data_rowᵢ = [Zᵢ, xᵢ, yᵢ, zᵢ] = list(map(float, ℓᵢ.split()))
        so that
            A = ⎡
                   Z₁   x₁    y₁    z₁
                   Z₂   x₂    y₂    z₂
                   ⋮     ⋮     ⋮     ⋮
                   Zₙ   xₙ    yₙ    zₙ
                 ⎤ ∈ ℝⁿˣ⁴.

    This class exposes:
        • n_atoms:  int
        • atom_numbers: List[int]   (Z₁, …, Zₙ)
        • coords:  np.ndarray of shape (n, 3)
        • mol:     PLAMS Molecule object with positions converted from “au” → “Angstrom”.
    """

    def __init__(self, path: str):
        """Initialize the object."""
        self.path = Path(path)

    def _parse_text(self) -> str:
        """Handle parse text internally."""
        return self.path.read_text()

    @property
    def _lines(self) -> List[str]:
        """
        Return all non‐empty lines from the file.
        If the original text is T, let L_full = T.splitlines().
        Then
            _lines = [ℓ for ℓ in L_full if ℓ.strip() != ""],
        so that _lines[0] is the “n” line, and the next n entries are the atomic rows.
        """
        raw_lines = self._parse_text().splitlines()
        return [line for line in raw_lines if line.strip()]

    @cached_property
    def n_atoms(self) -> int:
        """
        The first non‐empty line, ℓ₀, is an integer:
            n = int(ℓ₀).
        """
        return int(self._lines[0])

    @cached_property
    def _data_array(self) -> np.ndarray:
        """
        Build a NumPy array A ∈ ℝⁿˣ⁴ from the next n_atoms lines:
            For i in {1,…,n}:
                row_i = _lines[i]
                tokens = row_i.split()   # [Zᵢ, xᵢ, yᵢ, zᵢ]
                Aᵢ₀ = int(tokens[0])
                Aᵢ₁, Aᵢ₂, Aᵢ₃ = float(tokens[1]), float(tokens[2]), float(tokens[3])
        Implementation uses np.loadtxt on a StringIO of “\n”.join(_lines[1 : 1+n_atoms]).
        """

        subset = self._lines[1 : 1 + self.n_atoms]
        text_block = "\n".join(subset)

        return (
            np.loadtxt(
                fname=Path(self.path).open(),
                comments="#",
                delimiter=None,
                dtype=float,
                converters={0: lambda s: float(s)},
                ndmin=2,
                skiprows=0,
            )
            if False
            else np.fromstring(text_block, sep=" ").reshape(self.n_atoms, 4)
        )

    @property
    def atom_numbers(self) -> List[int]:
        """
        Z_list = [ int(A[i, 0]) ∀ i ∈ {0,…,n - 1} ].
        """
        return [int(z) for z in self._data_array[:, 0]]

    @property
    def coords(self) -> np.ndarray:
        """
        R ∈ ℝⁿˣ³, where R[i, :] = (xᵢ, yᵢ, zᵢ) = A[i, 1:4].
        """
        return self._data_array[:, 1:4]

    @property
    def mol(self) -> Molecule:
        """
        Construct a PLAMS Molecule from (Z_list, R):
            numbers = atom_numbers = [Z₁, …, Zₙ],
            positions (converted from “au” → “Angstrom”) = Units.convert(coords, "au", "Angstrom").
        Thus,
            mol = Molecule(numbers=numbers, positions=R_converted).
        """
        Molecule = require_plams()[1]
        numbers = self.atom_numbers
        positions_angstrom = Units.convert(self.coords, "au", "Angstrom")
        return Molecule(numbers=numbers, positions=positions_angstrom)

    def parse(self):
        """Parse the supplied input."""
        return self.coords
