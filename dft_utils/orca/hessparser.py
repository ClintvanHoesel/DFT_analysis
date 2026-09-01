import re
from dataclasses import dataclass
from functools import cache
from typing import Any, Dict, Optional, Union

import numpy as np

from .._plams import require_plams
from ..units import Units
from .FileParser import FileParser


def _parse_text_to_nparr_tokens2(text: str, n_r, n_c):
    """Handle parse text to nparr tokens2 internally."""
    A = np.zeros((n_r, n_c), float)

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        tokens = line.split()
        if not tokens:
            i += 1
            continue

        if all(tok.isdigit() for tok in tokens):
            cols = list(map(int, tokens))
            i += 1
            if not lines[i].strip():
                i += 1

            j = 0
            while i < len(lines) and lines[i].strip() and j < n_r:
                row_tokens = lines[i].split()
                row = int(row_tokens[0])
                values = row_tokens[1:]
                for m, val in enumerate(values):
                    A[row, cols[m]] = float(val)
                i += 1
                j += 1
            continue

        i += 1

    return A


def _parse_text_to_nparr_tokens(text: str):
    """Handle parse text to nparr tokens internally."""
    lines = text.splitlines()
    first_line = lines[0].split()
    if len(first_line) == 1:
        n_r = int(first_line[0])
        n_c = n_r
    elif len(first_line) == 2:
        n_r = int(first_line[0])
        n_c = int(first_line[1])
    else:
        raise ValueError("First line is not correct!")
    text = "\n".join(lines[1:])
    return _parse_text_to_nparr_tokens2(text, n_r, n_c)


def safe_convert(value):
    """Try to convert a string to int, then float, otherwise return the original."""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _parse_text_to_nparr_simple(text: str):
    """Handle parse text to nparr simple internally."""
    lines = text.splitlines()
    first_line = lines[0].split()
    if len(first_line) == 1:
        n_r = int(first_line[0])
        n_c = n_r
    elif len(first_line) == 2:
        n_r = int(first_line[0])
        n_c = int(first_line[1])
    else:
        raise ValueError("First line is not correct!")
    arr = [
        [safe_convert(x) for x in line.split()]
        for line in lines[1:]
        if (line.strip() and not line.strip()[0] == "#")
    ]
    return np.array(arr)


@dataclass
class HessParameter:
    """Base class for all parameter types"""

    name: str
    content: Optional[str] = None

    @property
    def content(self) -> Optional[float]:
        """
        On access: fetch the raw str from __dict__['value'],
        then convert to float (or return None).
        """
        raw = object.__getattribute__(self, "__dict__").get("_content", None)
        return raw

    @content.setter
    def content(self, new: Union[str, int, None]) -> None:
        """
        On assignment to .value: normalize `new` to a string or None
        and store it in the underlying dict.
        """
        object.__setattr__(self, "_content", new)
        for f in [int, float]:
            try:
                new = f(new)
                object.__setattr__(self, "_content", new)
                return
            except Exception:
                pass

        if len(new.splitlines()) > 2:
            for f in [_parse_text_to_nparr_simple, _parse_text_to_nparr_tokens]:
                try:
                    new = f(new)
                    object.__setattr__(self, "_content", new)
                    return
                except Exception:
                    pass

    @classmethod
    def from_match(cls, match: re.Match) -> "HessParameter":
        """Create a parameter from a regex match"""
        return cls(
            name=match.group("name"),
            content=match.group("content").strip() if match.group("content") else None,
        )


class ParameterFactory:
    """Factory for creating parameter objects based on type"""

    @classmethod
    def create_parameter(cls, match: re.Match) -> HessParameter:
        """Create a parameter from a regex match"""
        return HessParameter.from_match(match)


pattern = re.compile(
    r"\$(?P<name>[\w]+) *\n(?P<content>[^&$]*)",
    re.MULTILINE,
)


class HessParser(FileParser):
    BLOCK_RE = re.compile(
        r"\$(?P<name>[\w]+) *\n(?P<content>[^&$]*)",
        re.MULTILINE,
    )

    def _parse_text(self) -> str:
        """Handle parse text internally."""
        text = self.path.read_text()
        return text

    def _split_text_into_blocks(self, text: str):
        """Handle split text into blocks internally."""
        blocks = []
        for match in self.BLOCK_RE.finditer(text):
            name = match.group("name")
            content = match.group("content")
            blocks.append((name, content, match))

        return blocks

    def _parse_block(self, match: str) -> Dict[str, Any]:
        """Handle parse block internally."""
        return ParameterFactory.create_parameter(match)

    @cache
    def parse(self) -> Dict[str, Dict[str, Any]]:
        """Parse the supplied input."""
        text = self._parse_text()

        blocks = self._split_text_into_blocks(text)

        result: Dict[str, Dict[str, Any]] = {}
        for blk_name, _blk_text, blk_match in blocks:
            name = blk_name
            parsed = self._parse_block(blk_match)

            result[name] = parsed
        return result

    @property
    def hessian(self):
        """Process hessian."""
        p = self.parse()

        hessian = next((v for k, v in p.items() if k == "hessian"))
        hessian = hessian.content
        return hessian

    @property
    def atom_types(self):
        """Process atom types."""
        return self.parse()["atoms"].content[:, 0].tolist()

    @property
    def atom_coords(self):
        """Process atom coords."""
        return self.parse()["atoms"].content[:, 2:].astype(float)

    @property
    def atom_numbers(self):
        """Process atom numbers."""
        PT = require_plams()[4]
        p = self.atom_types
        return [PT.get_atomic_number(symbol) for symbol in p]

    @property
    def mol(self):
        """Process mol."""
        Molecule = require_plams()[1]
        return Molecule(
            numbers=self.atom_numbers,
            positions=Units.convert(self.atom_coords, "au", "Angstrom"),
        )
