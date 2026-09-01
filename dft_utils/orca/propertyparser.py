import re
from dataclasses import dataclass
from functools import cache, cached_property
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from .FileParser import FileParser


@dataclass
class Parameter:
    """Base class for all parameter types"""

    name: str
    type: Optional[str] = None
    dim: Optional[str] = None
    units: Optional[str] = None
    comment: Optional[str] = None
    value: Optional[str] = None
    content: Optional[str] = None

    @property
    def value(self) -> Optional[float]:
        """
        On access: fetch the raw str from __dict__['value'],
        then convert to float (or return None).
        """
        raw = object.__getattribute__(self, "__dict__").get("_value", None)
        if raw is None:
            raw = raw = object.__getattribute__(self, "__dict__").get("comment", None)
        if raw is not None:
            try:
                raw = int(raw)
            except:
                pass
        return raw

    @value.setter
    def value(self, new: Union[str, int, None]) -> None:
        """
        On assignment to .value: normalize `new` to a string or None
        and store it in the underlying dict.
        """
        object.__setattr__(self, "_value", new)

    @property
    def dims(self):
        """Process dims."""
        dims = self.dim
        if dims is not None:
            dims = dims[1:-1].split(",")
            dims = (int(x) for x in dims)
        return dims

    @classmethod
    def from_match(cls, match: re.Match) -> "Parameter":
        """Create a parameter from a regex match"""
        return cls(
            name=match.group("name"),
            value=match.group("value").strip() if match.group("value") else None,
            comment=match.group("comment") if match.group("comment") else None,
            content=match.group("content").strip() if match.group("content") else None,
            dim=match.group("dim") if match.group("dim") else None,
            type=match.group("type") if match.group("type") else None,
            units=match.group("units") if match.group("units") else None,
        )


@dataclass
class IntParameter(Parameter):
    @property
    def value(self) -> Optional[float]:
        """
        On access: fetch the raw str from __dict__['value'],
        then convert to float (or return None).
        """
        raw = object.__getattribute__(self, "__dict__").get("_value", None)
        return raw

    @value.setter
    def value(self, new: Union[str, int, None]) -> None:
        """
        On assignment to .value: normalize `new` to a string or None
        and store it in the underlying dict.
        """
        if new is None:
            stored: Optional[int] = None
        else:
            stored = int(new) if not isinstance(new, int) else new
        object.__setattr__(self, "_value", stored)


@dataclass
class FloatParameter(Parameter):
    @property
    def value(self) -> Optional[float]:
        """
        On access: fetch the raw str from __dict__['value'],
        then convert to float (or return None).
        """
        raw = object.__getattribute__(self, "__dict__").get("_value", None)
        return raw

    @value.setter
    def value(self, new: Union[str, float, None]) -> None:
        """
        On assignment to .value: normalize `new` to a string or None
        and store it in the underlying dict.
        """
        if new is None:
            stored: Optional[float] = None
        else:
            stored = float(new) if not isinstance(new, float) else new
        object.__setattr__(self, "_value", stored)


@dataclass
class StringParameter(Parameter):
    @property
    def value(self) -> Optional[float]:
        """
        On access: fetch the raw str from __dict__['value'],
        then convert to float (or return None).
        """
        raw = object.__getattribute__(self, "__dict__").get("comment", None)
        if raw is None:
            raw = object.__getattribute__(self, "__dict__").get("_value", None)
        return raw

    @value.setter
    def value(self, new: Union[str, float, None]) -> None:
        """
        On assignment to .value: normalize `new` to a string or None
        and store it in the underlying dict.
        """
        if new is None:
            stored: Optional[str] = None
        else:
            stored = str(new) if not isinstance(new, str) else new
        object.__setattr__(self, "_value", stored)


@dataclass
class CoordsParameter(Parameter):
    @staticmethod
    def _parse_text_to_coords(text: str):
        """Handle parse text to coords internally."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        atom_types = []
        coordinates = []

        for line in lines:

            parts = line.split()
            if len(parts) < 4:
                continue

            atom_type = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])

            atom_types.append(atom_type)
            coordinates.append([x, y, z])

        coords_array = np.array(coordinates)

        stored = (atom_types, coords_array)
        return stored

    @property
    def content(self) -> Optional[float]:
        """
        On access: fetch the raw str from __dict__['value'],
        then convert to float (or return None).
        """
        raw = object.__getattribute__(self, "__dict__").get("_content", None)
        return raw

    @content.setter
    def content(self, new: Union[str, float, None]) -> None:
        """
        On assignment to .value: normalize `new` to a string or None
        and store it in the underlying dict.
        """
        if new is None:
            stored: Optional[str] = None
        else:
            stored = CoordsParameter._parse_text_to_coords(new)
        object.__setattr__(self, "_content", stored)


@dataclass
class DoublesArrayParameter(Parameter):
    @staticmethod
    def _parse_text_to_nparr(text: str, n_r, n_c):
        """Handle parse text to nparr internally."""
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

    @property
    def content(self) -> Optional[float]:
        """
        On access: fetch the raw str from __dict__['value'],
        then convert to float (or return None).
        """
        raw = object.__getattribute__(self, "__dict__").get("_content", None)
        return raw

    @content.setter
    def content(self, new: Union[str, float, None]) -> None:
        """
        On assignment to .value: normalize `new` to a string or None
        and store it in the underlying dict.
        """
        if new is None:
            stored: Optional[str] = None
        else:
            n_r, n_c = self.dims
            stored = DoublesArrayParameter._parse_text_to_nparr(new, n_r, n_c)
        object.__setattr__(self, "_content", stored)


class ParameterFactory:
    """Factory for creating parameter objects based on type"""

    _type_map: Dict[str, Callable[[re.Match], Parameter]] = {
        "integer": IntParameter.from_match,
        "double": FloatParameter.from_match,
        "coordinates": CoordsParameter.from_match,
        "string": StringParameter.from_match,
        "arrayofdoubles": DoublesArrayParameter.from_match,
    }

    @classmethod
    def register_type(
        cls, type_name: str, creator: Callable[[re.Match], Parameter]
    ) -> None:
        """Register a new parameter type"""
        cls._type_map[type_name.lower()] = creator

    @classmethod
    def create_parameter(cls, match: re.Match) -> Parameter:
        """Create a parameter from a regex match"""
        param_type = match.group("type") if match.group("type") else None

        if param_type and param_type.lower() in cls._type_map:
            try:
                return cls._type_map[param_type.lower()](match)
            except Exception as e:
                print(param_type, match.group("name"), e)
                print(list(match.groups()))

        return Parameter.from_match(match)


pattern = re.compile(
    r"&(?P<name>\w+) +(?:\[(?:(?:&Type \"(?P<type>[\w]*)\")|(?:&Dim\s*(?P<dim>\([\w,]*\)))|(?:&Units \"(?P<units>[\w]*)\")|(?:, ))+\] *)?(?P<value>[\w\.\+\-]+ *)?(?:\"(?P<comment>[\w\.\+\- ]+)\")? *\n(?P<content>[^&$]*)",
    re.MULTILINE,
)


def parse_parameters(text: str) -> List[Parameter]:
    """Parse parameters from text using the regex pattern"""

    parameters = {}
    for match in pattern.finditer(text):
        parameter = ParameterFactory.create_parameter(match)

        parameters[parameter.name] = parameter

    return parameters


class PropertyParser(FileParser):
    """
    Parse an ORCA .property.txt into a nested dict:
      { BlockName: { ParamName: value or list } }
    """

    BLOCK_RE = re.compile(
        r"\$(?P<name>\w+)(?P<content>((?:(?!\$End)[\s\S])*))\$End", re.MULTILINE
    )

    @cache
    def parse(self) -> Dict[str, Dict[str, Any]]:
        """Parse the supplied input."""
        text = self._parse_text()

        blocks = self._split_text_into_blocks(text)

        result: List[str, Dict[str, Any]] = []
        for blk_name, blk_text in blocks:
            name = blk_name
            parsed = self._parse_block(blk_text)

            result.append((name, parsed))
        return result

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
            blocks.append((name, content))

        return blocks

    def _parse_block(self, text: str) -> Dict[str, Any]:
        """Handle parse block internally."""
        return parse_parameters(text)

    @cached_property
    def GeometryIndices(self):
        """Process GeometryIndices."""
        return sorted(
            set(
                d[1]["GeometryIndex"].value
                for d in self.parse()
                if "GeometryIndex" in d[1]
            )
        )

    @cached_property
    def LastGI(self):
        """Process LastGI."""
        return max(self.GeometryIndices)

    @property
    def electrons_A(self):
        """Process electrons A."""
        p = self.parse()
        d = next((x for x in p if x[0] == "DFT_Energy"))[1]
        return d["NALPHAEL"].value

    @property
    def electrons_B(self):
        """Process electrons B."""
        p = self.parse()
        d = next((x for x in p if x[0] == "DFT_Energy"))[1]
        return d["NALPHAEL"].value

    @property
    def electrons_tot(self):
        """Process electrons tot."""
        p = self.parse()
        d = next((x for x in p if x[0] == "DFT_Energy"))[1]
        return d["NTOTALEL"].value

    @property
    def atom_xyzs(self):
        """Process atom xyzs."""
        p = self.parse()
        d = next(
            (
                x
                for x in p
                if x[0] == "Geometry" and x[1]["GeometryIndex"].value == self.LastGI
            )
        )[1]
        return d["CartesianCoordinates"].content[1]

    @property
    def atom_symbols(self):
        """Process atom symbols."""
        p = self.parse()
        d = next(
            (
                x
                for x in p
                if x[0] == "Geometry" and x[1]["GeometryIndex"].value == self.LastGI
            )
        )[1]
        return d["CartesianCoordinates"].content[0]

    @property
    def bond_energy(self):
        """Process bond energy."""
        p = self.parse()
        d = next(
            (
                x
                for x in p
                if x[0] == "SCF_Energy" and x[1]["GeometryIndex"].value == self.LastGI
            )
        )[1]

        return d["totalEnergy"].content[0, 0]

    @property
    def all_nuc_gradients(self):
        """Process all nuc gradients."""
        p = self.parse()
        d = [val[1]["GRAD"].content[:, 0] for val in p if "CIS_Nuc_Gradient" in val[0]]
        d = np.array(d)
        return d

    @property
    def nuc_gradients(self):
        """Process nuc gradients."""
        p = self.parse()
        d = [val for val in p if "CIS_Nuc_Gradient" in val[0]]
        d = [val for val in d if val[1]["GeometryIndex"].value == 2]
        d = [val[1]["GRAD"].content[:, 0] for val in d]

        d = np.array(d)
        return d


if __name__ == "__main__":
    sample_text = """
    &param1 [&Type "string", &Units "meters"] "This is a string parameter"
    This is the content for param1

    &param2 [&Type "int", &Dim (2,3)] 42 "An integer parameter"
    Extra content here

    &param3 [&Type "boolean"] true "A boolean flag"

    &param4 [&Type "list"]
    item1
    item2
    item3
    """

    params = parse_parameters(sample_text)
    for param in params:
        print(f"{param.name} ({type(param).__name__}):")
        print(f"  Value: {param.get_value()}")
        print(f"  Comment: {param.comment}")
        print(f"  Dimension: {param.dim}")
        print(f"  Units: {param.units}")
        if param.content:
            print(
                f"  Content: {param.content[:50]}..."
                if len(param.content) > 50
                else f"  Content: {param.content}"
            )
        print()
