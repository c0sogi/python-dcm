import copy
import itertools
import logging
import re
import sys
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from io import (
    BytesIO,
)
from math import isnan, nan
from typing import (
    Callable,
    Final,
    Iterable,
    Iterator,
    Optional,
)

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from .linear_interpolation import lininterp1, lininterp2
from .typing import (
    BytesReadable,
    BytesWritable,
    CurveFunction,
    FileDescriptorOrPath,
    FloatDtype,
    K,
    MapFunction,
    PathOrReadable,
    StringReadable,
    StringWritable,
    V,
    Writable,
)

if sys.version_info >= (3, 11):
    from typing import LiteralString, Self  # python >= 3.11
else:
    LiteralString = str
    Self = "DCM"

logger: Final[logging.Logger] = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# re.Patterns: dimensions
PATTERN_1D: Final[bytes] = rb"\s+(.*?)\s+(\d+)"
PATTERN_2D: Final[bytes] = rb"\s+(.*?)\s+(\d+)\s+(\d+)"

# re.Patterns: properties
VERSION_PATTERN: Final[re.Pattern[bytes]] = re.compile(rb"(\d\.\d)")
PURE_INT_PATTERN: Final[re.Pattern[str]] = re.compile(r"[-+]?\d+")
PURE_FLOAT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?"
)
TEXT_PATTERN: Final[re.Pattern[bytes]] = re.compile(rb'"(.*?)"')
VARIANT_PATTERN: Final[re.Pattern[bytes]] = re.compile(rb"VAR\s+(.*?)=(.*)")
AXIS_X_PATTERN: Final[re.Pattern[bytes]] = re.compile(rb"SSTX\s+(.*)")
AXIS_Y_PATTERN: Final[re.Pattern[bytes]] = re.compile(rb"SSTY\s+(.*)")

EMPTY_SERIES: Final[pd.Series] = pd.Series()
EMPTY_DATAFRAME: Final[pd.DataFrame] = pd.DataFrame()
EMPTY_ARRAY: Final[np.ndarray] = np.array(())

# re.Patterns: classes
PARAMETER_BLOCK_PATTERN: Final[re.Pattern[bytes]] = re.compile(
    b"FESTWERTEBLOCK" + rb"\s+(.*?)\s+(\d+)(?:\s+\@\s+(\d+))?"
)
DISTRIBUTION_PATTERN: Final[re.Pattern[bytes]] = re.compile(
    b"STUETZSTELLENVERTEILUNG" + PATTERN_1D
)
CHARACTERISTIC_LINE_PATTERN: Final[re.Pattern[bytes]] = re.compile(
    b"KENNLINIE" + PATTERN_1D
)
CHARACTERISTIC_MAP_PATTERN: Final[re.Pattern[bytes]] = re.compile(
    b"KENNFELD" + PATTERN_2D
)
FIXED_CHARACTERISTIC_LINE_PATTERN: Final[re.Pattern[bytes]] = re.compile(
    b"FESTKENNLINIE" + PATTERN_1D
)
FIXED_CHARACTERISTIC_MAP_PATTERN: Final[re.Pattern[bytes]] = re.compile(
    b"FESTKENNFELD" + PATTERN_2D
)
GROUP_CHARACTERISTIC_LINE_PATTERN: Final[re.Pattern[bytes]] = re.compile(
    b"GRUPPENKENNLINIE" + PATTERN_1D
)
GROUP_CHARACTERISTIC_MAP_PATTERN: Final[re.Pattern[bytes]] = re.compile(
    b"GRUPPENKENNFELD" + PATTERN_2D
)
FUNCTION_PATTERN: Final[re.Pattern[bytes]] = re.compile(
    rb"FKT\s+(.*?)(?: \"(.*?)?\"(?: \"(.*?)?\")?)?$"
)


@dataclass
class BasicInformation:
    name: str
    comment: str
    description: str
    display_name: str
    function: str
    variants: dict[str, str]

    def unparse(self) -> str:
        """Returns the basic information as a string"""
        s: str
        if self.comment:
            s = (
                "\n".join((f"*{ln}" for ln in self.comment.splitlines()))
                + "\n"
            )
        else:
            s = ""
        s += f'   LANGNAME "{self.description}"\n'
        if self.function:
            s += f"   FUNKTION {self.function}\n"
        if self.display_name:
            s += f"   DISPLAYNAME {self.display_name}\n"
        for var_name, var_value in self.variants.items():
            s += f"   VAR {var_name}={var_value}\n"
        return s

    def parse(self, line: bytes, comment: bool = True) -> bool:
        """Parses a line of the file and updates the object

        Args:
            line: The line to parse

        Returns:
            True if the line was parsed successfully and False otherwise
        """
        if line.startswith(b"LANGNAME"):
            self.description = _parse_text(line)
        elif line.startswith(b"FUNKTION"):
            self.function = _parse_name_without_keyword(line)
        elif line.startswith((b"*", b"!", b".")):
            if comment:
                self.comment += (
                    line[1:].strip().decode(errors="replace") + "\n"
                )
            else:
                return False
        elif line.startswith(b"DISPLAYNAME"):
            self.display_name = _parse_text(line)
        elif line.startswith(b"VAR"):
            key, value = _parse_variants(line)
            self.variants[key] = value
        else:
            return False
        return True


@dataclass
class Parameter(BasicInformation):
    unit: str
    value: float
    text: str

    def __str__(self) -> str:
        s: str = f"FESTWERT {self.name}\n{self.unparse()}"
        s += f'   EINHEIT_W "{self.unit}"\n'
        if isnan(self.value):
            s += f'   TEXT "{self.text}"\n'
        else:
            s += f"   WERT {self.value}\n"
        return s + "END"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Parameter):
            return False
        return (
            self.name == other.name
            and self.value_or_text == other.value_or_text
        )

    @classmethod
    def from_bytestream(cls, stream: BytesReadable, line: bytes) -> Self:
        self = cls(
            name=_parse_name_without_keyword(line),
            description="",
            display_name="",
            function="",
            unit="",
            value=nan,
            text="",
            variants={},
            comment="",
        )
        while not (line := stream.readline().strip()).startswith(b"END"):
            if self.parse(line):
                continue
            elif line.startswith(b"WERT"):
                self.value = float(line.removeprefix(b"WERT"))
            elif line.startswith(b"EINHEIT_W"):
                self.unit = _parse_text(line)
            elif line.startswith(b"TEXT"):
                self.text = _parse_text(line)
            else:
                logger.warning(f"Unknown parameter field: {line}")
        return self

    @property
    def value_or_text(self) -> str | float:
        return self.value if not isnan(self.value) else self.text

    def as_number(self, raise_on_nan: bool = True) -> int | float | bool:
        if isnan(self.value):
            text: str = self.text
            if PURE_INT_PATTERN.fullmatch(text):
                return int(text)
            elif PURE_FLOAT_PATTERN.fullmatch(text):
                return float(text)
            elif text == "true":
                return True
            elif text == "false":
                return False
            elif raise_on_nan:
                raise ValueError(
                    f"Cannot interpret value or text as number: {self.value}, {self.text}"
                )
            return nan
        else:
            return self.value

    def as_int(self, raise_on_nan: bool = True) -> int:
        return int(self.as_number(raise_on_nan))

    def as_float(self, raise_on_nan: bool = True) -> float:
        return float(self.as_number(raise_on_nan))

    def as_bool(self, raise_on_nan: bool = True) -> bool:
        number = self.as_number(raise_on_nan=False)
        if isnan(number):
            if raise_on_nan:
                raise ValueError(
                    f"Cannot interpret value or text as boolean: {self.value}, {self.text}"
                )
            return False
        return bool(number)

    def as_bin(self, raise_on_nan: bool = True) -> list[int]:
        n: int = self.as_int(raise_on_nan)
        positions: list[int] = []
        position = 0
        while n:
            if n & 1:  # check if the least significant bit is set
                positions.append(position)
            n >>= 1  # shift right
            position += 1
        return positions

    def as_hex(self, raise_on_nan: bool = True) -> list[int]:
        i: int = self.as_int(raise_on_nan)
        if i == 0:
            return [0]
        hex_digits: list[int] = []
        while i > 0:
            hex_digits.append(i % 16)
            i //= 16
        return hex_digits


@dataclass
class ParameterBlock(BasicInformation):
    unit: str
    array: npt.NDArray[FloatDtype]

    def __str__(self) -> str:
        if self.array.ndim == 1:
            x_dimension: int = self.array.size
            y_dimension: Optional[int] = None
        else:
            y_dimension, x_dimension = self.array.shape

        s: str = f"FESTWERTEBLOCK {self.name} {x_dimension}"
        if y_dimension is not None:
            s += f" @ {y_dimension}\n"
        else:
            s += "\n"
        s += self.unparse()
        s += f'   EINHEIT_W "{self.unit}"\n'

        stringifier: Callable[[object], str]
        if np.issubdtype(self.array.dtype, np.number):
            classifier: str = "WERT"
            stringifier = str
        else:
            classifier = "TEXT"
            stringifier = lambda x: f'"{x}"'  # noqa: E731

        if self.array.ndim == 1:
            s += (
                f"   {classifier} "
                + " ".join((stringifier(x) for x in self.array.flatten()))
                + "\n"
            )
        else:
            for line in self.array:
                s += (
                    f"   {classifier} "
                    + " ".join((stringifier(x) for x in line))
                    + "\n"
                )
        return s + "END"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParameterBlock):
            return False
        return bool(
            self.name == other.name
            and np.array_equal(self.array, other.array)
        )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, name: str) -> "ParameterBlock":
        y_dimension, x_dimension = df.shape
        if x_dimension == 1 or y_dimension == 1:
            array = np.asarray(df.values).flatten()
        else:
            array = (
                np.asarray(df.values)
                .flatten()
                .reshape(y_dimension, x_dimension)
            )
        return cls(
            name=name,
            unit="",
            description="",
            display_name="",
            function="",
            comment="",
            variants={},
            array=array,
        )

    @classmethod
    def from_bytestream(cls, stream: BytesReadable, line: bytes) -> Self:
        m: Optional[re.Match[bytes]] = PARAMETER_BLOCK_PATTERN.search(
            line.strip()
        )
        if m is None:
            raise ValueError(f"Invalid block: {line}")
        name: str = m.group(1).decode(errors="replace")
        x_dimension: int = int(m.group(2))
        y_dimension_str: Optional[bytes] = m.group(3)
        if y_dimension_str is not None:
            y_dimension: Optional[int] = int(y_dimension_str)
        else:
            y_dimension = None

        self = cls(
            name=name,
            description="",
            display_name="",
            function="",
            unit="",
            variants={},
            comment="",
            array=EMPTY_ARRAY,
        )
        values: list[str | float] = []
        while not (line := stream.readline().strip()).startswith(b"END"):
            if self.parse(line):
                continue
            elif line.startswith(b"WERT"):
                values.extend(_parse_values_without_keyword(line))
            elif line.startswith(b"TEXT"):
                values.extend(
                    (
                        match.decode(errors="replace")
                        for match in TEXT_PATTERN.findall(line)
                    )
                )
            elif line.startswith(b"EINHEIT_W"):
                self.unit = _parse_text(line)
            else:
                logger.warning(f"Unknown parameter field: {line}")
        if not values:
            raise ValueError("No values or texts found in block")
        if y_dimension is None:
            self.array = np.array(values).reshape(x_dimension)
        else:
            self.array = np.array(values).reshape(y_dimension, x_dimension)
        return self

    @property
    def values_flat(self) -> list[float]:
        if np.issubdtype(self.array.dtype, np.number):
            return [
                float(item)
                for item in self.array.flatten().tolist()
                if np.issubdtype(type(item), np.number)
            ]
        else:
            raise ValueError("Array does not contain numbers")

    @property
    def texts_flat(self) -> list[str]:
        if np.issubdtype(self.array.dtype, np.number):
            raise ValueError("Array does not contain strings")
        return [str(item) for item in self.array.flatten().tolist()]


@dataclass
class Textstring(BasicInformation):
    text: str

    def __str__(self) -> str:
        s: str = f"TEXTSTRING {self.name}\n{self.unparse()}"
        s += f'   TEXT "{self.text}"\n'
        return s + "END"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Textstring):
            return False
        return self.name == other.name and self.text == other.text

    @classmethod
    def from_bytestream(cls, stream: BytesReadable, line: bytes) -> Self:
        self = cls(
            name=_parse_name_without_keyword(line),
            description="",
            display_name="",
            function="",
            text="",
            variants={},
            comment="",
        )
        while not (line := stream.readline().strip()).startswith(b"END"):
            if self.parse(line):
                continue
            elif line.startswith(b"TEXT"):
                self.text += _parse_text(line)
            else:
                logger.warning(f"Unknown parameter field: {line}")
        return self


@dataclass
class Distribution(BasicInformation):
    unit_x: str = ""
    values: list[float] = field(default_factory=list)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Distribution):
            return False
        return self.name == other.name and self.values == other.values

    def __str__(self) -> str:
        s: str = (
            f"STUETZSTELLENVERTEILUNG {self.name} {len(self.values)}\n{self.unparse()}"
        )
        s += f'   EINHEIT_X "{self.unit_x}"\n'
        if self.values:
            s += f"   ST/X {' '.join((str(x) for x in self.values))}\n"
        return s + "END"

    @classmethod
    def from_bytestream(cls, stream: BytesReadable, line: bytes) -> Self:
        m: re.Match[bytes] | None = DISTRIBUTION_PATTERN.search(line.strip())
        if m is None:
            raise ValueError(f"Invalid distribution: {line}")
        name: str = m.group(1).decode(errors="replace")
        x_dimension: int = int(m.group(2))

        self = cls(
            name=name,
            description="",
            display_name="",
            function="",
            unit_x="",
            values=[],
            variants={},
            comment="",
        )
        while not (line := stream.readline().strip()).startswith(b"END"):
            if self.parse(line):
                continue
            elif line.startswith(b"ST/X"):
                self.values.extend(_parse_values_without_keyword(line))
            elif line.startswith(b"EINHEIT_X"):
                self.unit_x = _parse_text(line)
            else:
                logger.warning(f"Unknown parameter field: {line}")
        if len(self.values) != x_dimension:
            raise ValueError(
                f"Number of values does not match x dimension: {name}"
            )
        return self


@dataclass
class Function:
    name: str = ""
    description: str = ""
    version: str = ""

    def __str__(self) -> str:
        return f'   FKT {self.name} "{self.description}" "{self.version}"'

    @classmethod
    def from_bytestream(cls, stream: BytesReadable) -> list[Self]:
        functions: list[Self] = []
        while not (line := stream.readline().strip()).startswith(b"END"):
            function_match: re.Match[bytes] | None = FUNCTION_PATTERN.search(
                line
            )
            if function_match is None:
                continue
            functions.append(
                cls(
                    name=function_match.group(1).decode(errors="replace"),
                    description=(
                        function_match.group(2).decode(errors="replace")
                        if function_match.group(2)
                        else ""
                    ),
                    version=(
                        function_match.group(3).decode(errors="replace")
                        if function_match.group(3)
                        else ""
                    ),
                )
            )

        return functions


@dataclass
class CharacteristicLine(BasicInformation):
    unit_x: str
    unit_values: str
    series: "pd.Series[float]"

    @property
    def classifier(self) -> LiteralString:
        return "KENNLINIE"

    def __str__(self) -> str:
        s: str = (
            f"{self.classifier} {self.name} {len(self.series)}\n{self.unparse()}"
        )
        s += f'   EINHEIT_X "{self.unit_x}"\n'
        s += f'   EINHEIT_W "{self.unit_values}"\n'
        if self.x_mapping:
            s += f"*SSTX {self.x_mapping}\n"

        s += f"   ST/X {' '.join(str(x) for x in self.series.index)}\n"
        s += f"   WERT {' '.join(str(x) for x in self.series.values)}\n"
        for name, var in self.variants.items():
            s += f"   VAR {name}={var}" + "\n"
        return s + "END"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CharacteristicLine):
            return False
        return bool(
            self.name == other.name and self.series.equals(other.series)
        )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, name: str) -> Self:
        if df.shape[0] == 1:
            index: list[float] = df.columns.astype(FloatDtype).tolist()
            index_name: str = df.columns.name or df.index.name
        elif df.shape[1] == 1:
            index: list[float] = df.index.astype(FloatDtype).tolist()
            index_name: str = df.index.name or df.columns.name
        else:
            raise ValueError("DataFrame must have exactly one row or column")

        series = pd.Series(df.values.flatten(), index=index, name=name)
        series.index.name = index_name
        return cls(
            name=name,
            series=series,
            description="",
            display_name="",
            function="",
            unit_x="",
            unit_values="",
            comment="",
            variants={},
        )

    @classmethod
    def from_bytestream(
        cls, stream: BytesReadable, line: bytes, pattern: re.Pattern[bytes]
    ) -> Self:
        m: re.Match[bytes] | None = pattern.search(line)
        if m is None:
            raise ValueError(f"Invalid characteristic map: {line}")

        name: str = m.group(1).decode(errors="replace")
        x_dimension: int = int(m.group(2))
        self = cls(
            name=name,
            description="",
            display_name="",
            function="",
            unit_x="",
            unit_values="",
            variants={},
            comment="",
            series=EMPTY_SERIES,
        )

        index_name: str = ""
        index: list[float] = []
        values: list[float] = []

        while not (line := stream.readline().strip()).startswith(b"END"):
            if self.parse(line, comment=False):
                continue
            elif line.startswith(b"WERT"):
                values.extend(_parse_values_without_keyword(line))
            elif line.startswith(b"ST/X"):
                index.extend(_parse_values_without_keyword(line))
            elif line.startswith(b"EINHEIT_W"):
                self.unit_values = _parse_text(line)
            elif line.startswith(b"EINHEIT_X"):
                self.unit_x = _parse_text(line)
            elif line.startswith((b"*", b"!", b".")):
                re_match_x: re.Match[bytes] | None = AXIS_X_PATTERN.search(
                    line
                )
                if re_match_x:
                    index_name = re_match_x.group(1).decode(errors="replace")
                else:
                    self.comment += (
                        line[1:].strip().decode(errors="replace") + "\n"
                    )
            else:
                logger.warning(f"Unknown parameter field: {line}")
        if len(values) != x_dimension:
            raise ValueError(
                f"Number of values does not match x dimension: {name}"
            )
        self.series = pd.Series(values, index=index, name=name)
        self.series.index.name = index_name
        return self

    @property
    def index(self) -> list[float]:
        return self.series.index.astype(FloatDtype).tolist()

    @property
    def x_mapping(self) -> str:
        return self.series.index.name

    @property
    def as_function(self) -> CurveFunction:
        return apply_curve(self.series)

    def apply(self, x: pd.Series) -> pd.Series:
        return pd.Series(
            np.asarray(apply_curve(self.series)(x)),
            index=x.index,
            name=self.name,
        )

    def to_figure(self, fig=None, ax=None, **kwargs):
        """Plots the characteristic line

        Args:
            ax: Axis to plot the line on
            **kwargs: Additional keyword arguments for the plot function
        """

        if ax is None:
            ax = plt.gca()

        if fig is None:
            fig = plt.gcf()

        index: list[float] = self.index
        values: list[float] = np.asarray(self.series).ravel().tolist()
        ax.plot(index, values, **kwargs)
        ax.set_xlabel(f"{self.x_mapping} [{self.unit_x}]")
        ax.set_ylabel(f"{self.name} [{self.unit_values}]")
        ax.grid(True)
        fig.suptitle(
            f"{self.name} [{self.function}]",
            fontsize=int(1.2 * kwargs.get("fontsize", 10)),
        )
        ax.set_title(
            self.description + self.comment,
            fontsize=kwargs.get("fontsize", 10),
        )
        fig.tight_layout()
        return fig, ax

    def to_series(self) -> pd.Series:
        return self.series


@dataclass
class FixedCharacteristicLine(CharacteristicLine):
    @property
    def classifier(self) -> LiteralString:
        return "FESTKENNLINIE"


@dataclass
class GroupCharacteristicLine(CharacteristicLine):
    @property
    def classifier(self) -> LiteralString:
        return "GRUPPENKENNLINIE"


@dataclass
class CharacteristicMap(BasicInformation):
    dataframe: pd.DataFrame
    unit_x: str
    unit_y: str
    unit_values: str

    @property
    def classifier(self) -> LiteralString:
        return "KENNFELD"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CharacteristicMap):
            return False
        return bool(
            self.name == other.name
            and self.dataframe.equals(other.dataframe)
        )

    def __str__(self) -> str:
        y_dimension, x_dimension = self.dataframe.shape
        s: str = (
            f"{self.classifier} {self.name} {x_dimension} {y_dimension}"
            + "\n"
        ) + self.unparse()
        s += f'   EINHEIT_X "{self.unit_x}"' + "\n"
        s += f'   EINHEIT_Y "{self.unit_y}"' + "\n"
        s += f'   EINHEIT_W "{self.unit_values}"' + "\n"
        if self.x_mapping:
            s += f"*SSTX {self.x_mapping}" + "\n"
        if self.y_mapping:
            s += f"*SSTY {self.y_mapping}" + "\n"
        s += (
            "   ST/X "
            + " ".join(str(x) for x in self.dataframe.columns)
            + "\n"
        )
        for y, z in self.dataframe.iterrows():
            s += f"   ST/Y {y}" + "\n"
            s += f"   WERT {' '.join(str(x) for x in z)}" + "\n"
        for var_name, var_value in self.variants.items():
            s += f"   VAR {var_name}={var_value}" + "\n"
        return s + "END"

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, name: str) -> Self:
        return cls(
            dataframe=df,
            name=name,
            description="",
            display_name="",
            function="",
            unit_x="",
            unit_y="",
            unit_values="",
            variants={},
            comment="",
        )

    @classmethod
    def from_bytestream(
        cls,
        stream: BytesReadable,
        line: bytes,
        pattern: re.Pattern[bytes],
    ) -> Self:
        m: re.Match[bytes] | None = pattern.search(line)
        if m is None:
            raise ValueError(f"Invalid characteristic map: {line}")
        name: str = m.group(1).decode(errors="replace")
        x_dimension: int = int(m.group(2))
        y_dimension: int = int(m.group(3))

        index_name: str = ""
        column_name: str = ""
        columns: list[float] = []
        index: list[float] = []
        values: list[float] = []

        self = cls(
            name=name,
            dataframe=EMPTY_DATAFRAME,
            description="",
            display_name="",
            function="",
            unit_x="",
            unit_y="",
            unit_values="",
            variants={},
            comment="",
        )

        while not (line := stream.readline().strip()).startswith(b"END"):
            if self.parse(line, comment=False):
                continue
            elif line.startswith(b"WERT"):
                values.extend(_parse_values_without_keyword(line))
            elif line.startswith(b"ST/X"):
                columns.extend(_parse_values_without_keyword(line))
            elif line.startswith(b"ST/Y"):
                index.append(float(line.removeprefix(b"ST/Y")))
            elif line.startswith(b"EINHEIT_W"):
                self.unit_values = _parse_text(line)
            elif line.startswith(b"EINHEIT_X"):
                self.unit_x = _parse_text(line)
            elif line.startswith(b"EINHEIT_Y"):
                self.unit_y = _parse_text(line)
            elif line.startswith((b"*", b"!", b".")):
                mx: re.Match[bytes] | None = AXIS_X_PATTERN.search(line)
                if mx:
                    column_name = mx.group(1).decode(errors="replace")
                else:
                    my: re.Match[bytes] | None = AXIS_Y_PATTERN.search(line)
                    if my:
                        index_name = my.group(1).decode(errors="replace")
                    else:
                        self.comment += (
                            line[1:].strip().decode(errors="replace") + "\n"
                        )
            else:
                logger.warning(f"Unknown parameter field: {line}")

        if len(index) != y_dimension:
            raise ValueError(
                f"Number of values does not match y dimension: {name}"
            )
        if len(columns) != x_dimension:
            raise ValueError(
                f"Number of values does not match x dimension: {name}"
            )
        self.dataframe = pd.DataFrame(
            np.asarray(values).reshape(len(index), len(columns)),
            index=index,
            columns=columns,
        )
        self.dataframe.index.name = index_name
        self.dataframe.columns.name = column_name
        return self

    @property
    def index(self) -> list[float]:
        return self.dataframe.index.astype(FloatDtype).tolist()

    @property
    def columns(self) -> list[float]:
        return self.dataframe.columns.astype(FloatDtype).tolist()

    @property
    def as_function(self) -> MapFunction:
        return apply_map(self.dataframe)

    @property
    def x_mapping(self) -> str:
        return self.dataframe.columns.name

    @property
    def y_mapping(self) -> str:
        return self.dataframe.index.name

    def apply(self, x: pd.Series, y: pd.Series) -> pd.Series:
        return pd.Series(
            np.asarray(apply_map(self.dataframe)(x, y)),
            index=x.index,
            name=self.name,
        )

    def to_figure(self, fig=None, ax=None, **kwargs):
        """Plots the characteristic map as a heatmap

        Args:
            ax: Axis to plot the map on
            **kwargs: Additional keyword arguments for the plot function
        """

        if ax is None:
            ax = plt.gca()

        if fig is None:
            fig = plt.gcf()

        x_values: list[float] = self.columns
        y_values: list[float] = self.index
        z_values: npt.NDArray[FloatDtype] = self.dataframe.values

        c = ax.pcolormesh(
            x_values, y_values, z_values, shading="auto", **kwargs
        )
        ax.set_xlabel(f"{self.x_mapping} [{self.unit_x}]")
        ax.set_ylabel(f"{self.y_mapping} [{self.unit_y}]")
        plt.colorbar(c, ax=ax, label=self.unit_values)
        ax.grid(True)
        fig.suptitle(
            f"{self.name} [{self.function}]",
            fontsize=int(1.2 * kwargs.get("fontsize", 10)),
        )
        ax.set_title(
            self.description + self.comment,
            fontsize=kwargs.get("fontsize", 10),
        )
        fig.tight_layout()
        return fig, ax

    def to_dataframe(self) -> pd.DataFrame:
        return self.dataframe


@dataclass
class FixedCharacteristicMap(CharacteristicMap):
    @property
    def classifier(self) -> LiteralString:
        return "FESTKENNFELD"


@dataclass
class GroupCharacteristicMap(CharacteristicMap):
    @property
    def classifier(self) -> LiteralString:
        return "GRUPPENKENNFELD"


@dataclass
class DCM:
    """Parser for the DCM (Data Conservation Format) format used by e.g. Vector, ETAS,..."""

    file_header: bytearray = field(default_factory=bytearray)
    other_comments: bytearray = field(default_factory=bytearray)
    dcm_format: bytearray = field(default_factory=bytearray)
    functions: dict[str, Function] = field(default_factory=dict)
    text_strings: dict[str, Textstring] = field(default_factory=dict)
    parameters: dict[str, Parameter] = field(default_factory=dict)
    parameter_blocks: dict[str, ParameterBlock] = field(default_factory=dict)
    curves: dict[str, CharacteristicLine] = field(default_factory=dict)
    maps: dict[str, CharacteristicMap] = field(default_factory=dict)
    distributions: dict[str, Distribution] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path_or_file: PathOrReadable) -> Self:
        return cls().read(path_or_file)

    def __str__(self) -> str:
        # Print the file header

        ls: str = "\n"
        ls2: str = "\n" * 2
        ls3: str = "\n" * 3

        output = (
            ""
            + ls.join(
                (
                    f"* {line.decode(errors='replace')}"
                    for line in self.file_header.splitlines()
                )
            )
            + ls2
        )
        # Print the file version
        output += f"KONSERVIERUNG_FORMAT {self.dcm_format or 2.0}" + ls2

        # Print the other comments
        output += (
            ls.join(
                (
                    f"* {line.decode(errors='replace')}"
                    for line in self.other_comments.splitlines()
                )
            )
            + ls3
        )

        # Print the functions list
        output += (
            "FUNKTIONEN"
            + ls
            + ls.join((str(func) for func in self.functions.values()))
            + ls
            + "END"
            + ls2
        )

        # Print rest of DCM objects
        for objs in (
            self.parameters,
            self.parameter_blocks,
            self.curves,
            self.maps,
            self.text_strings,
            self.distributions,
        ):
            output += ls2.join((str(obj) for obj in objs.values())) + ls2
        return output

    def __getitem__(self, key: str) -> BasicInformation:
        if key in self.text_strings:
            return self.text_strings[key]
        if key in self.parameters:
            return self.parameters[key]
        if key in self.parameter_blocks:
            return self.parameter_blocks[key]
        if key in self.curves:
            return self.curves[key]
        if key in self.maps:
            return self.maps[key]
        if key in self.distributions:
            return self.distributions[key]
        raise KeyError(f"Key not found: {key}")

    def __setitem__(self, key: str, value: BasicInformation) -> None:
        if isinstance(value, Textstring):
            self.text_strings[key] = value
        elif isinstance(value, Parameter):
            self.parameters[key] = value
        elif isinstance(value, ParameterBlock):
            self.parameter_blocks[key] = value
        elif isinstance(value, CharacteristicLine):
            self.curves[key] = value
        elif isinstance(value, CharacteristicMap):
            self.maps[key] = value
        elif isinstance(value, Distribution):
            self.distributions[key] = value
        else:
            raise ValueError(f"Unsupported type: {type(value)}")

    def __or__(self, dcm: "DCM") -> "DCM":
        return self.shallow_transfer(
            text_strings=self.text_strings | dcm.text_strings,
            parameters=self.parameters | dcm.parameters,
            parameter_blocks=self.parameter_blocks | dcm.parameter_blocks,
            curves=self.curves | dcm.curves,
            maps=self.maps | dcm.maps,
            distributions=self.distributions | dcm.distributions,
            functions=self.functions | dcm.functions,
        )

    def __sub__(self, other: "DCM") -> "DCM":
        def sub(left: dict[K, V], right: dict[K, V]) -> dict[K, V]:
            return {k: v for k, v in left.items() if k not in right}

        return self.shallow_transfer(
            text_strings=sub(self.text_strings, other.text_strings),
            parameters=sub(self.parameters, other.parameters),
            parameter_blocks=sub(
                self.parameter_blocks, other.parameter_blocks
            ),
            curves=sub(self.curves, other.curves),
            maps=sub(self.maps, other.maps),
            distributions=sub(self.distributions, other.distributions),
            functions={
                obj.function: self.functions[obj.function]
                for obj in self.objects.values()
                if obj.function in self.functions
            },
        )

    def __and__(self, other: "DCM") -> "DCM":
        def and_(left: dict[K, V], right: dict[K, V]) -> dict[K, V]:
            return {
                k: v for k, v in left.items() if k in right and v == right[k]
            }

        return self.shallow_transfer(
            text_strings=and_(self.text_strings, other.text_strings),
            parameters=and_(self.parameters, other.parameters),
            parameter_blocks=and_(
                self.parameter_blocks, other.parameter_blocks
            ),
            curves=and_(self.curves, other.curves),
            maps=and_(self.maps, other.maps),
            distributions=and_(self.distributions, other.distributions),
            functions={
                obj.function: self.functions[obj.function]
                for obj in self.objects.values()
                if obj.function in self.functions
            },
        )

    def __mod__(self, other: "DCM") -> "DCM":
        def mod(left: dict[K, V], right: dict[K, V]) -> dict[K, V]:
            return {
                k: v for k, v in left.items() if k in right and v != right[k]
            }

        return self.shallow_transfer(
            text_strings=mod(self.text_strings, other.text_strings),
            parameters=mod(self.parameters, other.parameters),
            parameter_blocks=mod(
                self.parameter_blocks, other.parameter_blocks
            ),
            curves=mod(self.curves, other.curves),
            maps=mod(self.maps, other.maps),
            distributions=mod(self.distributions, other.distributions),
            functions={
                obj.function: self.functions[obj.function]
                for obj in self.objects.values()
                if obj.function in self.functions
            },
        )

    @property
    def keys(self) -> set[str]:
        return set(
            itertools.chain(
                self.functions.keys(),
                self.text_strings.keys(),
                self.parameters.keys(),
                self.parameter_blocks.keys(),
                self.curves.keys(),
                self.maps.keys(),
                self.distributions.keys(),
            )
        )

    @property
    def lines(self) -> dict[str, CharacteristicLine]:
        return self.curves

    @property
    def objects(self) -> dict[str, BasicInformation]:
        return dict(
            {
                **self.text_strings,
                **self.parameters,
                **self.parameter_blocks,
                **self.curves,
                **self.maps,
                **self.distributions,
            }
        )

    def load_from_excel(
        self,
        maps_path: PathOrReadable = "Map.xlsx",
        curves_path: PathOrReadable = "Curve.xlsx",
        parameters_path: PathOrReadable = "Parameter.xlsx",
        parameter_blocks_path: PathOrReadable = "ParameterBlock.xlsx",
    ) -> Self:
        """Loads maps, curves, parameters and parameter blocks from excel files

        Args:
            maps_path (str): Path to the excel file containing the maps
            curves_path (str): Path to the excel file containing the curves
            parameters_path (str): Path to the excel file containing the parameters
            parameter_blocks_path (str): Path to the excel file containing the parameter blocks
        """
        return (
            self.load_maps(maps_path)
            .load_lines(curves_path)
            .load_parameters(parameters_path)
            .load_parameter_blocks(parameter_blocks_path)
        )

    def load_maps(self, excel_path: PathOrReadable) -> Self:
        with _open_stream(excel_path) as excel_file:
            if excel_file is None:
                return self
            excel = pd.ExcelFile(excel_file)
            for sheet in excel.sheet_names:
                if isinstance(sheet, str):
                    df = excel.parse(sheet, index_col=0)
                    if df.empty:
                        continue
                    char_map: CharacteristicMap = (
                        CharacteristicMap.from_dataframe(df, sheet)
                    )
                    self.maps[char_map.name] = char_map
        return self

    def load_curves(self, excel_path: PathOrReadable) -> Self:
        with _open_stream(excel_path) as excel_file:
            if excel_file is None:
                return self
            excel = pd.ExcelFile(excel_file)
            for sheet in excel.sheet_names:
                if isinstance(sheet, str):
                    df = excel.parse(sheet, index_col=0)
                    if df.empty:
                        continue
                    char_line: CharacteristicLine = (
                        CharacteristicLine.from_dataframe(df, sheet)
                    )
                    self.curves[char_line.name] = char_line
        return self

    @property
    def load_lines(self) -> Callable[[PathOrReadable], Self]:
        # Alias for load_curves
        return self.load_curves

    def load_parameter_blocks(self, excel_path: PathOrReadable) -> Self:
        with _open_stream(excel_path) as excel_file:
            if excel_file is None:
                return self
            excel = pd.ExcelFile(excel_file)
            for sheet in excel.sheet_names:
                if isinstance(sheet, str):
                    df = excel.parse(sheet, header=None)
                    if df.empty:
                        continue
                    block: ParameterBlock = ParameterBlock.from_dataframe(
                        df, sheet
                    )
                    self.parameter_blocks[block.name] = block
        return self

    def load_parameters(self, excel_path: PathOrReadable) -> Self:
        with _open_stream(excel_path) as excel_file:
            if excel_file is None:
                return self
            df: pd.DataFrame = pd.read_excel(excel_file)
            for _, row in df.iterrows():
                param: Parameter = Parameter(
                    name=str(row["name"]),
                    description=str(row.get("description", "")),
                    display_name=str(row.get("display_name", "")),
                    function=str(row.get("function", "")),
                    unit=str(row.get("unit", "")),
                    value=float(row.get("value", nan)),
                    text=str(row.get("text", "")),
                    comment=str(row.get("comment", "")),
                    variants={},
                )
                self.parameters[param.name] = param
        return self

    def write(
        self,
        path_or_file: FileDescriptorOrPath | Writable,
        file_encoding: str = "utf-8",
    ) -> None:
        content: str = str(self)
        if isinstance(path_or_file, StringWritable):
            path_or_file.write(content)
        elif isinstance(path_or_file, BytesWritable):
            path_or_file.write(content.encode(file_encoding))
        else:
            with open(path_or_file, "wb") as f:
                f.write(content.encode(file_encoding))

    def read(self, path_or_file: PathOrReadable) -> Self:
        line: bytes
        dcm_format: re.Match[bytes] | None = None
        is_file_header_finished: bool = False

        with _open_stream(path_or_file) as stream:
            if stream is None:
                raise FileNotFoundError(f"File not found: {path_or_file}")
            for line in stream:
                # Check if empty line
                line = line.strip()
                if not line:
                    continue

                # Check if line is comment
                elif line.startswith((b"*", b"!", b".")):
                    if not is_file_header_finished:
                        self.file_header.extend(line[1:] + b"\n")
                    else:
                        self.other_comments.extend(line[1:] + b"\n")
                    continue
                else:
                    # At this point first comment block passed
                    is_file_header_finished = True

                # Check if format version line
                if dcm_format is None:
                    if line.startswith(b"KONSERVIERUNG_FORMAT"):
                        dcm_format = VERSION_PATTERN.search(line)
                        if dcm_format:
                            self.dcm_format.clear()
                            self.dcm_format.extend(dcm_format.group(1))
                            continue
                        else:
                            raise Exception(
                                "Incorrect file structure. DCM file format missing!"
                            )
                    else:
                        raise Exception(
                            "Incorrect file structure. DCM file format missing!"
                        )

                parameter: Parameter
                block_parameter: ParameterBlock
                function: Function
                distribution: Distribution
                text_string: Textstring

                char_line: CharacteristicLine
                fixed_line: FixedCharacteristicLine
                group_line: GroupCharacteristicLine

                char_map: CharacteristicMap
                fixed_map: FixedCharacteristicMap
                group_map: GroupCharacteristicMap

                if line.startswith(b"FESTWERTEBLOCK"):
                    block_parameter = ParameterBlock.from_bytestream(
                        stream, line
                    )
                    self.parameter_blocks[block_parameter.name] = (
                        block_parameter
                    )
                elif line.startswith(b"FESTWERT"):
                    parameter = Parameter.from_bytestream(stream, line)
                    self.parameters[parameter.name] = parameter
                elif line.startswith(b"FUNKTIONEN"):
                    for function in Function.from_bytestream(stream):
                        self.functions[function.name] = function
                elif line.startswith(b"KENNLINIE"):
                    char_line = CharacteristicLine.from_bytestream(
                        stream,
                        line,
                        CHARACTERISTIC_LINE_PATTERN,
                    )
                    self.curves[char_line.name] = char_line
                elif line.startswith(b"FESTKENNLINIE"):
                    fixed_line = FixedCharacteristicLine.from_bytestream(
                        stream,
                        line,
                        FIXED_CHARACTERISTIC_LINE_PATTERN,
                    )
                    self.curves[fixed_line.name] = fixed_line
                elif line.startswith(b"GRUPPENKENNLINIE"):
                    group_line = GroupCharacteristicLine.from_bytestream(
                        stream,
                        line,
                        GROUP_CHARACTERISTIC_LINE_PATTERN,
                    )
                    self.curves[group_line.name] = group_line
                elif line.startswith(b"KENNFELD"):
                    char_map = CharacteristicMap.from_bytestream(
                        stream,
                        line,
                        CHARACTERISTIC_MAP_PATTERN,
                    )
                    self.maps[char_map.name] = char_map
                elif line.startswith(b"FESTKENNFELD"):
                    fixed_map = FixedCharacteristicMap.from_bytestream(
                        stream,
                        line,
                        FIXED_CHARACTERISTIC_MAP_PATTERN,
                    )
                    self.maps[fixed_map.name] = fixed_map
                elif line.startswith(b"GRUPPENKENNFELD"):
                    group_map = GroupCharacteristicMap.from_bytestream(
                        stream,
                        line,
                        GROUP_CHARACTERISTIC_MAP_PATTERN,
                    )
                    self.maps[group_map.name] = group_map

                elif line.startswith(b"STUETZSTELLENVERTEILUNG"):
                    distribution = Distribution.from_bytestream(stream, line)
                    self.distributions[distribution.name] = distribution
                elif line.startswith(b"TEXTSTRING"):
                    text_string = Textstring.from_bytestream(stream, line)
                    self.text_strings[text_string.name] = text_string
                else:
                    # Unknown start of line
                    logger.warning(f"- Unknown line detected: {line}")

        return self

    def copy(self) -> "DCM":
        return copy.deepcopy(self)

    def shallow_transfer(
        self,
        text_strings: dict[str, Textstring],
        parameters: dict[str, Parameter],
        parameter_blocks: dict[str, ParameterBlock],
        curves: dict[str, CharacteristicLine],
        maps: dict[str, CharacteristicMap],
        distributions: dict[str, Distribution],
        functions: dict[str, Function],
    ) -> "DCM":
        return DCM(
            file_header=self.file_header,
            other_comments=self.other_comments,
            dcm_format=self.dcm_format,
            text_strings=text_strings,
            parameters=parameters,
            parameter_blocks=parameter_blocks,
            curves=curves,
            maps=maps,
            distributions=distributions,
            functions=functions,
        )


def apply_map(df: pd.DataFrame) -> MapFunction:
    def wrapper(
        x: npt.ArrayLike, y: npt.ArrayLike
    ) -> npt.NDArray[FloatDtype]:
        return lininterp2(
            X=df.columns.to_numpy(dtype=FloatDtype),
            Y=df.index.to_numpy(dtype=FloatDtype),
            V=df.to_numpy(dtype=FloatDtype),
            x=np.asarray(x, dtype=FloatDtype),
            y=np.asarray(y, dtype=FloatDtype),
        ).astype(FloatDtype)

    return wrapper


def apply_curve(series: pd.Series) -> CurveFunction:
    def wrapper(x: npt.ArrayLike) -> npt.NDArray[FloatDtype]:
        return lininterp1(
            X=series.index.to_numpy(dtype=FloatDtype),
            V=series.to_numpy(dtype=FloatDtype),
            x=np.asarray(x, dtype=FloatDtype),
        ).astype(FloatDtype)

    return wrapper


@contextmanager
def _open_stream(
    path_or_file: PathOrReadable,
) -> Iterator[Optional[BytesReadable]]:
    stream: Optional[BytesReadable] = None
    try:
        with suppress(BaseException):

            if isinstance(path_or_file, BytesReadable):
                stream = path_or_file
            elif isinstance(path_or_file, StringReadable):
                stream = BytesIO(path_or_file.read().encode("utf-8"))
            else:
                stream = open(path_or_file, "rb")
        yield stream
    finally:
        if stream is not None:
            stream.close()


def _parse_variants(line: bytes) -> tuple[str, str]:
    variant: Optional[re.Match[bytes]] = VARIANT_PATTERN.search(line.strip())
    if variant is None:
        raise ValueError(f"Invalid variant: {line}")
    return (
        variant.group(1).strip().decode(errors="replace"),
        variant.group(2).strip().decode(errors="replace"),
    )


def _parse_text(
    line: bytes, text_pattern: re.Pattern[bytes] = TEXT_PATTERN
) -> str:
    match: Optional[re.Match[bytes]] = text_pattern.search(line)
    if match is None:
        return ""
    return match.group(1).decode(errors="replace")


def _parse_name_without_keyword(line: bytes) -> str:
    return line.strip().split(maxsplit=1)[1].decode(errors="replace")


def _parse_values_without_keyword(line: bytes) -> Iterable[float]:
    return (float(value) for value in line.strip().split()[1:])
