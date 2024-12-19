import warnings
from dataclasses import dataclass, field
from typing import Final, Optional

import re
from .dcm import PURE_FLOAT_PATTERN, PURE_INT_PATTERN, FileDescriptorOrPath

sc_pattern: re.Pattern[bytes] = re.compile(
    rb'SYSTEM_CONSTANT\s+"(\w+)"\s+"([^"]+)"'
)


@dataclass
class SystemConstants:
    __constants__: dict[str, int | float | bool] = field(
        default_factory=dict
    )

    def __getattr__(self, name: str) -> int | float | bool:
        value: Optional[int | float | bool] = self.__constants__.get(name)
        if value is not None:
            return value
        warnings.warn(
            f"No System Constant found with name: {name}. Returning 0."
        )
        return 0

    def from_a2l(self, a2l_filepath: FileDescriptorOrPath) -> int:
        total_constants: int = 0

        with open(a2l_filepath, "rb") as f:
            for line in f:
                match: Optional[re.Match[bytes]] = sc_pattern.search(
                    line.strip()
                )
                if match is None:
                    continue
                bname, bvalue = match.groups()
                name: str = bname.decode(errors="replace")
                value: str = bvalue.decode(errors="replace")
                if PURE_INT_PATTERN.fullmatch(value):
                    self.__constants__[name] = int(value)
                elif PURE_FLOAT_PATTERN.fullmatch(value):
                    self.__constants__[name] = float(value)
                elif value == "true":
                    self.__constants__[name] = True
                elif value == "false":
                    self.__constants__[name] = False
                else:
                    warnings.warn(f"Invalid SYSTEM_CONSTANT {name}: {value}")
                    continue
                total_constants += 1
        return total_constants


SC: Final = SystemConstants()
