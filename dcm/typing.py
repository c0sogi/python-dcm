import os
from io import (
    BufferedReader,
    BufferedWriter,
    BytesIO,
    StringIO,
    TextIOWrapper,
)
from typing import (
    Callable,
    TypeAlias,
    TypeVar,
)

import numpy as np
import numpy.typing as npt

FloatDtype: TypeAlias = np.float64

FileDescriptorOrPath: TypeAlias = (
    int | str | bytes | os.PathLike[str] | os.PathLike[bytes]
)

BytesReadable: TypeAlias = BytesIO | BufferedReader
BytesWritable: TypeAlias = BytesIO | BufferedWriter
StringReadable: TypeAlias = StringIO | TextIOWrapper
StringWritable: TypeAlias = StringIO | TextIOWrapper

Readable: TypeAlias = BytesReadable | StringReadable
Writable: TypeAlias = BytesWritable | StringWritable

PathOrReadable: TypeAlias = FileDescriptorOrPath | Readable

K = TypeVar("K")
V = TypeVar("V")

MapFunction: TypeAlias = Callable[
    [npt.ArrayLike, npt.ArrayLike], npt.NDArray[FloatDtype]
]
CurveFunction: TypeAlias = Callable[[npt.ArrayLike], npt.NDArray[FloatDtype]]
