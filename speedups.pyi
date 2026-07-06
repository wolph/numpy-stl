# Local stub for the optional speedups package so type
# checkers resolve it without it installed.
from typing import IO

import numpy as np
import numpy.typing as npt

def ascii_read(
    fh: IO[bytes],
    buf: bytes,
) -> tuple[bytes, npt.NDArray[np.void]]: ...
def ascii_write(
    fh: IO[bytes],
    name: bytes,
    arr: npt.NDArray[np.void],
) -> None: ...
