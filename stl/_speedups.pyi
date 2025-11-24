from typing import IO

import numpy as np
from typing_extensions import Buffer, TypeAlias

_DataArray: TypeAlias = np.ndarray[tuple[int], np.dtype[np.void]]

def ascii_read(fh: IO[bytes], buf: Buffer) -> tuple[bytes, _DataArray]: ...
def ascii_write(fh: IO[bytes], name: bytes, data: _DataArray) -> None: ...
