"""Compatibility layer for optional speedups package."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

_speedups_available: bool = importlib.util.find_spec('speedups') is not None

ascii_read: Any = None
ascii_write: Any = None

if _speedups_available:
    try:
        # speedups>=2.1.0 ships the STL functions in the
        # ``speedups.stl`` submodule. The 2.0.x top-level
        # re-exports no longer exist.
        from speedups.stl import (
            ascii_read as _speedups_ascii_read,
            ascii_write as _speedups_ascii_write,
        )
    except ImportError:
        _speedups_available = False
    else:
        ascii_read = _speedups_ascii_read
        ascii_write = _speedups_ascii_write


def has_speedups() -> bool:
    """Return True when the external speedups package is installed."""
    return _speedups_available
