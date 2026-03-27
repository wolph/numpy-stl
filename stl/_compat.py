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
        from speedups import (  # noqa: F401
            ascii_read,  # type: ignore[assignment]
            ascii_write,  # type: ignore[assignment]
        )
    except ImportError:
        _speedups_available = False


def has_speedups() -> bool:
    """Return True when the external speedups package is installed."""
    return _speedups_available
