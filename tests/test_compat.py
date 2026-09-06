import importlib
import importlib.machinery
import sys
import types

from stl._compat import ascii_read, ascii_write, has_speedups


def test_has_speedups_returns_bool(speedups):
    result = has_speedups()
    assert isinstance(result, bool)


def test_ascii_read_type(speedups):
    # ascii_read is either None (no speedups) or callable
    if has_speedups():
        assert callable(ascii_read)
    else:
        assert ascii_read is None


def test_ascii_write_type(speedups):
    # ascii_write is either None (no speedups) or callable
    if has_speedups():
        assert callable(ascii_write)
    else:
        assert ascii_write is None


def test_has_speedups_consistent_with_exports(speedups):
    if has_speedups():
        assert ascii_read is not None
        assert ascii_write is not None
    else:
        assert ascii_read is None
        assert ascii_write is None


def test_missing_speedups_package(monkeypatch):
    """Verify _compat when no speedups package is installed at all."""
    saved_compat = sys.modules.pop('stl._compat', None)
    try:
        monkeypatch.setattr(importlib.util, 'find_spec', lambda name: None)
        compat = importlib.import_module('stl._compat')
        assert compat.ascii_read is None
        assert compat.ascii_write is None
        assert compat.has_speedups() is False
    finally:
        if saved_compat is not None:
            sys.modules['stl._compat'] = saved_compat
        else:
            sys.modules.pop('stl._compat', None)


def test_import_error_fallback():
    """Verify _compat gracefully handles a broken speedups package."""
    fake = types.ModuleType('speedups')
    fake.__path__ = []
    fake.__spec__ = importlib.machinery.ModuleSpec(
        'speedups',
        None,
        is_package=True,
    )

    # Evict a cached real ``speedups.stl`` as well, otherwise the
    # submodule import succeeds straight from sys.modules.
    saved_modules = {
        name: sys.modules.pop(name, None)
        for name in ('speedups', 'speedups.stl')
    }
    saved_compat = sys.modules.pop('stl._compat', None)
    try:
        sys.modules['speedups'] = fake
        compat = importlib.import_module('stl._compat')
        assert compat.ascii_read is None
        assert compat.ascii_write is None
        assert compat.has_speedups() is False
    finally:
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module
            else:
                sys.modules.pop(name, None)
        if saved_compat is not None:
            sys.modules['stl._compat'] = saved_compat
        else:
            sys.modules.pop('stl._compat', None)


def test_speedups_stl_submodule_exports():
    """Verify _compat binds the functions from ``speedups.stl``."""

    def fake_ascii_read(*args: object) -> tuple[bytes, None]:
        return b'', None

    def fake_ascii_write(*args: object) -> None:
        return None

    fake_pkg = types.ModuleType('speedups')
    fake_pkg.__path__ = []
    fake_pkg.__spec__ = importlib.machinery.ModuleSpec(
        'speedups',
        None,
        is_package=True,
    )
    fake_stl = types.ModuleType('speedups.stl')
    fake_stl.ascii_read = fake_ascii_read
    fake_stl.ascii_write = fake_ascii_write

    saved_modules = {
        name: sys.modules.get(name) for name in ('speedups', 'speedups.stl')
    }
    saved_compat = sys.modules.pop('stl._compat', None)
    try:
        sys.modules['speedups'] = fake_pkg
        sys.modules['speedups.stl'] = fake_stl
        compat = importlib.import_module('stl._compat')
        assert compat.has_speedups() is True
        assert compat.ascii_read is fake_ascii_read
        assert compat.ascii_write is fake_ascii_write
    finally:
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module
            else:
                sys.modules.pop(name, None)
        if saved_compat is not None:
            sys.modules['stl._compat'] = saved_compat
        else:
            sys.modules.pop('stl._compat', None)
