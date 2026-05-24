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


def test_import_error_fallback():
    """Verify _compat gracefully handles a broken speedups package."""
    fake = types.ModuleType('speedups')
    fake.__path__ = []
    fake.__spec__ = importlib.machinery.ModuleSpec(
        'speedups',
        None,
        is_package=True,
    )

    saved_module = sys.modules.get('speedups')
    saved_compat = sys.modules.pop('stl._compat', None)
    try:
        sys.modules['speedups'] = fake
        compat = importlib.import_module('stl._compat')
        assert compat.ascii_read is None
        assert compat.ascii_write is None
        assert compat.has_speedups() is False
    finally:
        if saved_module is not None:
            sys.modules['speedups'] = saved_module
        else:
            sys.modules.pop('speedups', None)
        if saved_compat is not None:
            sys.modules['stl._compat'] = saved_compat
        else:
            sys.modules.pop('stl._compat', None)
