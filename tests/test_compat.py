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
