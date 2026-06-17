import io
import pathlib
import struct

import numpy as np
import pytest

from stl import Mode, mesh

TESTS_PATH = pathlib.Path(__file__).parent


@pytest.mark.parametrize('mode', [Mode.BINARY, Mode.AUTOMATIC])
def test_ascii_like_binary(tmpdir, speedups, mode):
    _test(tmpdir, speedups, mode, False)
    _test(tmpdir, speedups, mode, True)


def test_binary_in_ascii_mode(tmpdir, speedups):
    with pytest.raises(RuntimeError):
        _test(tmpdir, speedups, mode=Mode.ASCII, use_filehandle=False)

    with pytest.raises(RuntimeError):
        _test(tmpdir, speedups, mode=Mode.ASCII, use_filehandle=True)


def _test(tmpdir, speedups, mode, use_filehandle=True):
    filename = TESTS_PATH / 'stl_binary' / 'rear_case.stl'
    if use_filehandle:
        with open(filename, 'rb') as fh:
            mesh.Mesh.from_file(filename, fh=fh, speedups=speedups, mode=mode)

        with open(filename, 'rb') as fh:
            # Test with BytesIO
            fh = io.BytesIO(fh.read())
            mesh.Mesh.from_file(filename, fh=fh, speedups=speedups, mode=mode)
    else:
        mesh.Mesh.from_file(filename, speedups=speedups, mode=mode)


@pytest.mark.parametrize('mode', [Mode.BINARY, Mode.AUTOMATIC])
def test_write_bytes_io(binary_file, mode):
    mesh_ = mesh.Mesh.from_file(binary_file)

    # Write to io.Bytes() in BINARY mode.
    fh = io.BytesIO()
    mesh_.save('mesh.stl', fh, mode=mode)

    assert len(fh.getvalue()) > 84
    assert fh.getvalue()[84:] == mesh_.data.tobytes()

    read = mesh.Mesh.from_file('nameless', fh=io.BytesIO(fh.getvalue()))
    assert np.allclose(read.vectors, mesh_.vectors)


def test_binary_file():
    list(mesh.Mesh.from_multi_file(TESTS_PATH / 'stl_tests' / 'triamid.stl'))


def test_binary_count_field_parsed_as_unsigned(speedups):
    # A count with the high bit set used to be unpacked as a negative
    # signed int, slip past the MAX_COUNT assertion, and make
    # np.fromfile(count=-1) silently read the whole file.
    record = b'\x00' * mesh.Mesh.dtype.itemsize
    for count in (0xFFFFFFFF, 2**31 + 5):
        raw = b'#' * 80 + struct.pack('<I', count) + record * 3
        with pytest.raises(AssertionError, match='too large'):
            mesh.Mesh.from_file(
                'huge.stl',
                fh=io.BytesIO(raw),
                mode=Mode.BINARY,
                speedups=speedups,
            )


def test_load_empty_file_raises_value_error(speedups):
    with pytest.raises(ValueError, match='empty'):
        mesh.Mesh.from_file('empty.stl', fh=io.BytesIO(b''), speedups=speedups)


class _NonSeekableStream(io.RawIOBase):
    """Minimal non-seekable binary stream, like a pipe."""

    def __init__(self, data: bytes) -> None:
        self._buffer = io.BytesIO(data)

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def seekable(self) -> bool:
        return False


def test_automatic_load_from_non_seekable_stream(binary_file, speedups):
    # HalfDonut's binary header starts with 'solid', so auto-detection
    # falls back from ASCII to binary, which needs to rewind. On a
    # pipe-like stream the data must be buffered instead of seeked.
    raw = pathlib.Path(binary_file).read_bytes()
    loaded = mesh.Mesh.from_file(
        'pipe.stl', fh=_NonSeekableStream(raw), speedups=speedups
    )
    assert len(loaded.data) > 0


class _DuckReader:
    """File-like object with only a read() method, no io.IOBase API."""

    def __init__(self, data: bytes) -> None:
        self._buffer = io.BytesIO(data)

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)


def test_load_from_duck_typed_reader(binary_file, speedups):
    # Custom file-likes without seekable()/fileno() must not crash
    # with AttributeError; they get buffered like pipes.
    raw = pathlib.Path(binary_file).read_bytes()
    loaded = mesh.Mesh.from_file(
        'duck.stl', fh=_DuckReader(raw), speedups=speedups
    )
    assert len(loaded.data) > 0


class _FailingHandle(io.RawIOBase):
    """File handle that fails every write with ENOSPC."""

    def writable(self) -> bool:
        return True

    def write(self, data: object) -> int:
        raise OSError(28, 'No space left on device')


def test_save_propagates_write_errors(binary_file, speedups):
    mesh_ = mesh.Mesh.from_file(binary_file, speedups=speedups)
    with pytest.raises(OSError, match='No space left'):
        mesh_.save('out.stl', fh=_FailingHandle(), mode=Mode.BINARY)
