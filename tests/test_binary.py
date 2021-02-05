import io
import numpy
import pytest
import pathlib
from stl import mesh, Mode


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
            mesh.Mesh.from_file(filename, fh=fh, speedups=speedups,
                                mode=mode)

        with open(filename, 'rb') as fh:
            # Test with BytesIO
            fh = io.BytesIO(fh.read())
            mesh.Mesh.from_file(filename, fh=fh, speedups=speedups,
                                mode=mode)
    else:
        mesh.Mesh.from_file(filename,
                            speedups=speedups, mode=mode)


@pytest.mark.parametrize('mode', [Mode.BINARY, Mode.AUTOMATIC])
def test_write_bytes_io(binary_file, mode):
    mesh_ = mesh.Mesh.from_file(binary_file)

    # Write to io.Bytes() in BINARY mode.
    fh = io.BytesIO()
    mesh_.save('mesh.stl', fh, mode=mode)

    assert len(fh.getvalue()) > 84
    assert fh.getvalue()[84:] == mesh_.data.tobytes()

    read = mesh.Mesh.from_file('nameless', fh=io.BytesIO(fh.getvalue()))
    assert numpy.allclose(read.vectors, mesh_.vectors)


def test_binary_file():
    list(mesh.Mesh.from_multi_file(TESTS_PATH / 'stl_tests' / 'triamid.stl'))
