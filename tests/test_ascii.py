# type: ignore[reportAttributeAccessIssue]
import io
import locale
import os
import pathlib
import subprocess
import sys
import warnings

import numpy as np
import pytest
from stl.utils import b

from stl import Mode, mesh

FILES_PATH = pathlib.Path(__file__).parent / 'stl_tests'


def test_ascii_file(speedups):
    filename = FILES_PATH / 'bwb.stl'
    mesh.Mesh.from_file(filename, speedups=speedups)


def test_chinese_name(tmpdir, speedups):
    name = 'Test Chinese name 月球'
    _stl_file = (
        f"""
    solid {name}
      facet normal -0.014565 0.073223 -0.002897
        outer loop
          vertex 0.399344 0.461940 1.044090
          vertex 0.500000 0.500000 1.500000
          vertex 0.576120 0.500000 1.117320
        endloop
      endfacet
    endsolid
    """
    ).lstrip()

    tmp_file = tmpdir.join('tmp.stl')
    with tmp_file.open('wb+') as fh:
        fh.write(b(_stl_file))
        fh.seek(0)
        test_mesh = mesh.Mesh.from_file(
            str(tmp_file), fh=fh, speedups=speedups
        )
        if speedups:
            assert test_mesh.name.lower() == b(name).lower()
        else:
            assert test_mesh.name == b(name)


def test_long_name(tmpdir, speedups):
    name = 'Just Some Very Long Name which will not fit within the standard'
    name += name
    _stl_file = (
        f"""
    solid {name}
      facet normal -0.014565 0.073223 -0.002897
        outer loop
          vertex 0.399344 0.461940 1.044090
          vertex 0.500000 0.500000 1.500000
          vertex 0.576120 0.500000 1.117320
        endloop
      endfacet
    endsolid
    """
    ).lstrip()

    tmp_file = tmpdir.join('tmp.stl')
    with tmp_file.open('wb+') as fh:
        fh.write(b(_stl_file))
        fh.seek(0)
        test_mesh = mesh.Mesh.from_file(
            str(tmp_file), fh=fh, speedups=speedups
        )

        if speedups:
            assert test_mesh.name.lower() == b(name).lower()
        else:
            assert test_mesh.name == b(name)


def test_scientific_notation(tmpdir, speedups):
    name = 'just some very long name which will not fit within the standard'
    name += name
    _stl_file = (
        f"""
    solid {name}
      facet normal 1.014565e-10 7.3223e-5 -10
        outer loop
          vertex 0.399344 0.461940 1.044090e-5
          vertex 5.00000e-5 5.00000e-5 1.500000e-3
          vertex 0 2.22045e-15 -10
        endloop
      endfacet
    endsolid
    """
    ).lstrip()

    tmp_file = tmpdir.join('tmp.stl')
    with tmp_file.open('wb+') as fh:
        fh.write(b(_stl_file))
        fh.seek(0)
        test_mesh = mesh.Mesh.from_file(
            str(tmp_file), fh=fh, speedups=speedups
        )
        assert test_mesh.name == b(name)


@pytest.mark.skipif(
    sys.platform.startswith('win'), reason='Only makes sense on Unix'
)
def test_locale_restore(speedups):
    if not speedups:
        pytest.skip('Only makes sense with speedups')

    old_locale = locale.nl_langinfo(locale.CODESET)

    filename = FILES_PATH / 'bwb.stl'
    mesh.Mesh.from_file(filename, speedups=speedups)

    new_locale = locale.nl_langinfo(locale.CODESET)
    assert old_locale == new_locale


@pytest.mark.skipif(
    sys.platform.startswith('win'), reason='Only makes sense on Unix'
)
def test_use_with_qt_with_custom_locale_decimal_delimeter(speedups):
    if not speedups:
        pytest.skip('Only makes sense with speedups')

    try:
        from PySide2 import QtWidgets
    except ImportError:
        try:
            from PyQt5 import QtWidgets
        except ImportError:
            warnings.warn(
                'Unable to import PySide2/PyQt5, skipping locale tests',
                ImportWarning,
                stacklevel=1,
            )
            pytest.skip('PySide2/PyQt5 missing')
    assert QtWidgets

    dir_path = os.path.dirname(os.path.realpath(__file__))
    script_path = os.path.join(dir_path, 'qt-lc_numeric-reproducer')

    env = os.environ.copy()
    env['LC_NUMERIC'] = 'cs_CZ.utf-8'

    prefix = tuple()
    if sys.platform.startswith('linux'):
        prefix = ('xvfb-run', '-a')

    p = subprocess.Popen(
        (*prefix, sys.executable, script_path),
        env=env,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = p.communicate()

    # Unable to read the file with speedups, retrying
    # https://github.com/WoLpH/numpy-stl/issues/52
    sys.stdout.write(out)
    sys.stderr.write(err)

    assert 'File too large' not in out
    assert 'File too large' not in err
    assert p.returncode == 0


def test_ascii_io():
    # Create a vanilla mesh.
    mesh_ = mesh.Mesh(np.empty(3, mesh.Mesh.dtype))
    mesh_.vectors = np.arange(27).reshape((3, 3, 3))

    # Check that unhelpful 'expected str but got bytes' error is caught and
    # replaced.
    with pytest.raises(TypeError, match='handles should be in binary mode'):
        mesh_.save('nameless', fh=io.StringIO(), mode=Mode.ASCII)

    # Write to an io.BytesIO().
    fh = io.BytesIO()
    mesh_.save('nameless', fh=fh, mode=Mode.ASCII)
    # Assert binary file is still only ascii characters.
    fh.getvalue().decode('ascii')

    import tempfile

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        # Save the mesh to the temporary file
        mesh_.save(temp_file.name, mode=Mode.ASCII)

        # Read the mesh back from the temporary file
        read = mesh.Mesh.from_file(temp_file.name)

    # Read the mesh back in.
    read = mesh.Mesh.from_file('anonymous.stl', fh=io.BytesIO(fh.getvalue()))
    # Check what comes out is the same as what went in.
    assert np.allclose(mesh_.vectors, read.vectors)


def test_ascii_write_is_lossless_for_tiny_values(speedups):
    # '{:f}' formatting used to flatten |value| < 5e-7 to '0.000000'.
    # speedups=False throughout: the pure-Python writer is under test
    # (the external speedups C writer has its own formatting).
    data = np.zeros(1, dtype=mesh.Mesh.dtype)
    data['vectors'][0] = np.array(
        [[1e-10, 5e-8, -2.5e-9], [0, 1, 0], [0, 0, 1]],
        dtype=np.float32,
    )
    original = mesh.Mesh(data, remove_empty_areas=False, speedups=False)

    fh = io.BytesIO()
    original.save('tiny.stl', fh=fh, mode=Mode.ASCII, update_normals=False)
    fh.seek(0)

    round_tripped = mesh.Mesh.from_file(
        'tiny.stl', fh=fh, mode=Mode.ASCII, speedups=False
    )
    assert (round_tripped.vectors == original.vectors).all()


class _ChunkOnlyStream(io.RawIOBase):
    """Non-seekable stream that forbids unbounded (slurp) reads."""

    def __init__(self, data: bytes) -> None:
        self._buffer = io.BytesIO(data)

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        assert size is not None and size >= 0, 'full slurp not allowed'
        return self._buffer.read(size)

    def seekable(self) -> bool:
        return False


def test_explicit_ascii_mode_streams_from_pipe(speedups):
    # With mode=ASCII the pure-Python reader consumes the stream in
    # bounded chunks; the file must not be buffered into memory whole.
    content = (
        b'solid streaming\n'
        b'facet normal 0 0 1\n'
        b'  outer loop\n'
        b'    vertex 0 0 0\n'
        b'    vertex 1 0 0\n'
        b'    vertex 0 1 0\n'
        b'  endloop\n'
        b'endfacet\n'
        b'endsolid streaming\n'
    )
    loaded = mesh.Mesh.from_file(
        'stream.stl',
        fh=_ChunkOnlyStream(content),
        mode=Mode.ASCII,
        speedups=False,
    )
    assert len(loaded.data) == 1


def test_ascii_reader_handles_many_blank_lines(speedups):
    # The blank-line skip used to recurse once per line and hit
    # RecursionError around 1000 consecutive blank lines.
    content = (
        b'solid test\n' + b'\n' * 5000 + b'facet normal 0 0 1\n'
        b'  outer loop\n'
        b'    vertex 0 0 0\n'
        b'    vertex 1 0 0\n'
        b'    vertex 0 1 0\n'
        b'  endloop\n'
        b'endfacet\n'
        b'endsolid test\n'
    )
    loaded = mesh.Mesh.from_file(
        'blank.stl',
        fh=io.BytesIO(content),
        mode=Mode.ASCII,
        speedups=False,
    )
    assert len(loaded.data) == 1
