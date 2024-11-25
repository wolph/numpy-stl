import os
import sys
import locale
import pytest
import pathlib
import warnings
import subprocess
import io
import numpy

from stl.utils import b
from stl import mesh, Mode


FILES_PATH = pathlib.Path(__file__).parent / 'stl_tests'


def test_ascii_file(speedups):
    filename = FILES_PATH / 'bwb.stl'
    mesh.Mesh.from_file(filename, speedups=speedups)


def test_chinese_name(tmpdir, speedups):
    name = 'Test Chinese name 月球'
    _stl_file = (
        """
    solid %s
      facet normal -0.014565 0.073223 -0.002897
        outer loop
          vertex 0.399344 0.461940 1.044090
          vertex 0.500000 0.500000 1.500000
          vertex 0.576120 0.500000 1.117320
        endloop
      endfacet
    endsolid
    """
        % name
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
        """
    solid %s
      facet normal -0.014565 0.073223 -0.002897
        outer loop
          vertex 0.399344 0.461940 1.044090
          vertex 0.500000 0.500000 1.500000
          vertex 0.576120 0.500000 1.117320
        endloop
      endfacet
    endsolid
    """
        % name
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
        """
    solid %s
      facet normal 1.014565e-10 7.3223e-5 -10
        outer loop
          vertex 0.399344 0.461940 1.044090e-5
          vertex 5.00000e-5 5.00000e-5 1.500000e-3
          vertex 0 2.22045e-15 -10
        endloop
      endfacet
    endsolid
    """
        % name
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

    venv = os.environ.get('VIRTUAL_ENV', '')
    if (3, 6) == sys.version_info[:2] and venv.startswith('/home/travis/'):
        pytest.skip('PySide2/PyQt5 tests are broken on Travis Python 3.6')

    try:
        from PySide2 import QtWidgets
    except ImportError:
        try:
            from PyQt5 import QtWidgets
        except ImportError:
            warnings.warn(
                'Unable to import PySide2/PyQt5, skipping locale tests',
                ImportWarning,
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
        prefix + (sys.executable, script_path),
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
    mesh_ = mesh.Mesh(numpy.empty(3, mesh.Mesh.dtype))
    mesh_.vectors = numpy.arange(27).reshape((3, 3, 3))

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
    assert numpy.allclose(mesh_.vectors, read.vectors)
