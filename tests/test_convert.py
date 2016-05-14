import os
import pytest
import tempfile

from stl import stl

ascii_file = 'tests/stl_ascii/HalfDonut.stl'
binary_file = 'tests/stl_binary/HalfDonut.stl'


@pytest.fixture
def current_path():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def ascii_path(current_path):
    return os.path.join(current_path, 'stl_ascii')


@pytest.fixture
def binary_path(current_path):
    return os.path.join(current_path, 'stl_binary')


def _test_conversion(from_, to, mode):
    for name in os.listdir(from_):
        source_file = os.path.join(from_, name)
        expected_file = os.path.join(to, name)
        if not os.path.exists(expected_file):
            continue

        mesh = stl.StlMesh(source_file)
        with open(expected_file, 'rb') as expected_fh:
            expected = expected_fh.read()
            # For binary files, skip the header
            if mode is stl.BINARY:
                expected = expected[80:]

            with tempfile.TemporaryFile() as dest_fh:
                mesh.save(name, dest_fh, mode)
                # Go back to the beginning to read
                dest_fh.seek(0)
                dest = dest_fh.read()
                # For binary files, skip the header
                if mode is stl.BINARY:
                    dest = dest[80:]

                assert dest.strip() == expected.strip()


def test_ascii_to_binary(ascii_path, binary_path):
    _test_conversion(ascii_path, binary_path, mode=stl.BINARY)


def test_binary_to_ascii(ascii_path, binary_path):
    _test_conversion(binary_path, ascii_path, mode=stl.ASCII)


def test_stl_mesh(tmpdir):
    tmp_file = tmpdir.join('tmp.stl')

    mesh = stl.StlMesh(ascii_file)
    with pytest.raises(ValueError):
        mesh.save(filename=str(tmp_file), mode='test')

    mesh.save(str(tmp_file))
    mesh.save(str(tmp_file), update_normals=False)
