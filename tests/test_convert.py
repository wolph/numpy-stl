# import os
import pytest
import tempfile

from stl import stl


def _test_conversion(from_, to, mode, speedups):

    for name in from_.listdir():
        source_file = from_.join(name)
        expected_file = to.join(name)
        if not expected_file.exists():
            continue

        mesh = stl.StlMesh(source_file, speedups=speedups)
        with open(str(expected_file), 'rb') as expected_fh:
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


def test_ascii_to_binary(ascii_path, binary_path, speedups):
    _test_conversion(ascii_path, binary_path, mode=stl.BINARY,
                     speedups=speedups)


def test_binary_to_ascii(ascii_path, binary_path, speedups):
    _test_conversion(binary_path, ascii_path, mode=stl.ASCII,
                     speedups=speedups)


def test_stl_mesh(ascii_file, tmpdir, speedups):
    tmp_file = tmpdir.join('tmp.stl')

    mesh = stl.StlMesh(ascii_file, speedups=speedups)
    with pytest.raises(ValueError):
        mesh.save(filename=str(tmp_file), mode='test')

    mesh.save(str(tmp_file))
    mesh.save(str(tmp_file), update_normals=False)
