import zipfile

import pytest

from stl import mesh

_STL_FILE = b"""
solid test.stl
facet normal -0.014565 0.073223 -0.002897
  outer loop
    vertex 0.399344 0.461940 1.044090
    vertex 0.500000 0.500000 1.500000
    vertex 0.576120 0.500000 1.117320
  endloop
endfacet
endsolid test.stl
"""


def test_single_stl(tmpdir, speedups):
    tmp_file = tmpdir / 'tmp.stl'
    with tmp_file.open('wb+') as fh:
        fh.write(_STL_FILE)
        fh.seek(0)
        for m in mesh.Mesh.from_multi_file(
            str(tmp_file), fh=fh, speedups=speedups
        ):
            pass


def test_multiple_stl(tmpdir, speedups):
    tmp_file = tmpdir / 'tmp.stl'
    with tmp_file.open('wb+') as fh:
        for _ in range(10):
            fh.write(_STL_FILE)
        fh.seek(0)
        i = 0
        for i, m in enumerate(
            mesh.Mesh.from_multi_file(str(tmp_file), fh=fh, speedups=speedups)
        ):
            assert m.name == b'test.stl'

        assert i == 9


def test_single_stl_file(tmpdir, speedups):
    tmp_file = tmpdir / 'tmp.stl'
    with tmp_file.open('wb+') as fh:
        fh.write(_STL_FILE)
        fh.seek(0)
        for m in mesh.Mesh.from_multi_file(str(tmp_file), speedups=speedups):
            pass


def test_multiple_stl_file(tmpdir, speedups):
    tmp_file = tmpdir / 'tmp.stl'
    with tmp_file.open('wb+') as fh:
        for _ in range(10):
            fh.write(_STL_FILE)

        fh.seek(0)
        i = -1
        for i, m in enumerate(
            mesh.Mesh.from_multi_file(str(tmp_file), speedups=speedups)
        ):
            assert m.name == b'test.stl'

        assert i == 9


def test_multiple_stl_files(tmpdir, speedups):
    tmp_file = tmpdir / 'tmp.stl'
    with tmp_file.open('wb+') as fh:
        fh.write(_STL_FILE)
        fh.seek(0)

        filenames = [str(tmp_file)] * 10

        m = mesh.Mesh.from_files(filenames, speedups=speedups)
        assert m.data.size == 10


def test_3mf_file(three_mf_path):
    for m in mesh.Mesh.from_3mf_file(three_mf_path / 'Moon.3mf'):
        print(m)


def test_3mf_missing_file(three_mf_path):
    with pytest.raises(FileNotFoundError):
        for m in mesh.Mesh.from_3mf_file(three_mf_path / 'some_file.3mf'):
            print(m)


def test_3mf_wrong_file(ascii_file):
    with pytest.raises(zipfile.BadZipfile):
        for m in mesh.Mesh.from_3mf_file(ascii_file):
            print(m)
