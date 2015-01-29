import pytest
import struct

from stl import stl

_STL_FILE = '''
solid test.stl
facet normal -0.014565 0.073223 -0.002897
  outer loop
    vertex 0.399344 0.461940 1.044090
    vertex 0.500000 0.500000 1.500000
    vertex 0.576120 0.500000 1.117320
  endloop
endfacet
endsolid test.stl
'''.lstrip()


def test_valid_ascii(tmpdir):
    tmp_file = tmpdir.join('tmp.stl')
    with tmp_file.open('w+') as fh:
        fh.write(_STL_FILE)
        fh.seek(0)
        stl.StlMesh(str(tmp_file), fh=fh)


def test_ascii_with_missing_name(tmpdir):
    tmp_file = tmpdir.join('tmp.stl')
    with tmp_file.open('w+') as fh:
        # Split the file into lines
        lines = _STL_FILE.splitlines()

        # Remove everything except solid
        lines[0] = lines[0].split()[0]

        # Join the lines to test files that start with solid without space
        fh.write('\n'.join(lines))
        fh.seek(0)
        stl.StlMesh(str(tmp_file), fh=fh)


def test_incomplete_ascii_file(tmpdir):
    tmp_file = tmpdir.join('tmp.stl')
    with tmp_file.open('w+') as fh:
        fh.write('solid some_file.stl')
        fh.seek(0)
        with pytest.raises(struct.error):
            stl.StlMesh(str(tmp_file), fh=fh)

    with tmp_file.open('w+') as fh:
        fh.write(_STL_FILE[:-20])
        fh.seek(0)
        with pytest.raises(AssertionError):
            stl.StlMesh(str(tmp_file), fh=fh)

    with tmp_file.open('w+') as fh:
        fh.write(_STL_FILE[:82])
        fh.seek(0)
        with pytest.raises(struct.error):
            stl.StlMesh(str(tmp_file), fh=fh)

    with tmp_file.open('w+') as fh:
        fh.write(_STL_FILE[:100])
        fh.seek(0)
        with pytest.raises(AssertionError):
            stl.StlMesh(str(tmp_file), fh=fh)


def test_corrupt_ascii_file(tmpdir):
    tmp_file = tmpdir.join('tmp.stl')
    with tmp_file.open('w+') as fh:
        fh.write(_STL_FILE)
        fh.seek(40)
        print >>fh, '####\n' * 100
        fh.seek(0)
        with pytest.raises(AssertionError):
            stl.StlMesh(str(tmp_file), fh=fh)

    with tmp_file.open('w+') as fh:
        fh.write(_STL_FILE)
        fh.seek(40)
        print >>fh, ' ' * 100
        fh.seek(80)
        fh.write(struct.pack('@i', 10))
        fh.seek(0)
        with pytest.raises(AssertionError):
            stl.StlMesh(str(tmp_file), fh=fh)


def test_corrupt_binary_file(tmpdir):
    tmp_file = tmpdir.join('tmp.stl')
    with tmp_file.open('w+') as fh:
        fh.write('#########\n' * 8)
        fh.write('#\0\0\0')
        fh.seek(0)
        stl.StlMesh(str(tmp_file), fh=fh)

    with tmp_file.open('w+') as fh:
        fh.write('#########\n' * 9)
        fh.seek(0)
        with pytest.raises(AssertionError):
            stl.StlMesh(str(tmp_file), fh=fh)

    with tmp_file.open('w+') as fh:
        fh.write('#########\n' * 8)
        fh.write('#\0\0\0')
        fh.seek(0)
        fh.write('solid test.stl')
        fh.seek(0)
        stl.StlMesh(str(tmp_file), fh=fh)


