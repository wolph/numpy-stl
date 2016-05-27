from stl import mesh

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


def test_single_stl(tmpdir):
    tmp_file = tmpdir.join('tmp.stl')
    with tmp_file.open('w+') as fh:
        fh.write(_STL_FILE)
        fh.seek(0)
        for m in mesh.Mesh.from_multi_file(str(tmp_file), fh=fh):
            pass


def test_multiple_stl(tmpdir):
    tmp_file = tmpdir.join('tmp.stl')
    with tmp_file.open('w+') as fh:
        for _ in range(10):
            fh.write(_STL_FILE)
        fh.seek(0)
        for i, m in enumerate(mesh.Mesh.from_multi_file(str(tmp_file), fh=fh)):
            assert m.name == b'test.stl'

        assert i == 9


def test_single_stl_file(tmpdir):
    tmp_file = tmpdir.join('tmp.stl')
    with tmp_file.open('w+') as fh:
        fh.write(_STL_FILE)
        fh.seek(0)
        for m in mesh.Mesh.from_multi_file(str(tmp_file)):
            pass


def test_multiple_stl_file(tmpdir):
    tmp_file = tmpdir.join('tmp.stl')
    with tmp_file.open('w+') as fh:
        for _ in range(10):
            fh.write(_STL_FILE)

        fh.seek(0)
        for i, m in enumerate(mesh.Mesh.from_multi_file(str(tmp_file))):
            assert m.name == b'test.stl'

        assert i == 9


