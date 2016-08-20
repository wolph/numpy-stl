from stl.utils import b
from stl import mesh


def test_long_name(tmpdir, speedups):
    name = 'just some very long name which will not fit within the standard'
    name += name
    _stl_file = ('''
    solid %s
      facet normal -0.014565 0.073223 -0.002897
        outer loop
          vertex 0.399344 0.461940 1.044090
          vertex 0.500000 0.500000 1.500000
          vertex 0.576120 0.500000 1.117320
        endloop
      endfacet
    endsolid
    ''' % name).lstrip()

    tmp_file = tmpdir.join('tmp.stl')
    with tmp_file.open('wb+') as fh:
        fh.write(b(_stl_file))
        fh.seek(0)
        test_mesh = mesh.Mesh.from_file(str(tmp_file), fh=fh,
                                        speedups=speedups)
        assert test_mesh.name == b(name)


