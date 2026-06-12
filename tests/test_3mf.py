import pathlib
import zipfile

import numpy as np
import pytest

from stl import mesh

_RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships
 xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel0"
  Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>"""

_MODEL_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<model xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
 <resources>
  <object id="1" type="model">
   <mesh>
    <vertices>{vertices}</vertices>
    <triangles>{triangles}</triangles>
   </mesh>
  </object>
 </resources>
 <build><item objectid="1"/></build>
</model>"""

_GOOD_VERTICES = (
    '<vertex x="0" y="0" z="0"/>'
    '<vertex x="1" y="0" z="0"/>'
    '<vertex x="0" y="1" z="0"/>'
)
_GOOD_TRIANGLES = '<triangle v1="0" v2="1" v3="2"/>'


def _make_3mf(
    tmp_path: pathlib.Path,
    vertices: str = _GOOD_VERTICES,
    triangles: str = _GOOD_TRIANGLES,
) -> str:
    path = tmp_path / 'test.3mf'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('_rels/.rels', _RELS)
        archive.writestr(
            '3D/3dmodel.model',
            _MODEL_TEMPLATE.format(vertices=vertices, triangles=triangles),
        )
    return str(path)


def test_3mf_valid_file(tmp_path):
    meshes = list(mesh.Mesh.from_3mf_file(_make_3mf(tmp_path)))

    assert len(meshes) == 1
    assert np.allclose(
        meshes[0].vectors[0],
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
    )


def test_3mf_vertex_missing_coordinate(tmp_path):
    # Used to crash with a raw KeyError: 'z'.
    filename = _make_3mf(
        tmp_path,
        vertices=(
            '<vertex x="0" y="0"/>'
            '<vertex x="1" y="0" z="0"/>'
            '<vertex x="0" y="1" z="0"/>'
        ),
    )
    with pytest.raises(ValueError, match='z'):
        list(mesh.Mesh.from_3mf_file(filename))


def test_3mf_triangle_missing_index(tmp_path):
    # Used to crash with a raw KeyError: 'v3'.
    filename = _make_3mf(tmp_path, triangles='<triangle v1="0" v2="1"/>')
    with pytest.raises(ValueError, match='v3'):
        list(mesh.Mesh.from_3mf_file(filename))


@pytest.mark.parametrize('bad_index', ['99', '-1'])
def test_3mf_triangle_index_out_of_range(tmp_path, bad_index):
    # Out-of-range used to raise a raw IndexError; negative indices
    # silently wrapped around.
    filename = _make_3mf(
        tmp_path,
        triangles=f'<triangle v1="0" v2="1" v3="{bad_index}"/>',
    )
    with pytest.raises(ValueError, match='index'):
        list(mesh.Mesh.from_3mf_file(filename))
