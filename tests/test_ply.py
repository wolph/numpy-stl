import io
import pathlib
import struct

import numpy as np
import pytest

from stl import (
    mesh,
    ply,
    stl as stl_module,
)

PLY_ASCII_PATH = pathlib.Path(__file__).parent / 'ply_ascii'
PLY_BINARY_PATH = pathlib.Path(__file__).parent / 'ply_binary'

CUBE_VERTICES = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=np.float32,
)

CUBE_FACES = [
    (0, 3, 1),
    (1, 3, 2),
    (0, 4, 7),
    (0, 7, 3),
    (4, 5, 6),
    (4, 6, 7),
    (5, 1, 2),
    (5, 2, 6),
    (2, 3, 6),
    (3, 7, 6),
    (0, 1, 5),
    (0, 5, 4),
]


def _expected_vectors() -> np.ndarray:
    """Build the expected (12, 3, 3) vectors array."""
    vectors = np.zeros((len(CUBE_FACES), 3, 3), dtype=np.float32)
    for i, (a, b, c) in enumerate(CUBE_FACES):
        vectors[i][0] = CUBE_VERTICES[a]
        vectors[i][1] = CUBE_VERTICES[b]
        vectors[i][2] = CUBE_VERTICES[c]
    return vectors


def _make_vertex_element(
    *property_names: str,
    count: int = 1,
) -> ply._Element:
    element = ply._Element('vertex', count)
    for name in property_names:
        element.properties.append(ply._Property(name, 'float'))
    return element


def _make_face_element(
    *,
    count: int = 1,
    include_list: bool = True,
    scalar_prefix: bool = False,
) -> ply._Element:
    element = ply._Element('face', count)
    if scalar_prefix:
        element.properties.append(ply._Property('material_index', 'uchar'))
    if include_list:
        element.properties.append(
            ply._Property(
                'vertex_indices',
                'list',
                is_list=True,
                count_type='uchar',
                item_type='int',
            )
        )
    return element


def _triangle_mesh_data() -> np.ndarray:
    data = np.zeros(1, dtype=mesh.Mesh.dtype)
    data['vectors'][0] = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return data


class TestReadAsciiPly:
    def test_read_ascii_ply_face_count(self):
        m = mesh.Mesh.from_ply_file(str(PLY_ASCII_PATH / 'Cube.ply'))
        assert len(m.data) == 12

    def test_read_ascii_ply_vectors(self):
        m = mesh.Mesh.from_ply_file(str(PLY_ASCII_PATH / 'Cube.ply'))
        expected = _expected_vectors()
        np.testing.assert_array_almost_equal(m.vectors, expected)

    def test_read_ascii_ply_with_filehandle(self):
        path = PLY_ASCII_PATH / 'Cube.ply'
        with open(path, 'rb') as fh:
            m = mesh.Mesh.from_ply_file(str(path), fh=fh)
        assert len(m.data) == 12


class TestReadBinaryPly:
    def test_read_binary_le_face_count(self):
        m = mesh.Mesh.from_ply_file(str(PLY_BINARY_PATH / 'Cube.ply'))
        assert len(m.data) == 12

    def test_read_binary_le_vectors(self):
        m = mesh.Mesh.from_ply_file(str(PLY_BINARY_PATH / 'Cube.ply'))
        expected = _expected_vectors()
        np.testing.assert_array_almost_equal(m.vectors, expected)

    def test_read_binary_be_face_count(self):
        m = mesh.Mesh.from_ply_file(str(PLY_BINARY_PATH / 'CubeBigEndian.ply'))
        assert len(m.data) == 12

    def test_read_binary_be_vectors(self):
        m = mesh.Mesh.from_ply_file(str(PLY_BINARY_PATH / 'CubeBigEndian.ply'))
        expected = _expected_vectors()
        np.testing.assert_array_almost_equal(m.vectors, expected)


class TestTriangulation:
    def test_quad_faces_produce_correct_triangle_count(self):
        m = mesh.Mesh.from_ply_file(str(PLY_ASCII_PATH / 'Quad.ply'))
        # 6 quads -> 6 * 2 = 12 triangles
        assert len(m.data) == 12

    def test_quad_faces_cover_all_vertices(self):
        m = mesh.Mesh.from_ply_file(str(PLY_ASCII_PATH / 'Quad.ply'))
        # All 8 cube vertices should appear
        unique_verts = np.unique(m.vectors.reshape(-1, 3), axis=0)
        assert len(unique_verts) == 8


class TestWritePly:
    def test_write_binary_round_trip(self, tmp_path):
        original = mesh.Mesh.from_ply_file(str(PLY_ASCII_PATH / 'Cube.ply'))
        out_path = tmp_path / 'cube_out.ply'
        original.save_ply(str(out_path))

        reloaded = mesh.Mesh.from_ply_file(str(out_path))
        np.testing.assert_array_almost_equal(
            original.vectors, reloaded.vectors
        )

    def test_write_ascii_round_trip(self, tmp_path):
        original = mesh.Mesh.from_ply_file(str(PLY_ASCII_PATH / 'Cube.ply'))
        out_path = tmp_path / 'cube_out.ply'
        original.save_ply(str(out_path), mode='ascii')

        reloaded = mesh.Mesh.from_ply_file(str(out_path))
        np.testing.assert_array_almost_equal(
            original.vectors, reloaded.vectors
        )

    def test_write_big_endian_round_trip(self, tmp_path):
        original = mesh.Mesh.from_ply_file(str(PLY_ASCII_PATH / 'Cube.ply'))
        out_path = tmp_path / 'cube_be.ply'
        original.save_ply(str(out_path), mode='binary_big_endian')

        reloaded = mesh.Mesh.from_ply_file(str(out_path))
        np.testing.assert_array_almost_equal(
            original.vectors, reloaded.vectors
        )

    def test_write_with_filehandle(self, tmp_path):
        original = mesh.Mesh.from_ply_file(str(PLY_ASCII_PATH / 'Cube.ply'))
        out_path = tmp_path / 'cube_fh.ply'
        with open(out_path, 'wb') as fh:
            original.save_ply(str(out_path), fh=fh)

        reloaded = mesh.Mesh.from_ply_file(str(out_path))
        assert len(reloaded.data) == 12

    def test_stl_to_ply_round_trip(self, tmp_path):
        """Load an STL, save as PLY, reload, compare."""
        stl_path = (
            pathlib.Path(__file__).parent / 'stl_binary' / 'HalfDonut.stl'
        )
        original = mesh.Mesh.from_file(str(stl_path))
        ply_path = tmp_path / 'donut.ply'
        original.save_ply(str(ply_path))

        reloaded = mesh.Mesh.from_ply_file(str(ply_path))
        np.testing.assert_array_almost_equal(
            original.vectors, reloaded.vectors
        )


class TestPlyErrors:
    def test_invalid_magic(self, tmp_path):
        bad = tmp_path / 'bad.ply'
        bad.write_bytes(b'not a ply file\n')
        with pytest.raises(ValueError, match='Not a PLY'):
            mesh.Mesh.from_ply_file(str(bad))

    def test_missing_vertex_element(self, tmp_path):
        bad = tmp_path / 'bad.ply'
        bad.write_bytes(
            b'ply\n'
            b'format ascii 1.0\n'
            b'element face 1\n'
            b'property list uchar int vertex_indices\n'
            b'end_header\n'
            b'3 0 1 2\n'
        )
        with pytest.raises(ValueError, match='no vertex element'):
            mesh.Mesh.from_ply_file(str(bad))

    def test_missing_face_element(self, tmp_path):
        bad = tmp_path / 'bad.ply'
        bad.write_bytes(
            b'ply\n'
            b'format ascii 1.0\n'
            b'element vertex 3\n'
            b'property float x\n'
            b'property float y\n'
            b'property float z\n'
            b'end_header\n'
            b'0 0 0\n'
            b'1 0 0\n'
            b'0 1 0\n'
        )
        with pytest.raises(ValueError, match='no face element'):
            mesh.Mesh.from_ply_file(str(bad))

    def test_truncated_ascii_file(self, tmp_path):
        bad = tmp_path / 'bad.ply'
        bad.write_bytes(
            b'ply\n'
            b'format ascii 1.0\n'
            b'element vertex 3\n'
            b'property float x\n'
            b'property float y\n'
            b'property float z\n'
            b'element face 1\n'
            b'property list uchar int vertex_indices\n'
            b'end_header\n'
            b'0 0 0\n'
            # Missing 2 vertices and face data
        )
        with pytest.raises(ValueError):
            mesh.Mesh.from_ply_file(str(bad))


class TestPlyHelpers:
    def test_load_ascii_uses_speedup_reader_when_available(
        self, monkeypatch, tmp_path
    ):
        expected = np.zeros(0, dtype=mesh.Mesh.dtype)
        captured: dict[str, object] = {}

        def fake_ascii_read(fh, header):
            captured['header'] = header
            captured['fileno'] = fh.fileno()
            return b'fast-path', expected

        monkeypatch.setattr(stl_module, '_ascii_read', fake_ascii_read)
        path = tmp_path / 'speedup-input.stl'
        path.write_bytes(b'solid speedup\nendsolid speedup\n')

        with open(path, 'rb') as fh:
            name, data = mesh.Mesh._load_ascii(fh, b'solid speedup')

        assert name == b'fast-path'
        assert data is expected
        assert captured['header'] == b'solid speedup'

    def test_write_ascii_uses_speedup_writer_when_available(
        self, monkeypatch, tmp_path
    ):
        captured: dict[str, object] = {}

        def fake_ascii_write(fh, name, data):
            captured['name'] = name
            captured['rows'] = len(data)
            fh.write(b'fast-ascii')

        monkeypatch.setattr(stl_module, '_ascii_write', fake_ascii_write)
        triangle = mesh.Mesh(_triangle_mesh_data(), remove_empty_areas=False)
        path = tmp_path / 'speedup-output.stl'

        with open(path, 'wb') as fh:
            triangle._write_ascii(fh, 'triangle')

        assert path.read_bytes() == b'fast-ascii'
        assert captured == {'name': b'triangle', 'rows': 1}

    def test_parse_header_ignores_blank_lines(self):
        fh = io.BytesIO(
            b'ply\nformat ascii 1.0\n\nobj_info sample-object\nend_header\n'
        )

        format_str, elements, obj_name = ply._parse_header(fh)

        assert format_str == 'ascii'
        assert elements == []
        assert obj_name == 'sample-object'

    def test_parse_header_requires_end_header(self):
        fh = io.BytesIO(b'ply\nformat ascii 1.0\n')

        with pytest.raises(
            ValueError, match='Unexpected end of file while parsing header'
        ):
            ply._parse_header(fh)

    def test_parse_header_requires_format_line(self):
        fh = io.BytesIO(b'ply\nelement vertex 0\nend_header\n')

        with pytest.raises(ValueError, match='No format line found'):
            ply._parse_header(fh)

    def test_find_elements_handles_extra_elements(self):
        vertex = _make_vertex_element('x', 'y', 'z')
        face = _make_face_element()
        extra = ply._Element('commentary', 1)

        found_vertex, found_face = ply._find_elements([vertex, face, extra])

        assert found_vertex is vertex
        assert found_face is face

    def test_find_xyz_indices_accepts_trailing_properties(self):
        vertex = _make_vertex_element('x', 'y', 'z', 'temperature')

        assert ply._find_xyz_indices(vertex) == (0, 1, 2)

    def test_find_xyz_indices_requires_all_axes(self):
        vertex = _make_vertex_element('x', 'y')

        with pytest.raises(ValueError, match='missing x, y, or z'):
            ply._find_xyz_indices(vertex)

    def test_read_ascii_skips_unknown_elements(self):
        vertex = _make_vertex_element('x', 'y', 'z', count=3)
        edge = ply._Element('edge', 1)
        edge.properties.append(ply._Property('id', 'uchar'))
        face = _make_face_element()

        vertices, faces = ply._read_ascii(
            io.BytesIO(b'0 0 0\n1 0 0\n0 1 0\n7\n3 0 1 2\n'),
            [vertex, edge, face],
        )

        np.testing.assert_array_equal(
            vertices,
            np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
        )
        assert faces == [[0, 1, 2]]

    def test_read_binary_faces_requires_list_property(self):
        face = _make_face_element(include_list=False)

        with pytest.raises(ValueError, match='no list property'):
            ply._read_binary_faces(io.BytesIO(), face, '<')

    def test_read_binary_faces_requires_complete_face_count(self):
        face = _make_face_element()

        with pytest.raises(ValueError, match='Unexpected EOF reading face'):
            ply._read_binary_faces(io.BytesIO(), face, '<')

    def test_read_binary_faces_requires_complete_face_indices(self):
        face = _make_face_element()
        fh = io.BytesIO(struct.pack('<B', 3) + struct.pack('<ii', 0, 1))

        with pytest.raises(
            ValueError, match='Unexpected EOF reading face indices'
        ):
            ply._read_binary_faces(fh, face, '<')

    def test_read_binary_faces_consumes_scalar_prefix(self):
        # The stream contains the material_index byte (9) followed by
        # the list count (3); the scalar must be consumed, not misread
        # as the count.
        face = _make_face_element(scalar_prefix=True)

        faces = ply._read_binary_faces(
            io.BytesIO(struct.pack('<BBiii', 9, 3, 0, 1, 2)),
            face,
            '<',
        )

        assert faces == [[0, 1, 2]]

    def test_skip_binary_elements_exits_without_face_marker(self):
        vertex = _make_vertex_element('x', 'y', 'z')
        face = _make_face_element()

        ply._skip_binary_elements(io.BytesIO(), [vertex], vertex, face)

    def test_skip_binary_elements_skips_scalar_elements_between_mesh_data(
        self,
    ):
        vertex = _make_vertex_element('x', 'y', 'z')
        edge = ply._Element('edge', 1)
        edge.properties.append(ply._Property('id', 'uchar'))
        face = _make_face_element()

        vertices, faces = ply._read_binary(
            io.BytesIO(
                struct.pack('<fff', 0.0, 1.0, 2.0)
                + struct.pack('<B', 7)
                + struct.pack('<Biii', 3, 0, 0, 0)
            ),
            [vertex, edge, face],
            'binary_little_endian',
        )

        np.testing.assert_array_equal(
            vertices, np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
        )
        assert faces == [[0, 0, 0]]

    def test_skip_binary_elements_ignores_elements_before_vertex(self):
        lead = ply._Element('comment', 1)
        lead.properties.append(ply._Property('id', 'uchar'))
        vertex = _make_vertex_element('x', 'y', 'z')
        face = _make_face_element()

        ply._skip_binary_elements(
            io.BytesIO(struct.pack('<B', 7)),
            [lead, vertex, face],
            vertex,
            face,
        )

    def test_skip_binary_elements_rejects_list_properties(self):
        vertex = _make_vertex_element('x', 'y', 'z')
        edge = ply._Element('edge', 1)
        edge.properties.append(
            ply._Property(
                'vertex_indices',
                'list',
                is_list=True,
                count_type='uchar',
                item_type='int',
            )
        )
        face = _make_face_element()

        with pytest.raises(ValueError, match='list properties'):
            ply._skip_binary_elements(
                io.BytesIO(), [vertex, edge, face], vertex, face
            )

    def test_read_binary_requires_full_vertex_buffer(self):
        vertex = _make_vertex_element('x', 'y', 'z', count=2)
        face = _make_face_element(count=0)

        with pytest.raises(ValueError, match='Unexpected EOF reading vertex'):
            ply._read_binary(
                io.BytesIO(struct.pack('<fff', 0.0, 0.0, 0.0)),
                [vertex, face],
                'binary_little_endian',
            )

    def test_triangulate_skips_short_faces(self):
        triangles = ply._triangulate([[0, 1], [0, 1, 2, 3]])

        assert triangles == [(0, 1, 2), (0, 2, 3)]

    def test_write_ply_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match='Unknown PLY mode'):
            ply.write_ply(io.BytesIO(), _triangle_mesh_data(), mode='wat')

    def test_write_ply_includes_object_name(self):
        fh = io.BytesIO()

        ply.write_ply(
            fh,
            _triangle_mesh_data(),
            name='triangle',
            mode='ascii',
        )

        assert b'obj_info triangle\n' in fh.getvalue()

    def test_write_ply_omits_object_name_when_none_is_given(self):
        fh = io.BytesIO()

        ply.write_ply(fh, _triangle_mesh_data(), mode='ascii')

        assert b'obj_info' not in fh.getvalue()

    def test_read_ply_rejects_unknown_format(self):
        fh = io.BytesIO(b'ply\nformat binary_middle_endian 1.0\nend_header\n')

        with pytest.raises(ValueError, match='Unknown PLY format'):
            ply.read_ply(fh, mesh.Mesh.dtype)

    def test_save_ply_can_skip_normal_updates_for_byte_names(
        self, monkeypatch
    ):
        captured: dict[str, object] = {}
        triangle = mesh.Mesh(_triangle_mesh_data(), remove_empty_areas=False)
        triangle.name = b'byte-name'

        def fake_write_ply(fh, data, name='', mode='binary_little_endian'):
            captured['name'] = name
            captured['rows'] = len(data)
            captured['mode'] = mode
            fh.write(b'ply-data')

        monkeypatch.setattr(ply, 'write_ply', fake_write_ply)
        fh = io.BytesIO()

        triangle.save_ply(
            'ignored.ply',
            fh=fh,
            update_normals=False,
        )

        assert captured == {
            'name': 'byte-name',
            'rows': 1,
            'mode': 'binary_little_endian',
        }
        assert fh.getvalue() == b'ply-data'

    def test_save_ply_handles_non_string_names(self, monkeypatch):
        captured: dict[str, object] = {}
        triangle = mesh.Mesh(_triangle_mesh_data(), remove_empty_areas=False)
        triangle.name = None

        def fake_write_ply(fh, data, name='', mode='binary_little_endian'):
            captured['name'] = name
            fh.write(b'ply-data')

        monkeypatch.setattr(ply, 'write_ply', fake_write_ply)

        triangle.save_ply(
            'ignored.ply',
            fh=io.BytesIO(),
            update_normals=False,
        )

        assert captured['name'] == ''


def _ply_header(*lines: str) -> bytes:
    return ('\n'.join(('ply', *lines, 'end_header')) + '\n').encode('ascii')


_TRIANGLE_VERTEX_LINES = (
    'element vertex 3',
    'property float x',
    'property float y',
    'property float z',
)
_TRIANGLE_VECTORS = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
    dtype=np.float32,
)


def test_read_binary_skips_elements_declared_before_vertex():
    # Spec-legal PLY files may declare elements (camera, material, ...)
    # before the vertex element; their data used to be consumed as
    # vertex data, silently corrupting the mesh.
    header = _ply_header(
        'format binary_little_endian 1.0',
        'element camera 1',
        'property float focal',
        'property float aperture',
        *_TRIANGLE_VERTEX_LINES,
        'element face 1',
        'property list uchar int vertex_indices',
    )
    body = (
        struct.pack('<2f', 35.0, 1.8)
        + struct.pack('<9f', 0, 0, 0, 1, 0, 0, 0, 1, 0)
        + struct.pack('<B3i', 3, 0, 1, 2)
    )
    data, _name = ply.read_ply(io.BytesIO(header + body), mesh.Mesh.dtype)

    assert len(data) == 1
    assert np.allclose(data['vectors'][0], _TRIANGLE_VECTORS)


def test_vertex_element_with_list_property_raises_value_error():
    # Used to crash with a raw KeyError: 'list'.
    header = _ply_header(
        'format binary_little_endian 1.0',
        *_TRIANGLE_VERTEX_LINES,
        'property list uchar uchar rgba',
        'element face 1',
        'property list uchar int vertex_indices',
    )
    with pytest.raises(ValueError, match='list'):
        ply.read_ply(io.BytesIO(header), mesh.Mesh.dtype)


def test_binary_face_scalar_properties_around_list_are_consumed():
    # Scalar face properties before/after the list property used to be
    # misread as the list count / next face's data.
    header = _ply_header(
        'format binary_little_endian 1.0',
        *_TRIANGLE_VERTEX_LINES,
        'element face 1',
        'property uchar material_id',
        'property list uchar int vertex_indices',
        'property float quality',
    )
    body = struct.pack('<9f', 0, 0, 0, 1, 0, 0, 0, 1, 0) + struct.pack(
        '<BB3if', 7, 3, 0, 1, 2, 0.5
    )
    data, _name = ply.read_ply(io.BytesIO(header + body), mesh.Mesh.dtype)

    assert len(data) == 1
    assert np.allclose(data['vectors'][0], _TRIANGLE_VECTORS)


def test_ascii_face_scalar_property_before_list_is_consumed():
    header = _ply_header(
        'format ascii 1.0',
        *_TRIANGLE_VERTEX_LINES,
        'element face 1',
        'property int material_id',
        'property list uchar int vertex_indices',
    )
    body = b'0 0 0\n1 0 0\n0 1 0\n7 3 0 1 2\n'
    data, _name = ply.read_ply(io.BytesIO(header + body), mesh.Mesh.dtype)

    assert len(data) == 1
    assert np.allclose(data['vectors'][0], _TRIANGLE_VECTORS)


def _ascii_triangle_with_face(face_line: bytes) -> io.BytesIO:
    header = _ply_header(
        'format ascii 1.0',
        *_TRIANGLE_VERTEX_LINES,
        'element face 1',
        'property list uchar int vertex_indices',
    )
    return io.BytesIO(header + b'0 0 0\n1 0 0\n0 1 0\n' + face_line)


def test_binary_second_list_property_on_face_is_skipped():
    # A second list property (e.g. texture coordinates) after the
    # vertex index list must be consumed per its own count field.
    header = _ply_header(
        'format binary_little_endian 1.0',
        *_TRIANGLE_VERTEX_LINES,
        'element face 2',
        'property list uchar int vertex_indices',
        'property list uchar float texcoords',
    )
    face = struct.pack('<B3i', 3, 0, 1, 2) + struct.pack(
        '<B6f', 6, 0, 0, 1, 0, 0, 1
    )
    body = struct.pack('<9f', 0, 0, 0, 1, 0, 0, 0, 1, 0) + face * 2
    data, _name = ply.read_ply(io.BytesIO(header + body), mesh.Mesh.dtype)

    assert len(data) == 2
    assert np.allclose(data['vectors'][0], _TRIANGLE_VECTORS)


def test_ascii_face_element_without_list_property_raises():
    header = _ply_header(
        'format ascii 1.0',
        *_TRIANGLE_VERTEX_LINES,
        'element face 1',
        'property int material_id',
    )
    body = b'0 0 0\n1 0 0\n0 1 0\n7\n'
    with pytest.raises(ValueError, match='no list property'):
        ply.read_ply(io.BytesIO(header + body), mesh.Mesh.dtype)


def test_binary_element_before_vertex_with_list_property_raises():
    header = _ply_header(
        'format binary_little_endian 1.0',
        'element strips 1',
        'property list uchar int strip_indices',
        *_TRIANGLE_VERTEX_LINES,
        'element face 1',
        'property list uchar int vertex_indices',
    )
    with pytest.raises(ValueError, match='list propert'):
        ply.read_ply(io.BytesIO(header), mesh.Mesh.dtype)


def test_face_index_out_of_range_raises_value_error():
    with pytest.raises(ValueError, match='index'):
        ply.read_ply(_ascii_triangle_with_face(b'3 0 1 99\n'), mesh.Mesh.dtype)


def test_negative_face_index_raises_value_error():
    # Negative indices used to wrap around silently via numpy indexing.
    with pytest.raises(ValueError, match='index'):
        ply.read_ply(_ascii_triangle_with_face(b'3 0 1 -1\n'), mesh.Mesh.dtype)
