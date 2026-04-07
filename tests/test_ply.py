import pathlib

import numpy as np
import pytest

from stl import mesh

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
