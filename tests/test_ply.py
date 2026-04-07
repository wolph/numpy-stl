import pathlib

import numpy as np

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
    """Build the expected (12, 3, 3) vectors array from CUBE_VERTICES and CUBE_FACES."""
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
