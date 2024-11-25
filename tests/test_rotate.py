# type: ignore[reportAttributeAccessIssue]
import math

import numpy as np
import pytest

from stl.mesh import Mesh

from . import utils


def test_rotation():
    # Create 6 faces of a cube
    data = np.zeros(6, dtype=Mesh.dtype)

    # Top of the cube
    data['vectors'][0] = np.array([[0, 1, 1], [1, 0, 1], [0, 0, 1]])
    data['vectors'][1] = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1]])
    # Right face
    data['vectors'][2] = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0]])
    data['vectors'][3] = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0]])
    # Left face
    data['vectors'][4] = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1]])
    data['vectors'][5] = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1]])

    mesh = Mesh(data, remove_empty_areas=False)

    # Since the cube faces are from 0 to 1 we can move it to the middle by
    # substracting .5
    data['vectors'] -= 0.5

    # Rotate 90 degrees over the X axis followed by the Y axis followed by the
    # X axis
    mesh.rotate([0.5, 0.0, 0.0], math.radians(90))
    mesh.rotate([0.0, 0.5, 0.0], math.radians(90))
    mesh.rotate([0.5, 0.0, 0.0], math.radians(90))

    # Since the cube faces are from 0 to 1 we can move it to the middle by
    # substracting .5
    data['vectors'] += 0.5

    # We use a slightly higher absolute tolerance here, for ppc64le
    # https://github.com/WoLpH/numpy-stl/issues/78
    assert np.allclose(
        mesh.vectors,
        np.array(
            [
                [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [1, 0, 0], [1, 1, 0]],
                [[0, 1, 1], [0, 1, 0], [1, 1, 1]],
                [[1, 1, 0], [0, 1, 0], [1, 1, 1]],
                [[0, 0, 1], [0, 1, 1], [0, 1, 0]],
                [[0, 0, 1], [0, 0, 0], [0, 1, 0]],
            ]
        ),
        atol=1e-07,
    )


def test_rotation_over_point():
    # Create a single face
    data = np.zeros(1, dtype=Mesh.dtype)

    data['vectors'][0] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    mesh = Mesh(data, remove_empty_areas=False)

    mesh.rotate([1, 0, 0], math.radians(180), point=[1, 2, 3])
    utils.array_equals(
        mesh.vectors,
        np.array([[[1.0, 4.0, 6.0], [0.0, 3.0, 6.0], [0.0, 4.0, 5.0]]]),
    )

    mesh.rotate([1, 0, 0], math.radians(-180), point=[1, 2, 3])
    utils.array_equals(
        mesh.vectors, np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    )

    mesh.rotate([1, 0, 0], math.radians(180), point=0.0)
    utils.array_equals(
        mesh.vectors,
        np.array([[[1.0, 0.0, -0.0], [0.0, -1.0, -0.0], [0.0, 0.0, -1.0]]]),
    )

    with pytest.raises(TypeError):
        mesh.rotate([1, 0, 0], math.radians(180), point='x')


def test_double_rotation():
    # Create a single face
    data = np.zeros(1, dtype=Mesh.dtype)

    data['vectors'][0] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    mesh = Mesh(data, remove_empty_areas=False)

    rotation_matrix = mesh.rotation_matrix([1, 0, 0], math.radians(180))
    combined_rotation_matrix = np.dot(rotation_matrix, rotation_matrix)

    mesh.rotate_using_matrix(combined_rotation_matrix)
    utils.array_equals(
        mesh.vectors,
        np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]),
    )


def test_no_rotation():
    # Create a single face
    data = np.zeros(1, dtype=Mesh.dtype)

    data['vectors'][0] = np.array([[0, 1, 1], [1, 0, 1], [0, 0, 1]])

    mesh = Mesh(data, remove_empty_areas=False)

    # Rotate by 0 degrees
    mesh.rotate([0.5, 0.0, 0.0], math.radians(0))
    assert np.allclose(
        mesh.vectors, np.array([[[0, 1, 1], [1, 0, 1], [0, 0, 1]]])
    )

    # Use a zero rotation matrix
    mesh.rotate([0.0, 0.0, 0.0], math.radians(90))
    assert np.allclose(
        mesh.vectors, np.array([[[0, 1, 1], [1, 0, 1], [0, 0, 1]]])
    )


def test_no_translation():
    # Create a single face
    data = np.zeros(1, dtype=Mesh.dtype)
    data['vectors'][0] = np.array([[0, 1, 1], [1, 0, 1], [0, 0, 1]])

    mesh = Mesh(data, remove_empty_areas=False)
    assert np.allclose(
        mesh.vectors, np.array([[[0, 1, 1], [1, 0, 1], [0, 0, 1]]])
    )

    # Translate mesh with a zero vector
    mesh.translate([0.0, 0.0, 0.0])
    assert np.allclose(
        mesh.vectors, np.array([[[0, 1, 1], [1, 0, 1], [0, 0, 1]]])
    )


def test_translation():
    # Create a single face
    data = np.zeros(1, dtype=Mesh.dtype)
    data['vectors'][0] = np.array([[0, 1, 1], [1, 0, 1], [0, 0, 1]])

    mesh = Mesh(data, remove_empty_areas=False)
    assert np.allclose(
        mesh.vectors, np.array([[[0, 1, 1], [1, 0, 1], [0, 0, 1]]])
    )

    # Translate mesh with vector [1, 2, 3]
    mesh.translate([1.0, 2.0, 3.0])
    assert np.allclose(
        mesh.vectors, np.array([[[1, 3, 4], [2, 2, 4], [1, 2, 4]]])
    )


def test_no_transformation():
    # Create a single face
    data = np.zeros(1, dtype=Mesh.dtype)
    data['vectors'][0] = np.array([[0, 1, 1], [1, 0, 1], [0, 0, 1]])

    mesh = Mesh(data, remove_empty_areas=False)
    assert np.allclose(
        mesh.vectors, np.array([[[0, 1, 1], [1, 0, 1], [0, 0, 1]]])
    )

    # Transform mesh with identity matrix
    mesh.transform(np.eye(4))
    assert np.allclose(
        mesh.vectors, np.array([[[0, 1, 1], [1, 0, 1], [0, 0, 1]]])
    )
    assert np.allclose(mesh.areas, 0.5)


def test_transformation():
    # Create a single face
    data = np.zeros(1, dtype=Mesh.dtype)
    data['vectors'][0] = np.array([[0, 1, 1], [1, 0, 1], [0, 0, 1]])

    mesh = Mesh(data, remove_empty_areas=False)
    assert np.allclose(
        mesh.vectors, np.array([[[0, 1, 1], [1, 0, 1], [0, 0, 1]]])
    )

    # Transform mesh with identity matrix
    tr = np.zeros((4, 4))
    tr[0:3, 0:3] = Mesh.rotation_matrix([0, 0, 1], 0.5 * np.pi)
    tr[0:3, 3] = [1, 2, 3]
    mesh.transform(tr)
    assert np.allclose(
        mesh.vectors, np.array([[[0, 2, 4], [1, 3, 4], [1, 2, 4]]])
    )
    assert np.allclose(mesh.areas, 0.5)
