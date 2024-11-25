# type: ignore[reportAttributeAccessIssue]
import numpy as np

from stl.base import BaseMesh, RemoveDuplicates
from stl.mesh import Mesh

from . import utils


def test_units_1d():
    data = np.zeros(1, dtype=Mesh.dtype)
    data['vectors'][0] = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])

    mesh = Mesh(data, remove_empty_areas=False)
    mesh.update_units()

    assert mesh.areas == 0
    assert np.allclose(mesh.centroids, [[1, 0, 0]])
    utils.array_equals(mesh.normals, [0, 0, 0])
    utils.array_equals(mesh.units, [0, 0, 0])
    utils.array_equals(mesh.get_unit_normals(), [0, 0, 0])


def test_units_2d():
    data = np.zeros(2, dtype=Mesh.dtype)
    data['vectors'][0] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    data['vectors'][1] = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])

    mesh = Mesh(data, remove_empty_areas=False)
    mesh.update_units()

    assert np.allclose(mesh.areas, [0.5, 0.5])
    assert np.allclose(mesh.centroids, [[1 / 3, 1 / 3, 0], [2 / 3, 2 / 3, 0]])
    assert np.allclose(mesh.normals, [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    assert np.allclose(mesh.units, [[0, 0, 1], [0, 0, -1]])
    assert np.allclose(
        mesh.get_unit_normals(), [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    )


def test_units_3d():
    data = np.zeros(1, dtype=Mesh.dtype)
    data['vectors'][0] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1.0]])

    mesh = Mesh(data, remove_empty_areas=False)
    mesh.update_units()

    assert (mesh.areas - 2**0.5) < 0.0001
    assert np.allclose(mesh.centroids, [1 / 3, 1 / 3, 1 / 3])
    assert np.allclose(mesh.normals, [0.0, -1.0, 1.0])
    assert np.allclose(mesh.units[0], [0.0, -0.70710677, 0.70710677])
    assert np.allclose(np.linalg.norm(mesh.units, axis=-1), 1)
    assert np.allclose(mesh.get_unit_normals(), [0.0, -0.70710677, 0.70710677])


def test_duplicate_polygons():
    data = np.zeros(6, dtype=Mesh.dtype)
    data['vectors'][0] = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    data['vectors'][1] = np.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]])
    data['vectors'][2] = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    data['vectors'][3] = np.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]])
    data['vectors'][4] = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    data['vectors'][5] = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    mesh = Mesh(data)
    assert mesh.data.size == 6

    mesh = Mesh(data, remove_duplicate_polygons=0)
    assert mesh.data.size == 6

    mesh = Mesh(data, remove_duplicate_polygons=False)
    assert mesh.data.size == 6

    mesh = Mesh(data, remove_duplicate_polygons=None)
    assert mesh.data.size == 6

    mesh = Mesh(data, remove_duplicate_polygons=RemoveDuplicates.NONE)
    assert mesh.data.size == 6

    mesh = Mesh(data, remove_duplicate_polygons=RemoveDuplicates.SINGLE)
    assert mesh.data.size == 3

    mesh = Mesh(data, remove_duplicate_polygons=True)
    assert mesh.data.size == 3

    assert np.allclose(
        mesh.vectors[0], np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    )
    assert np.allclose(
        mesh.vectors[1], np.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]])
    )
    assert np.allclose(
        mesh.vectors[2], np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    )

    mesh = Mesh(data, remove_duplicate_polygons=RemoveDuplicates.ALL)
    assert mesh.data.size == 3

    assert np.allclose(
        mesh.vectors[0], np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    )
    assert np.allclose(
        mesh.vectors[1], np.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]])
    )
    assert np.allclose(
        mesh.vectors[2], np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    )


def test_remove_all_duplicate_polygons():
    data = np.zeros(5, dtype=Mesh.dtype)
    data['vectors'][0] = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    data['vectors'][1] = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    data['vectors'][2] = np.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]])
    data['vectors'][3] = np.array([[3, 0, 0], [0, 0, 0], [0, 0, 0]])
    data['vectors'][4] = np.array([[3, 0, 0], [0, 0, 0], [0, 0, 0]])

    mesh = Mesh(data, remove_duplicate_polygons=False)
    assert mesh.data.size == 5
    Mesh.remove_duplicate_polygons(mesh.data, RemoveDuplicates.NONE)

    mesh = Mesh(data, remove_duplicate_polygons=RemoveDuplicates.ALL)
    assert mesh.data.size == 3

    assert (
        mesh.vectors[0] == np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    ).all()
    assert (
        mesh.vectors[1] == np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    ).all()
    assert (
        mesh.vectors[2] == np.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]])
    ).all()


def test_empty_areas():
    data = np.zeros(3, dtype=Mesh.dtype)
    data['vectors'][0] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    data['vectors'][1] = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])
    data['vectors'][2] = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])

    mesh = Mesh(data, calculate_normals=False, remove_empty_areas=False)
    assert mesh.data.size == 3

    # Test the normals recalculation which also calculates the areas by default
    mesh.areas[1] = 1
    mesh.areas[2] = 2
    assert np.allclose(mesh.areas, [[0.5], [1.0], [2.0]])

    mesh.centroids[1] = [1, 2, 3]
    mesh.centroids[2] = [4, 5, 6]
    assert np.allclose(
        mesh.centroids, [[1 / 3, 1 / 3, 0], [1, 2, 3], [4, 5, 6]]
    )

    mesh.update_normals(update_areas=False, update_centroids=False)
    assert np.allclose(mesh.areas, [[0.5], [1.0], [2.0]])
    assert np.allclose(
        mesh.centroids, [[1 / 3, 1 / 3, 0], [1, 2, 3], [4, 5, 6]]
    )

    mesh.update_normals(update_areas=True, update_centroids=True)
    assert np.allclose(mesh.areas, [[0.5], [0.0], [0.0]])
    assert np.allclose(
        mesh.centroids,
        [[1 / 3, 1 / 3, 0], [2 / 3, 1 / 3, 0], [2 / 3, 1 / 3, 0]],
    )

    mesh = Mesh(data, remove_empty_areas=True)
    assert mesh.data.size == 1


def test_base_mesh():
    data = np.zeros(10, dtype=BaseMesh.dtype)
    mesh = BaseMesh(data, remove_empty_areas=False)
    # Increment vector 0 item 0
    mesh.v0[0] += 1
    mesh.v1[0] += 2

    # Check item 0 (contains v0, v1 and v2)
    assert (
        mesh[0]
        == np.array(
            [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
    ).all()
    assert (
        mesh.vectors[0]
        == np.array(
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
    ).all()
    assert (mesh.v0[0] == np.array([1.0, 1.0, 1.0], dtype=np.float32)).all()
    assert (
        mesh.points[0]
        == np.array(
            [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
    ).all()
    assert (mesh.x[0] == np.array([1.0, 2.0, 0.0], dtype=np.float32)).all()

    mesh[0] = 3
    assert (
        mesh[0]
        == np.array(
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], dtype=np.float32
        )
    ).all()

    assert len(mesh) == len(list(mesh))
    assert (mesh.min_ < mesh.max_).all()
    mesh.update_normals()
    assert mesh.units.sum() == 0.0
    mesh.v0[:] = mesh.v1[:] = mesh.v2[:] = 0
    assert mesh.points.sum() == 0.0
