import numpy

from stl.mesh import Mesh
from stl.base import BaseMesh
from stl.base import RemoveDuplicates

from . import utils


def test_units_1d():
    data = numpy.zeros(1, dtype=Mesh.dtype)
    data['vectors'][0] = numpy.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])

    mesh = Mesh(data, remove_empty_areas=False)
    mesh.update_units()

    assert mesh.areas == 0
    assert numpy.allclose(mesh.centroids, [[1, 0, 0]])
    utils.array_equals(mesh.normals, [0, 0, 0])
    utils.array_equals(mesh.units, [0, 0, 0])
    utils.array_equals(mesh.get_unit_normals(), [0, 0, 0])


def test_units_2d():
    data = numpy.zeros(2, dtype=Mesh.dtype)
    data['vectors'][0] = numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    data['vectors'][1] = numpy.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])

    mesh = Mesh(data, remove_empty_areas=False)
    mesh.update_units()

    assert numpy.allclose(mesh.areas, [0.5, 0.5])
    assert numpy.allclose(
        mesh.centroids, [[1 / 3, 1 / 3, 0], [2 / 3, 2 / 3, 0]]
    )
    assert numpy.allclose(mesh.normals, [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    assert numpy.allclose(mesh.units, [[0, 0, 1], [0, 0, -1]])
    assert numpy.allclose(
        mesh.get_unit_normals(), [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    )


def test_units_3d():
    data = numpy.zeros(1, dtype=Mesh.dtype)
    data['vectors'][0] = numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 1.0]])

    mesh = Mesh(data, remove_empty_areas=False)
    mesh.update_units()

    assert (mesh.areas - 2**0.5) < 0.0001
    assert numpy.allclose(mesh.centroids, [1 / 3, 1 / 3, 1 / 3])
    assert numpy.allclose(mesh.normals, [0.0, -1.0, 1.0])
    assert numpy.allclose(mesh.units[0], [0.0, -0.70710677, 0.70710677])
    assert numpy.allclose(numpy.linalg.norm(mesh.units, axis=-1), 1)
    assert numpy.allclose(
        mesh.get_unit_normals(), [0.0, -0.70710677, 0.70710677]
    )


def test_duplicate_polygons():
    data = numpy.zeros(6, dtype=Mesh.dtype)
    data['vectors'][0] = numpy.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    data['vectors'][1] = numpy.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]])
    data['vectors'][2] = numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    data['vectors'][3] = numpy.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]])
    data['vectors'][4] = numpy.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    data['vectors'][5] = numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

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

    assert numpy.allclose(
        mesh.vectors[0], numpy.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    )
    assert numpy.allclose(
        mesh.vectors[1], numpy.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]])
    )
    assert numpy.allclose(
        mesh.vectors[2], numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    )

    mesh = Mesh(data, remove_duplicate_polygons=RemoveDuplicates.ALL)
    assert mesh.data.size == 3

    assert numpy.allclose(
        mesh.vectors[0], numpy.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    )
    assert numpy.allclose(
        mesh.vectors[1], numpy.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]])
    )
    assert numpy.allclose(
        mesh.vectors[2], numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    )


def test_remove_all_duplicate_polygons():
    data = numpy.zeros(5, dtype=Mesh.dtype)
    data['vectors'][0] = numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    data['vectors'][1] = numpy.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    data['vectors'][2] = numpy.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]])
    data['vectors'][3] = numpy.array([[3, 0, 0], [0, 0, 0], [0, 0, 0]])
    data['vectors'][4] = numpy.array([[3, 0, 0], [0, 0, 0], [0, 0, 0]])

    mesh = Mesh(data, remove_duplicate_polygons=False)
    assert mesh.data.size == 5
    Mesh.remove_duplicate_polygons(mesh.data, RemoveDuplicates.NONE)

    mesh = Mesh(data, remove_duplicate_polygons=RemoveDuplicates.ALL)
    assert mesh.data.size == 3

    assert (
        mesh.vectors[0] == numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    ).all()
    assert (
        mesh.vectors[1] == numpy.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    ).all()
    assert (
        mesh.vectors[2] == numpy.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]])
    ).all()


def test_empty_areas():
    data = numpy.zeros(3, dtype=Mesh.dtype)
    data['vectors'][0] = numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    data['vectors'][1] = numpy.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])
    data['vectors'][2] = numpy.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])

    mesh = Mesh(data, calculate_normals=False, remove_empty_areas=False)
    assert mesh.data.size == 3

    # Test the normals recalculation which also calculates the areas by default
    mesh.areas[1] = 1
    mesh.areas[2] = 2
    assert numpy.allclose(mesh.areas, [[0.5], [1.0], [2.0]])

    mesh.centroids[1] = [1, 2, 3]
    mesh.centroids[2] = [4, 5, 6]
    assert numpy.allclose(
        mesh.centroids, [[1 / 3, 1 / 3, 0], [1, 2, 3], [4, 5, 6]]
    )

    mesh.update_normals(update_areas=False, update_centroids=False)
    assert numpy.allclose(mesh.areas, [[0.5], [1.0], [2.0]])
    assert numpy.allclose(
        mesh.centroids, [[1 / 3, 1 / 3, 0], [1, 2, 3], [4, 5, 6]]
    )

    mesh.update_normals(update_areas=True, update_centroids=True)
    assert numpy.allclose(mesh.areas, [[0.5], [0.0], [0.0]])
    assert numpy.allclose(
        mesh.centroids,
        [[1 / 3, 1 / 3, 0], [2 / 3, 1 / 3, 0], [2 / 3, 1 / 3, 0]],
    )

    mesh = Mesh(data, remove_empty_areas=True)
    assert mesh.data.size == 1


def test_base_mesh():
    data = numpy.zeros(10, dtype=BaseMesh.dtype)
    mesh = BaseMesh(data, remove_empty_areas=False)
    # Increment vector 0 item 0
    mesh.v0[0] += 1
    mesh.v1[0] += 2

    # Check item 0 (contains v0, v1 and v2)
    assert (
        mesh[0]
        == numpy.array(
            [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0], dtype=numpy.float32
        )
    ).all()
    assert (
        mesh.vectors[0]
        == numpy.array(
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.0, 0.0, 0.0]],
            dtype=numpy.float32,
        )
    ).all()
    assert (
        mesh.v0[0] == numpy.array([1.0, 1.0, 1.0], dtype=numpy.float32)
    ).all()
    assert (
        mesh.points[0]
        == numpy.array(
            [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0], dtype=numpy.float32
        )
    ).all()
    assert (
        mesh.x[0] == numpy.array([1.0, 2.0, 0.0], dtype=numpy.float32)
    ).all()

    mesh[0] = 3
    assert (
        mesh[0]
        == numpy.array(
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], dtype=numpy.float32
        )
    ).all()

    assert len(mesh) == len(list(mesh))
    assert (mesh.min_ < mesh.max_).all()
    mesh.update_normals()
    assert mesh.units.sum() == 0.0
    mesh.v0[:] = mesh.v1[:] = mesh.v2[:] = 0
    assert mesh.points.sum() == 0.0
