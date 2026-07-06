# type: ignore[reportAttributeAccessIssue]

import io
import math

import numpy as np
import pytest
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

    assert np.allclose(mesh.areas, 2**0.5 / 2)
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


def test_mesh_identity_equality():
    data = np.zeros(2, dtype=Mesh.dtype)
    data['vectors'][0] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])

    mesh_a = Mesh(data.copy(), remove_empty_areas=False)
    mesh_b = Mesh(data.copy(), remove_empty_areas=False)
    assert mesh_a != mesh_b
    assert mesh_a == mesh_a

    lookup = {mesh_a: 'a', mesh_b: 'b'}
    assert lookup[mesh_a] == 'a'
    assert lookup[mesh_b] == 'b'

    assert mesh_a != 'not a mesh'


def _single_triangle_mesh() -> Mesh:
    data = np.zeros(1, dtype=Mesh.dtype)
    data['vectors'][0] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    return Mesh(data, remove_empty_areas=False)


def test_transform_updates_normals():
    # transform() used to rotate the vectors but leave the stored
    # normals at their pre-transform values.
    mesh = _single_triangle_mesh()
    rotation_x_90 = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    mesh.transform(rotation_x_90)

    transformed_normals = mesh.normals.copy()
    mesh.update_normals()
    utils.array_equals(transformed_normals, mesh.normals)


def test_translate_and_transform_refresh_min_max():
    mesh = _single_triangle_mesh()

    # Populate the lazy caches before mutating.
    assert np.allclose(mesh.min_, [0, 0, 0])
    assert np.allclose(mesh.max_, [1, 1, 0])

    mesh.translate([10, 20, 30])
    assert np.allclose(mesh.min_, [10, 20, 30])
    assert np.allclose(mesh.max_, [11, 21, 30])

    translation = np.identity(4)
    translation[0:3, 3] = [-10, -20, -30]
    mesh.transform(translation)
    assert np.allclose(mesh.min_, [0, 0, 0])
    assert np.allclose(mesh.max_, [1, 1, 0])


def test_rotate_refreshes_min_max():
    data = np.zeros(1, dtype=Mesh.dtype)
    data['vectors'][0] = np.array([[1, 0, 0], [2, 0, 0], [1, 1, 0]])
    mesh = Mesh(data, remove_empty_areas=False)

    assert np.allclose(mesh.min_, [1, 0, 0])
    assert np.allclose(mesh.max_, [2, 1, 0])

    mesh.rotate([0.5, 0.0, 0.0], math.radians(180))
    assert np.allclose(mesh.min_, [1, -1, 0], atol=1e-6)
    assert np.allclose(mesh.max_, [2, 0, 0], atol=1e-6)


def test_remove_duplicate_polygons_with_equal_vertex_sums():
    # Two DISTINCT triangles whose per-axis vertex sums match used to
    # be treated as duplicates, silently dropping one of them.
    data = np.zeros(2, dtype=Mesh.dtype)
    data['vectors'][0] = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0]])
    data['vectors'][1] = np.array([[0, 0, 0], [2, 0, 0], [1, 1, 0]])

    result = Mesh.remove_duplicate_polygons(data, RemoveDuplicates.SINGLE)
    assert len(result) == 2


def test_remove_duplicate_polygons_all_removes_duplicated_minority():
    data = np.zeros(6, dtype=Mesh.dtype)
    for i in range(4):
        data['vectors'][i] = np.array([[i, 0, 0], [i + 1, 0, 0], [i, 1, 0]])
    duplicated = np.array([[7, 7, 7], [8, 7, 7], [7, 8, 7]])
    data['vectors'][4] = duplicated
    data['vectors'][5] = duplicated

    result = Mesh.remove_duplicate_polygons(data, RemoveDuplicates.ALL)
    assert len(result) == 4
    assert np.allclose(result['vectors'], data['vectors'][:4])


def test_remove_duplicate_polygons_all_with_duplicates_sorting_first():
    # The duplicated polygon sorts BEFORE the unique ones; a
    # successor-based mask used to keep one duplicate copy and drop
    # the last unique polygon.
    data = np.zeros(5, dtype=Mesh.dtype)
    duplicated = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    data['vectors'][0] = duplicated
    data['vectors'][1] = duplicated
    for i in range(3):
        data['vectors'][i + 2] = np.array(
            [[i + 5, 0, 0], [i + 6, 0, 0], [i + 5, 1, 0]]
        )

    result = Mesh.remove_duplicate_polygons(data, RemoveDuplicates.ALL)
    assert len(result) == 3
    assert np.allclose(result['vectors'], data['vectors'][2:])


def test_remove_duplicate_polygons_all_majority_duplicates_fallback():
    # Documented fallback: when removing every duplicated polygon would
    # drop half the mesh or more, ALL keeps a single copy instead.
    data = np.zeros(4, dtype=Mesh.dtype)
    data['vectors'][0] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    data['vectors'][1] = data['vectors'][0]
    data['vectors'][2] = np.array([[5, 5, 5], [6, 5, 5], [5, 6, 5]])
    data['vectors'][3] = data['vectors'][2]

    result = Mesh.remove_duplicate_polygons(data, RemoveDuplicates.ALL)
    assert len(result) == 2


def test_save_load_accept_plain_int_modes(speedups):
    mesh = _single_triangle_mesh()
    mesh.speedups = False

    ascii_fh = io.BytesIO()
    mesh.save('ints.stl', fh=ascii_fh, mode=1)
    assert ascii_fh.getvalue().startswith(b'solid')

    binary_fh = io.BytesIO()
    mesh.save('ints.stl', fh=binary_fh, mode=2)
    assert not binary_fh.getvalue().startswith(b'solid')

    loaded = Mesh.from_file(
        'ints.stl',
        fh=io.BytesIO(binary_fh.getvalue()),
        mode=2,
        speedups=False,
    )
    assert np.allclose(loaded.vectors, mesh.vectors)

    with pytest.raises(ValueError, match=r'[Mm]ode'):
        mesh.save('ints.stl', fh=io.BytesIO(), mode=7)


def _tetrahedron() -> Mesh:
    # Consistently wound (outward) closed tetrahedron.
    a, b, c, d = (
        np.array([0, 0, 0]),
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
    )
    data = np.zeros(4, dtype=Mesh.dtype)
    data['vectors'][0] = np.array([a, c, b])
    data['vectors'][1] = np.array([a, b, d])
    data['vectors'][2] = np.array([a, d, c])
    data['vectors'][3] = np.array([b, c, d])
    return Mesh(data, remove_empty_areas=False)


def test_is_closed_exact_closed_mesh():
    assert _tetrahedron().check(exact=True)


def test_is_closed_exact_with_reversed_winding():
    # One face is wound backwards while its stored normal still points
    # the way the original winding implied. The orientation flag marks
    # the triangle as reversed and flips its edges back, so the mesh is
    # still recognized as closed.
    mesh = _tetrahedron()
    mesh.vectors[0] = mesh.vectors[0][::-1]
    assert mesh.check(exact=True)


def test_is_closed_exact_open_mesh(caplog):
    assert not _single_triangle_mesh().check(exact=True)
    assert 'mesh is not closed' in caplog.text


def test_is_closed_exact_duplicated_triangle():
    # Two identical triangles produce colliding directed edges.
    data = np.zeros(2, dtype=Mesh.dtype)
    data['vectors'][0] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    data['vectors'][1] = data['vectors'][0]
    assert not Mesh(data, remove_empty_areas=False).check(exact=True)


def test_is_closed_heuristic(caplog):
    assert _tetrahedron().check(exact=False)
    assert 'not exact' in caplog.text

    assert not _single_triangle_mesh().check(exact=False)


def test_logged_decorator_logger_name():
    # logged() must produce the same dotted name python-utils generates
    # Note: after other tests create Mesh instances, Logged.__new__ overwrites
    # the logger with the instantiated class's module and name
    assert Mesh.logger.name == 'stl.mesh.Mesh'
