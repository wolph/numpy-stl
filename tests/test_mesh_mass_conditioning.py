from fractions import Fraction

import numpy as np
import pytest

from stl import mesh

from .test_mesh_mass_conversion import box, box_properties
from .test_mesh_mass_mixed import subdivide


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('density', [0.0, 0.7, -2.3])
def test_nearly_coplanar_solid_with_broad_facets(axis, density):
    vertices = np.array(
        [
            [0, 0, 0],
            [82610, -82610, -82609],
            [82609, 82610, -247827],
            [165219, -1, -330435],
        ],
        dtype=np.float32,
    )
    data = np.zeros(4, dtype=mesh.Mesh.dtype)
    data['vectors'] = vertices[[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]]
    points = [[Fraction(float(x)) for x in p] for p in vertices]
    a, b, c = points[1:]
    determinant = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    assert determinant == 1
    volume = determinant / 6
    center = [sum(p[k] for p in points) / 4 for k in range(3)]
    # The covariance of a uniform simplex is its vertex scatter / 20.
    covariance = [
        [
            sum((p[i] - center[i]) * (p[j] - center[j]) for p in points)
            * volume
            / 20
            for j in range(3)
        ]
        for i in range(3)
    ]
    inertia = np.array(
        [
            [
                float(
                    (
                        sum(covariance[k][k] for k in range(3) if k != i)
                        if i == j
                        else -covariance[i][j]
                    )
                    * Fraction(density)
                )
                for j in range(3)
            ]
            for i in range(3)
        ]
    )
    data['vectors'] = np.roll(data['vectors'], axis, axis=2)
    original = data.copy()
    solid = mesh.Mesh(data, calculate_normals=False)
    assert solid.check(exact=True)
    result = solid.get_mass_properties_with_density(density)
    np.testing.assert_array_equal(result[0], float(volume))
    np.testing.assert_array_equal(result[1], float(volume) * density)
    np.testing.assert_array_equal(
        result[2], np.roll([float(x) for x in center], axis)
    )
    np.testing.assert_array_equal(
        result[3], np.roll(inertia, (axis, axis), axis=(0, 1))
    )
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('density', [0.0, 0.7, -2.3])
@pytest.mark.parametrize('signed', [False, True])
def test_partial_exact_first_moment_cancellation(axis, density, signed):
    ordinary = subdivide(box([-(2.0**100), 0, 0], [2.0**78] * 3))
    positive = box([2.0**100, 0, 0], [2.0**78, 2.0**127, 2.0**29])
    narrow = box([2.0**120, 0, 0], [2.0**100, 2.0**60, 1])
    negative = box([0, 0, 0], [2.0**100, 2.0**60, 1], -1)
    parts = [(ordinary, 1), (positive, 1), (narrow, 1)]
    if signed:
        parts.append((negative, -1))
    volume, mass, center, inertia = box_properties(parts, density)
    assert center[0] == 2.0**45
    data = np.concatenate([part for part, _ in parts])
    data['vectors'] = np.roll(data['vectors'], axis, axis=2)
    original = data.copy()
    solid = mesh.Mesh(data, calculate_normals=False)
    with np.errstate(over='ignore', invalid='ignore'):
        assert solid.check(exact=True)
        result = solid.get_mass_properties_with_density(density)
    np.testing.assert_array_equal(result[0], volume)
    np.testing.assert_array_equal(result[1], mass)
    np.testing.assert_array_equal(result[2], np.roll(center, axis))
    np.testing.assert_array_equal(
        result[3], np.roll(inertia, (axis, axis), axis=(0, 1))
    )
    np.testing.assert_array_equal(data, original)
