from fractions import Fraction

import numpy as np
import pytest

from stl import mesh

from .test_mesh_mass_conversion import box
from .test_mesh_mass_mixed import subdivide
from .test_mesh_mass_sliver import _unit_cube


def same_winding_parts():
    ordinary = subdivide(box([0, 0, 0], [2.0**67] * 3))
    narrow, _, _, _ = _unit_cube()
    edges = np.array(
        [[2.0**100, -8 * 2.0**100, 0], [0, 2.0**100, 0], [0, 0, 1]]
    )
    center = np.array([2.0**100, 2.0**100, 0])
    narrow['vectors'] = (narrow['vectors'] - 0.5) @ edges + center
    return ordinary, narrow, edges, center


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('density', [None, 0.7, -2.3])
def test_same_winding_exact_subset_inertia_cancellation(axis, density):
    ordinary, narrow, edges, center = same_winding_parts()
    # Uniform parallelepiped covariance is the sum of edge outer products / 12.
    # Form the complete central tensor rationally before converting to float.
    edges = [[Fraction(float(x)) for x in edge] for edge in edges]
    centers = [[Fraction(0)] * 3, [Fraction(float(x)) for x in center]]
    volumes = [Fraction(2**201), Fraction(2**200)]
    volume = sum(volumes)
    cog = [
        sum(v * c[k] for v, c in zip(volumes, centers, strict=True)) / volume
        for k in range(3)
    ]
    covariance = [
        [sum(edge[i] * edge[j] for edge in edges) / 12 for j in range(3)]
        for i in range(3)
    ]
    second = [[Fraction(0) for _ in range(3)] for _ in range(3)]
    for index, (v, c) in enumerate(zip(volumes, centers, strict=True)):
        offset = [c[k] - cog[k] for k in range(3)]
        for i in range(3):
            for j in range(3):
                local = (
                    (Fraction(2**134, 12) if i == j else 0)
                    if index == 0
                    else covariance[i][j]
                )
                second[i][j] += v * (local + offset[i] * offset[j])
    tensor = [
        [
            sum(second[k][k] for k in range(3) if k != i)
            if i == j
            else -second[i][j]
            for j in range(3)
        ]
        for i in range(3)
    ]
    assert tensor[0][1] == 0
    scale = Fraction(1 if density is None else density)
    expected = np.array([[float(x * scale) for x in row] for row in tensor])
    data = np.concatenate((ordinary, narrow))
    data['vectors'] = np.roll(data['vectors'], axis, axis=2)
    original = data.copy()
    solid = mesh.Mesh(data, calculate_normals=False)
    with np.errstate(over='ignore', invalid='ignore'):
        assert solid.check(exact=True)
        if density is None:
            result_volume, result_cog, inertia = solid.get_mass_properties()
        else:
            result_volume, mass, result_cog, inertia = (
                solid.get_mass_properties_with_density(density)
            )
            assert mass == float(volume * scale)
    assert result_volume == float(volume)
    np.testing.assert_array_equal(
        result_cog, np.roll([float(x) for x in cog], axis)
    )
    np.testing.assert_array_equal(
        inertia, np.roll(expected, (axis, axis), axis=(0, 1))
    )
    np.testing.assert_array_equal(data, original)
