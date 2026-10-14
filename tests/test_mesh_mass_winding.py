import tracemalloc
from fractions import Fraction

import numpy as np
import pytest

from stl import mesh

from .test_mesh_mass_mixed import subdivide
from .test_mesh_mass_sliver import _unit_cube


def winding_parts():
    narrow = np.array(
        [
            [565499, 1633004, 136135],
            [0, 2888250, 240676],
            [564749, 1630353, 135914],
            [103499, 0, 0],
        ],
        dtype=np.float32,
    )
    narrow -= narrow.mean(axis=0)
    u, v, w = 9041091 / 2**24, 8844941 / 2**24, 14763316 / 2**26
    ordinary = np.array(
        [[u, v, w], [-u, v, -w], [u, -v, -w], [-u, -v, w]],
        dtype=np.float32,
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    parts = []
    properties = []
    for vertices, negative in [(ordinary, False), (narrow, True)]:
        points = [[Fraction(float(x)) for x in row] for row in vertices]
        a, b, c = [[p[i] - points[0][i] for i in range(3)] for p in points[1:]]
        determinant = (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
        assert determinant > 0
        if negative:
            assert determinant == 1
        volume = determinant / (-6 if negative else 6)
        center = [sum(p[i] for p in points) / 4 for i in range(3)]
        assert center == [0, 0, 0]
        covariance = [
            [volume * sum(p[i] * p[j] for p in points) / 20 for j in range(3)]
            for i in range(3)
        ]
        properties.append((volume, covariance))
        data = np.zeros(4, dtype=mesh.Mesh.dtype)
        data['vectors'] = vertices[faces[:, ::-1] if negative else faces]
        parts.append(data if negative else subdivide(data))
    volume = sum(p[0] for p in properties)
    second = [
        [sum(p[1][i][j] for p in properties) for j in range(3)]
        for i in range(3)
    ]
    inertia = [
        [
            sum(second[k][k] for k in range(3) if k != i)
            if i == j
            else -second[i][j]
            for j in range(3)
        ]
        for i in range(3)
    ]
    return np.concatenate(parts), volume, inertia


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('density', [None, 0.7, -2.3])
@pytest.mark.parametrize('vertex_roll', [0, 1, 2])
def test_conditioned_component_winding(axis, density, vertex_roll):
    data, expected_volume, expected_inertia = winding_parts()
    data['vectors'] = np.roll(data['vectors'], axis, axis=2)
    data['vectors'] = np.roll(data['vectors'], vertex_roll, axis=1)
    original = data.copy()
    solid = mesh.Mesh(data, calculate_normals=False)
    assert solid.check(exact=True)
    if density is None:
        volume, cog, inertia = solid.get_mass_properties()
        scale = Fraction(1)
    else:
        volume, mass, cog, inertia = solid.get_mass_properties_with_density(
            density
        )
        scale = Fraction(density)
        assert mass == float(expected_volume) * density
    assert volume == float(expected_volume)
    np.testing.assert_array_equal(cog, [0, 0, 0])
    expected = np.array(
        [[float(x * scale) for x in row] for row in expected_inertia]
    )
    np.testing.assert_array_equal(
        inertia, np.roll(expected, (axis, axis), axis=(0, 1))
    )
    np.testing.assert_array_equal(data, original)


def test_strided_exact_integration_memory():
    unit, _, _, _ = _unit_cube()
    data = np.tile(unit, 300)
    data['vectors'] += np.repeat(np.arange(300), len(unit))[:, None, None] * [
        2,
        0,
        0,
    ]
    solid = mesh.Mesh(data, calculate_normals=False)
    assert not solid.vectors.flags.c_contiguous
    tracemalloc.start()
    try:
        result = solid._get_mass_properties_exact(1.0)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result is not None
    assert result[0] == 300
    np.testing.assert_array_equal(result[2], [299.5, 0.5, 0.5])
    # Streaming must not allocate a facet-sized contiguous vector copy.
    assert peak < solid.vectors.nbytes // 2
