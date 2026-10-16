from fractions import Fraction

import numpy as np
import pytest

from stl import mesh

from .test_mesh_mass_mixed import subdivide
from .test_mesh_mass_sliver import _unit_cube, sliver


def box(center, size, sign=1):
    data, _, _, _ = _unit_cube()
    data['vectors'] = (data['vectors'] - 0.5) * size + center
    if sign < 0:
        data['vectors'] = data['vectors'][:, ::-1]
    return data


def box_properties(parts, density, raw=False):
    properties = []
    for data, sign in parts:
        lower = [Fraction(float(x)) for x in data['vectors'].min(axis=(0, 1))]
        upper = [Fraction(float(x)) for x in data['vectors'].max(axis=(0, 1))]
        sizes = [b - a for a, b in zip(lower, upper, strict=True)]
        volume = sign * sizes[0] * sizes[1] * sizes[2]
        center = [(a + b) / 2 for a, b in zip(lower, upper, strict=True)]
        properties.append((volume, center, sizes))
    volume = sum(v for v, _, _ in properties)
    center = [
        sum(v * c[k] for v, c, _ in properties) / volume for k in range(3)
    ]
    inertia = [[Fraction(0) for _ in range(3)] for _ in range(3)]
    for v, c, sizes in properties:
        delta = [c[k] - center[k] for k in range(3)]
        for i in range(3):
            for j in range(3):
                inertia[i][j] += v * (
                    (sizes[(i + 1) % 3] ** 2 + sizes[(i + 2) % 3] ** 2) / 12
                    + sum(delta[k] ** 2 for k in range(3) if k != i)
                    if i == j
                    else -delta[i] * delta[j]
                )
    if raw:
        return volume, center, inertia
    return (
        float(volume),
        float(volume * Fraction(density)),
        np.array([float(x) for x in center]),
        np.array(
            [[float(x * Fraction(density)) for x in row] for row in inertia]
        ),
    )


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('shift', [-1, 0, 1])
def test_partial_exact_mixed_moment_cancellation(axis, shift):
    span = 2**20
    ordinary = subdivide(box([0, 0, 0], [1, 1, 1]))
    positive = box([span, 2**-10, 0], [span, 1 / span, 1])
    negative = box([1.5 * span, 2.25 * 2**-10, 0], [span, 0.5 / span, 0.5], -1)
    negative['vectors'][:, :, 1] += shift * np.spacing(
        np.float32(2.25 * 2**-10)
    )
    parts = [(ordinary, 1), (positive, 1), (negative, -1)]
    volume, mass, center, inertia = box_properties(parts, 1.0)
    data = np.concatenate([part for part, _ in parts])
    data['vectors'] = np.roll(data['vectors'], axis, axis=2)
    original = data.copy()
    solid = mesh.Mesh(data, calculate_normals=False)
    assert solid.check(exact=True)
    result = solid.get_mass_properties_with_density(1.0)
    np.testing.assert_array_equal(result[0], volume)
    np.testing.assert_array_equal(result[1], mass)
    np.testing.assert_array_equal(result[2], np.roll(center, axis))
    np.testing.assert_array_equal(
        result[3], np.roll(inertia, (axis, axis), axis=(0, 1))
    )
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('density', [0.0, 2**-600, -(2**-600)])
def test_density_applied_before_exact_inertia_conversion(axis, density):
    span = 2.0**126
    positive, _, _, _ = _unit_cube()
    positive['vectors'] *= span
    positive['vectors'][:, :, 0] -= span
    negative, _, _, _ = _unit_cube()
    negative['vectors'] *= span
    negative['vectors'][:, :, 0] += span
    negative['vectors'] = negative['vectors'][:, ::-1]
    tiny, _, _, _ = _unit_cube()
    tiny['vectors'] *= [2**-120, 2**-149, 2**-149]
    parts = [(positive, 1), (negative, -1), (tiny, 1)]
    volume, mass, center, inertia = box_properties(parts, density)
    data = np.concatenate([part for part, _ in parts])
    data['vectors'] = np.roll(data['vectors'], axis, axis=2)
    original = data.copy()
    solid = mesh.Mesh(data, calculate_normals=False)
    # Normal inspection uses float32 cross products before mass integration.
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


@pytest.mark.parametrize('density', [np.inf, -np.inf, np.nan])
def test_exact_inertia_nonfinite_density(density):
    data, _, _, _ = sliver()
    solid = mesh.Mesh(data, calculate_normals=False)
    expected = solid.get_mass_properties_with_density(1.0)
    with np.errstate(invalid='ignore'):
        result = solid.get_mass_properties_with_density(density)
        np.testing.assert_array_equal(result[1], expected[1] * density)
        np.testing.assert_array_equal(result[3], expected[3] * density)
    np.testing.assert_array_equal(result[0], expected[0])
    np.testing.assert_array_equal(result[2], expected[2])


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_unrepresentable_exact_subset_preserves_complete_mesh(axis):
    unit, _, _, _ = _unit_cube()
    ordinary = subdivide(unit)
    ordinary['vectors'] *= 2.0**97
    positive = unit.copy()
    positive['vectors'] *= [2.0**126, 2.0**90, 2.0**126]
    negative = positive.copy()
    negative['vectors'][:, :, 0] += 2.0**127
    negative['vectors'] = negative['vectors'][:, ::-1]
    tiny = unit.copy()
    tiny['vectors'] *= [1, 2**-50, 2**-50]
    parts = [(ordinary, 1), (positive, 1), (negative, -1), (tiny, 1)]
    volume, mass, center, inertia = box_properties(parts, 1.0)
    data = np.concatenate([part for part, _ in parts])
    data['vectors'] = np.roll(data['vectors'], axis, axis=2)
    original = data.copy()
    solid = mesh.Mesh(data, calculate_normals=False)
    with np.errstate(over='ignore', invalid='ignore'):
        assert solid.check(exact=True)
        result = solid.get_mass_properties_with_density(1.0)
    np.testing.assert_array_equal(result[0], volume)
    np.testing.assert_array_equal(result[1], mass)
    np.testing.assert_array_equal(result[2], np.roll(center, axis))
    np.testing.assert_array_equal(
        result[3], np.roll(inertia, (axis, axis), axis=(0, 1))
    )
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('density', [np.inf, -np.inf, np.nan])
def test_unrepresentable_unit_inertia_nonfinite_density(axis, density):
    unit, _, _, _ = _unit_cube()
    positive = unit.copy()
    positive['vectors'] *= 2.0**126
    positive['vectors'][:, :, 0] -= 2.0**126
    negative = unit.copy()
    negative['vectors'] *= 2.0**126
    negative['vectors'][:, :, 0] += 2.0**126
    negative['vectors'] = negative['vectors'][:, ::-1]
    tiny = unit.copy()
    tiny['vectors'] *= [2**-120, 2**-149, 2**-149]
    parts = [(positive, 1), (negative, -1), (tiny, 1)]
    volume, center, inertia = box_properties(parts, 1.0, raw=True)
    signs = np.array([[(x > 0) - (x < 0) for x in row] for row in inertia])
    data = np.concatenate([part for part, _ in parts])
    data['vectors'] = np.roll(data['vectors'], axis, axis=2)
    solid = mesh.Mesh(data, calculate_normals=False)
    with np.errstate(over='ignore', invalid='ignore'):
        result = solid.get_mass_properties_with_density(density)
        expected = np.roll(signs, (axis, axis), axis=(0, 1)) * density
        np.testing.assert_array_equal(result[1], float(volume) * density)
    np.testing.assert_array_equal(result[0], float(volume))
    np.testing.assert_array_equal(
        result[2], np.roll([float(x) for x in center], axis)
    )
    np.testing.assert_array_equal(result[3], expected)
