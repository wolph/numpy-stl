import numpy as np
import pytest

from stl import mesh

from .test_mesh_mass_sliver import _unit_cube, sliver


def mixed_components(zero_volume_subset=False, ordinary_sign=1):
    ordinary, volume, cog, inertia = _unit_cube()
    ordinary['vectors'] += [0, 0, 4]
    if ordinary_sign < 0:
        ordinary['vectors'] = ordinary['vectors'][:, ::-1]
    parts = [
        (
            ordinary_sign * volume,
            cog + np.array([0, 0, 4]),
            ordinary_sign * inertia,
        )
    ]
    narrow, volume, cog, inertia = sliver()
    data = [ordinary, narrow]
    parts.append((volume, cog, inertia))
    if zero_volume_subset:
        reversed_narrow = narrow.copy()
        reversed_narrow['vectors'] = reversed_narrow['vectors'][:, ::-1]
        reversed_narrow['vectors'] += [0, 0, -4]
        data.append(reversed_narrow)
        parts.append((-volume, cog + np.array([0, 0, -4]), -inertia))
    volume = sum(part[0] for part in parts)
    cog = sum(v * c for v, c, _ in parts) / volume
    inertia = sum(part[2] for part in parts)
    for v, center, _ in parts:
        offset = center - cog
        inertia += v * (
            np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
        )
    return np.concatenate(data), volume, cog, inertia


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('density', [0.0, 0.7, -2.3])
@pytest.mark.parametrize('zero_volume_subset', [False, True])
def test_mixed_component_mass_properties(
    axis, density, zero_volume_subset, monkeypatch
):
    data, expected_volume, expected_cog, expected_inertia = mixed_components(
        zero_volume_subset
    )
    data['vectors'] = np.roll(data['vectors'], axis, axis=2)
    solid = mesh.Mesh(data.copy(), calculate_normals=False)
    assert solid.check(exact=True)
    calls = []
    original = mesh.Mesh._get_mass_properties_exact

    def count_exact(self, density, *args):
        calls.append(len(args[0]) if args else len(self))
        return original(self, density, *args)

    monkeypatch.setattr(mesh.Mesh, '_get_mass_properties_exact', count_exact)
    volume, mass, cog, inertia = solid.get_mass_properties_with_density(
        density
    )
    np.testing.assert_allclose(volume, expected_volume, rtol=1e-12)
    np.testing.assert_allclose(mass, expected_volume * density, rtol=1e-12)
    np.testing.assert_allclose(
        cog, np.roll(expected_cog, axis), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        inertia,
        np.roll(expected_inertia, (axis, axis), axis=(0, 1)) * density,
        rtol=1e-12,
        atol=1e-7,
    )
    np.testing.assert_array_equal(solid.data, data)
    # An unrelated ordinary solid must not increase scalar exact work.
    # A signed zero-volume subset needs whole-mesh raw moments instead.
    assert calls == ([len(solid)] if zero_volume_subset else [14])


def test_opposing_signed_components_use_exact_moments(monkeypatch):
    data, expected_volume, expected_cog, expected_inertia = mixed_components(
        ordinary_sign=-1
    )
    solid = mesh.Mesh(data.copy(), calculate_normals=False)
    calls = []
    original = mesh.Mesh._get_mass_properties_exact

    def count_exact(self, density, *args):
        calls.append(len(args[0]) if args else len(self))
        return original(self, density, *args)

    monkeypatch.setattr(mesh.Mesh, '_get_mass_properties_exact', count_exact)
    volume, cog, inertia = solid.get_mass_properties()
    np.testing.assert_allclose(volume, expected_volume, rtol=1e-8)
    np.testing.assert_allclose(cog, expected_cog, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(inertia, expected_inertia, rtol=1e-8, atol=1e-8)
    np.testing.assert_array_equal(solid.data, data)
    assert calls == [len(solid)]


def test_zero_volume_narrow_component_preserves_ordinary_solid():
    ordinary, expected_volume, expected_cog, expected_inertia = _unit_cube()
    narrow = np.zeros(2, dtype=mesh.Mesh.dtype)
    narrow['vectors'][0] = [[3, 0, 0], [6e7, 0, 0], [3, 1e-6, 0]]
    narrow['vectors'][1] = narrow['vectors'][0, ::-1]
    solid = mesh.Mesh(
        np.concatenate((ordinary, narrow)), calculate_normals=False
    )
    assert solid.check(exact=True)
    volume, cog, inertia = solid.get_mass_properties()
    np.testing.assert_allclose(volume, expected_volume, rtol=1e-12)
    np.testing.assert_allclose(cog, expected_cog, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        inertia, expected_inertia, rtol=1e-12, atol=1e-12
    )


def test_zero_volume_narrow_and_ordinary_components():
    ordinary, _, _, _ = _unit_cube()
    ordinary['vectors'][:, :, 2] = 0
    narrow = np.zeros(2, dtype=mesh.Mesh.dtype)
    narrow['vectors'][0] = [[3, 0, 0], [6e7, 0, 0], [3, 1e-6, 0]]
    narrow['vectors'][1] = narrow['vectors'][0, ::-1]
    solid = mesh.Mesh(
        np.concatenate((ordinary, narrow)), calculate_normals=False
    )
    with pytest.warns(RuntimeWarning):
        volume, cog, inertia = solid.get_mass_properties()
    assert volume == 0
    assert np.isnan(cog).all()
    assert np.isnan(inertia).all()


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_mixed_components_preserve_small_principal_inertia(axis):
    rod, _, _, _ = _unit_cube()
    rod['vectors'] *= [1e10, 1e-10, 1e-10]
    tiny, _, _, _ = _unit_cube()
    tiny['vectors'] *= 1e-12
    tiny['vectors'] += [0, 0, 1e-8]
    parts = []
    for data in (rod, tiny):
        bounds = np.array(
            [
                data['vectors'].min(axis=(0, 1)),
                data['vectors'].max(axis=(0, 1)),
            ],
            dtype=np.float64,
        )
        lengths = bounds[1] - bounds[0]
        volume = np.prod(lengths)
        center = bounds.mean(axis=0)
        inertia = np.array(
            [
                volume
                * (lengths[(i + 1) % 3] ** 2 + lengths[(i + 2) % 3] ** 2)
                / 12
                for i in range(3)
            ]
        )
        parts.append((volume, center, inertia))
    expected_volume = sum(v for v, _, _ in parts)
    expected_cog = sum(v * c for v, c, _ in parts) / expected_volume
    expected_inertia = sum(inertia for _, _, inertia in parts)
    for volume, center, _ in parts:
        delta = center - expected_cog
        expected_inertia += volume * np.array(
            [
                delta[(i + 1) % 3] ** 2 + delta[(i + 2) % 3] ** 2
                for i in range(3)
            ]
        )
    data = np.concatenate((rod, tiny))
    data['vectors'] = np.roll(data['vectors'], axis, axis=2)
    result = mesh.Mesh(data, calculate_normals=False).get_mass_properties()
    np.testing.assert_allclose(result[0], expected_volume, rtol=1e-12)
    np.testing.assert_allclose(
        result[1], np.roll(expected_cog, axis), rtol=1e-12, atol=1e-30
    )
    np.testing.assert_allclose(
        np.diag(result[2]), np.roll(expected_inertia, axis), rtol=1e-12, atol=0
    )
