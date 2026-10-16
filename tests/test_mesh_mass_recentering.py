import numpy as np
import pytest

from stl import mesh

from .test_mesh_mass_subdivision import _boundary


def moderate_appendage():
    bounds = np.array(
        [[[0, 0, 0], [2, 2, 2]], [[-1000, 0, 0], [0, 1e-4, 1e-4]]],
        dtype=np.float32,
    ).astype(float)
    axes = [np.unique(bounds[:, :, i]) for i in range(3)]
    data = _boundary(
        axes, {(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1), (0, 0, 0)}
    )
    lengths = bounds[:, 1] - bounds[:, 0]
    volumes = np.prod(lengths, axis=1)
    centers = bounds.mean(axis=1)
    volume = volumes.sum()
    cog = (volumes[:, None] * centers).sum(axis=0) / volume
    inertia = np.zeros((3, 3))
    for v, length, center in zip(volumes, lengths, centers, strict=True):
        offset = center - cog
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            inertia[i, i] += v * (
                (length[j] ** 2 + length[k] ** 2) / 12
                + offset[j] ** 2
                + offset[k] ** 2
            )
            for j in range(3):
                if i != j:
                    inertia[i, j] -= v * offset[i] * offset[j]
    return data, volume, cog, inertia


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('density', [None, 0.7, 2.3])
def test_moderate_appendage_centered_integration(axis, density):
    data, expected_volume, expected_cog, expected_inertia = (
        moderate_appendage()
    )
    data['vectors'] = np.roll(data['vectors'], axis, axis=2)
    solid = mesh.Mesh(data.copy(), calculate_normals=False)
    before = solid.data.copy()
    assert solid.check(exact=True)
    assert len(np.unique(solid._get_mass_properties_components()[0])) == 1
    if density is None:
        volume, cog, inertia = solid.get_mass_properties()
        scale = 1.0
    else:
        volume, mass, cog, inertia = solid.get_mass_properties_with_density(
            density
        )
        scale = density
        np.testing.assert_allclose(mass, expected_volume * density, rtol=1e-12)
    np.testing.assert_allclose(volume, expected_volume, rtol=1e-12)
    np.testing.assert_allclose(
        cog, np.roll(expected_cog, axis), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        inertia,
        np.roll(expected_inertia, (axis, axis), axis=(0, 1)) * scale,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_array_equal(solid.data, before)
