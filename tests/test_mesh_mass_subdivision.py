import itertools

import numpy as np
import pytest

from stl import mesh


def _face(axes, index, axis, direction):
    other = [(axis + 1) % 3, (axis + 2) % 3]
    corners = []
    for offsets in [(0, 0), (1, 0), (1, 1), (0, 1)]:
        corner = np.array([axes[i][j] for i, j in enumerate(index)])
        corner[axis] = axes[axis][index[axis] + (direction == 1)]
        for i, offset in zip(other, offsets, strict=True):
            corner[i] = axes[i][index[i] + offset]
        corners.append(corner)
    if direction == -1:
        corners = corners[::-1]
    return corners


def _boundary(axes, occupied):
    triangles = []
    for index in sorted(occupied):
        for axis in range(3):
            for direction in (-1, 1):
                neighbor = list(index)
                neighbor[axis] += direction
                if tuple(neighbor) in occupied:
                    continue
                corners = _face(axes, index, axis, direction)
                triangles.extend(
                    [
                        [corners[0], corners[1], corners[2]],
                        [corners[0], corners[2], corners[3]],
                    ]
                )
    data = np.zeros(len(triangles), dtype=mesh.Mesh.dtype)
    data['vectors'] = triangles
    return data


def _retriangulated(
    strip: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    bounds = np.asarray(
        [
            [[0, 0, 0], [2, 2, 2]],
            [[2, 0, 0], [1e10, 1e-3, 1e-25]],
            [[1e10, 0, 0], [1e10 + 2048, 1, 1e-9]],
        ],
        dtype=np.float32,
    ).astype(float)
    # Plane partitions create thin strips on every cube crease; no holes,
    # T junctions, disconnected pieces or change to the represented solid.
    axes = [
        np.unique(np.r_[bounds[:, :, i].ravel(), [strip, 2 - strip]])
        for i in range(3)
    ]
    occupied = set()
    for index in itertools.product(*(range(len(a) - 1) for a in axes)):
        low = np.array([axes[i][j] for i, j in enumerate(index)])
        high = np.array([axes[i][j + 1] for i, j in enumerate(index)])
        if any(np.all(low >= b[0]) and np.all(high <= b[1]) for b in bounds):
            occupied.add(index)
    data = _boundary(axes, occupied)
    # Independent box volume/parallel-axis calculation, actual stored bounds.
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


@pytest.mark.parametrize('strip', [1e-2, 1e-5, 1e-7])
@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('density', [None, 0.7, 2.3])
def test_mass_properties_conforming_subdivision(strip, axis, density):
    data, expected_volume, expected_cog, expected_inertia = _retriangulated(
        strip
    )
    data['vectors'] = np.roll(data['vectors'], axis, axis=2)
    data = data[np.random.default_rng(3).permutation(len(data))]
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
        np.testing.assert_allclose(mass, expected_volume * density, rtol=1e-12)
        scale = density
    np.testing.assert_allclose(volume, expected_volume, rtol=1e-12)
    np.testing.assert_allclose(cog, np.roll(expected_cog, axis), rtol=1e-12)
    np.testing.assert_allclose(
        inertia,
        np.roll(expected_inertia, (axis, axis), axis=(0, 1)) * scale,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_array_equal(solid.data, before)


def test_zero_volume_large_planar_mesh():
    data = np.zeros(2, dtype=mesh.Mesh.dtype)
    data['vectors'] = [
        [[0, 0, 0], [1e10, 0, 0], [0, 2, 0]],
        [[1e10, 0, 0], [1e10, 2, 0], [0, 2, 0]],
    ]
    solid = mesh.Mesh(data, calculate_normals=False)
    with pytest.warns(RuntimeWarning):
        volume, cog, inertia = solid.get_mass_properties()
    assert volume == 0
    assert np.isnan(cog).all()
    assert np.isnan(inertia).all()
