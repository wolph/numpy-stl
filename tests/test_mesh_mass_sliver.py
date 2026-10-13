import numpy as np
import pytest

from stl import mesh


def _unit_cube():
    # Closed project fixture, with analytic unit-cube moments.
    from pathlib import Path

    native = mesh.Mesh.from_file(
        str(Path(__file__).parent / 'stl_binary/Cube.stl')
    )
    native.vectors[:] /= 2
    return native.data, 1.0, np.full(3, 0.5), np.eye(3) / 6


def sliver():
    cube, cv, cc, ci = _unit_cube()
    faces = cube['vectors'].astype(float)
    face_index = np.flatnonzero(np.all(faces[:, :, 1] == 0, axis=1))[0]
    a, b, c = faces[face_index]
    apex = np.array([-6e7, -1e-6, 0.5], dtype=np.float32).astype(float)
    triangles = list(np.delete(faces, face_index, axis=0))
    triangles.extend([[a, b, apex], [b, c, apex], [c, a, apex]])
    data = np.zeros(len(triangles), dtype=mesh.Mesh.dtype)
    data['vectors'] = triangles
    vertices = np.vstack([a, b, c, apex])
    tv = abs(apex[1]) / 6
    tc = vertices.mean(axis=0)
    offsets = vertices - tc
    covariance = offsets.T @ offsets / 20
    ti = -tv * covariance
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        ti[i, i] = tv * (covariance[j, j] + covariance[k, k])
    volume = cv + tv
    cog = (cv * cc + tv * tc) / volume
    inertia = ci + ti
    for v, center in [(cv, cc), (tv, tc)]:
        offset = center - cog
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            inertia[i, i] += v * (offset[j] ** 2 + offset[k] ** 2)
            for j in range(3):
                if i != j:
                    inertia[i, j] -= v * offset[i] * offset[j]
    return data, volume, cog, inertia


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('density', [None, 0.7, 2.3])
def test_long_sliver_mass_properties(axis, density):
    data, reference_volume, reference_cog, reference_inertia = sliver()
    data['vectors'] = np.roll(data['vectors'], axis, axis=2)
    native = mesh.Mesh(data, calculate_normals=False)
    assert native.check(exact=True)
    assert len(np.unique(native._get_mass_properties_components()[0])) == 1
    if density is None:
        volume, cog, inertia = native.get_mass_properties()
        scale = 1
    else:
        volume, mass, cog, inertia = native.get_mass_properties_with_density(
            density
        )
        scale = density
        np.testing.assert_allclose(
            mass, reference_volume * density, rtol=1e-12
        )
    np.testing.assert_allclose(volume, reference_volume, rtol=1e-12)
    np.testing.assert_allclose(
        cog, np.roll(reference_cog, axis), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        inertia,
        np.roll(reference_inertia, (axis, axis), axis=(0, 1)) * scale,
        rtol=1e-12,
        atol=1e-12,
    )
