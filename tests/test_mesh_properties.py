import logging
from pathlib import Path

import numpy as np
import pytest
from stl.mesh import Mesh

from stl import stl

tolerance = 1e-5


@pytest.mark.parametrize('density', [None, 0.7, 2.3])
@pytest.mark.parametrize(
    'offset', [(0, 0, 0), (1000, -2000, 3000), (1e6, 1e6, 1e6)]
)
def test_translated_cuboid_mass_properties(
    binary_path: Path,
    density: float | None,
    offset: tuple[float, float, float],
) -> None:
    """A closed cuboid keeps its physical mass properties after translation."""
    original = Mesh.from_file(str(binary_path / 'Cube.stl'))
    sides = np.array([2.0, 3.0, 5.0])
    data = original.data.copy()
    data['vectors'] *= sides / 2
    cuboid = Mesh(data)
    cuboid.translate(offset)
    before = cuboid.data.copy()
    if density is None:
        volume, cog, inertia = cuboid.get_mass_properties()
        mass = volume
        material_density = 1.0
    else:
        volume, mass, cog, inertia = cuboid.get_mass_properties_with_density(
            density
        )
        material_density = density
    expected_volume = np.prod(sides)
    expected_inertia = (
        expected_volume
        * material_density
        / 12
        * np.diag(
            [
                sides[1] ** 2 + sides[2] ** 2,
                sides[0] ** 2 + sides[2] ** 2,
                sides[0] ** 2 + sides[1] ** 2,
            ]
        )
    )
    np.testing.assert_allclose(volume, expected_volume, rtol=1e-12)
    np.testing.assert_allclose(
        mass, expected_volume * material_density, rtol=1e-12
    )
    np.testing.assert_allclose(cog, sides / 2 + offset, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        inertia, expected_inertia, rtol=1e-12, atol=1e-10
    )
    np.testing.assert_array_equal(cuboid.data, before)


@pytest.mark.parametrize('density', [None, 2.3])
def test_empty_mass_properties(density: float | None) -> None:
    """Empty meshes keep their existing undefined mass-property results."""
    empty = Mesh(np.zeros(0, dtype=Mesh.dtype), remove_empty_areas=False)
    with np.errstate(invalid='ignore'):
        if density is None:
            volume, cog, inertia = empty.get_mass_properties()
        else:
            volume, _, cog, inertia = empty.get_mass_properties_with_density(
                density
            )
    assert volume == 0
    assert np.isnan(cog).all()
    assert np.isnan(inertia).all()


@pytest.mark.parametrize('density', [None, 0.7, 2.3])
@pytest.mark.parametrize('tiny_first', [False, True])
@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize(
    ('length', 'thickness', 'distance'),
    [(20, 2e-8, 500000), (1e10, 3.16e-14, 1e15)],
)
def test_mass_properties_with_distant_small_component(
    binary_path: Path,
    density: float | None,
    tiny_first: bool,
    axis: int,
    length: float,
    thickness: float,
    distance: float,
) -> None:
    """A distant low-volume component cannot choose the integration origin."""
    cube = Mesh.from_file(str(binary_path / 'Cube.stl'))
    tiny_data = cube.data.copy()
    sides = np.full(3, thickness)
    sides[axis] = length
    offset = np.zeros(3)
    offset[axis] = distance
    tiny_data['vectors'] = tiny_data['vectors'] * sides / 2 + offset
    parts = [tiny_data, cube.data] if tiny_first else [cube.data, tiny_data]
    compound = Mesh(np.concatenate(parts))
    before = compound.data.copy()
    assert compound.check(exact=True)

    # Analytic cuboid moments use the actual stored float32 bounds.
    tiny_vectors = tiny_data['vectors'].astype(np.float64)
    low = tiny_vectors.min(axis=(0, 1))
    high = tiny_vectors.max(axis=(0, 1))
    sizes = high - low
    tiny_volume = sizes.prod()
    tiny_cog = (low + high) / 2
    expected_volume = 8 + tiny_volume
    expected_cog = (8 * np.ones(3) + tiny_volume * tiny_cog) / expected_volume
    expected_inertia = np.eye(3) * (16 / 3)
    expected_inertia += np.diag(
        tiny_volume / 12 * (np.sum(sizes**2) - sizes**2)
    )
    for volume, center in [(8, np.ones(3)), (tiny_volume, tiny_cog)]:
        shift = center - expected_cog
        expected_inertia += volume * (
            np.eye(3) * (shift @ shift) - np.outer(shift, shift)
        )

    if density is None:
        volume, cog, inertia = compound.get_mass_properties()
    else:
        volume, _, cog, inertia = compound.get_mass_properties_with_density(
            density
        )
        expected_inertia *= density
    np.testing.assert_allclose(volume, expected_volume, rtol=1e-12)
    np.testing.assert_allclose(cog, expected_cog, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        inertia, expected_inertia, rtol=1e-12, atol=1e-10
    )
    np.testing.assert_array_equal(compound.data, before)


@pytest.mark.parametrize('density', [None, 0.0, 2.3])
def test_mass_properties_with_inner_cavity(
    binary_path: Path, density: float | None
) -> None:
    """Opposite winding subtracts a cavity rather than adding its volume."""
    original = Mesh.from_file(str(binary_path / 'Cube.stl'))
    outer, inner = original.data.copy(), original.data.copy()
    outer['vectors'] *= 1.5
    inner['vectors'] = inner['vectors'][:, ::-1] / 2 + 1
    hollow = Mesh(np.concatenate([outer, inner]))
    assert hollow.check(exact=True)
    if density is None:
        volume, cog, inertia = hollow.get_mass_properties()
        scale = 1.0
    else:
        volume, mass, cog, inertia = hollow.get_mass_properties_with_density(
            density
        )
        np.testing.assert_allclose(mass, 26 * density)
        scale = density
    np.testing.assert_allclose(volume, 26)
    np.testing.assert_allclose(cog, [1.5, 1.5, 1.5])
    np.testing.assert_allclose(
        inertia, np.eye(3) * (40.5 - 1 / 6) * scale, atol=1e-12
    )


def test_zero_volume_component_keeps_cube_mass_properties(
    binary_path: Path,
) -> None:
    """A closed zero-volume surface cannot contaminate another component."""
    cube = Mesh.from_file(str(binary_path / 'Cube.stl'))
    vertices = np.array([[10, 10, 0], [11, 10, 0], [10, 11, 0], [11, 11, 0]])
    faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    data = np.zeros(4, dtype=Mesh.dtype)
    data['vectors'] = vertices[faces]
    compound = Mesh(np.concatenate([cube.data, data]))
    assert compound.check(exact=True)
    volume, cog, inertia = compound.get_mass_properties()
    np.testing.assert_allclose(volume, 8)
    np.testing.assert_allclose(cog, [1, 1, 1])
    np.testing.assert_allclose(inertia, np.eye(3) * 16 / 3, atol=1e-12)


def close(a, b):
    return np.allclose(a, b, atol=tolerance)


def test_mass_properties_for_half_donut(binary_ascii_path, speedups):
    """
    Checks the results of method get_mass_properties() on
    STL ASCII and binary files HalfDonut.stl
    One checks the results obtained with stl
    with the ones obtained with meshlab
    """
    filename = binary_ascii_path / 'HalfDonut.stl'
    mesh = stl.StlMesh(str(filename), speedups=speedups)
    volume, cog, inertia = mesh.get_mass_properties()
    assert close([volume], [2.343149])
    assert close(cog, [1.500001, 0.209472, 1.500001])
    assert close(
        inertia,
        [
            [+1.390429, +0.000000, +0.000000],
            [+0.000000, +2.701025, +0.000000],
            [+0.000000, +0.000000, +1.390429],
        ],
    )


def test_mass_properties_for_moon(binary_ascii_path, speedups):
    """
    Checks the results of method get_mass_properties() on
    STL ASCII and binary files Moon.stl
    One checks the results obtained with stl
    with the ones obtained with meshlab
    """
    filename = binary_ascii_path / 'Moon.stl'
    mesh = stl.StlMesh(str(filename), speedups=speedups)
    volume, cog, inertia = mesh.get_mass_properties()
    assert close([volume], [0.888723])
    assert close(cog, [0.906913, 0.170731, 1.500001])
    assert close(
        inertia,
        [
            [+0.562097, -0.000457, +0.000000],
            [-0.000457, +0.656851, +0.000000],
            [+0.000000, +0.000000, +0.112465],
        ],
    )


def _check_star_mass_properties(filename: str, speedups: bool) -> None:
    mesh = stl.StlMesh(filename, speedups=speedups)
    volume, cog, inertia = mesh.get_mass_properties()
    assert close([volume], [1.416599])
    assert close(cog, [1.299040, 0.170197, 1.499999])
    assert close(
        inertia,
        [
            [+0.509549, +0.000000, -0.000000],
            [+0.000000, +0.991236, +0.000000],
            [-0.000000, +0.000000, +0.509550],
        ],
    )


def test_mass_properties_for_star(binary_ascii_path, speedups):
    """
    Checks the results of method get_mass_properties() on
    STL ASCII and binary files Star.stl
    One checks the results obtained with stl
    with the ones obtained with meshlab
    """
    _check_star_mass_properties(str(binary_ascii_path / 'Star.stl'), speedups)


def test_mass_properties_for_star_with_empty_header(binary_path, speedups):
    """
    Checks get_mass_properties() on the binary-only
    StarWithEmptyHeader.stl (with no header).
    """
    _check_star_mass_properties(
        str(binary_path / 'StarWithEmptyHeader.stl'), speedups
    )


def test_mass_properties_for_half_donut_with_density(
    binary_ascii_path, speedups
):
    """
    Checks the results of method get_mass_properties_with_density() on
    STL ASCII and binary files HalfDonut.stl
    One checks the results obtained with stl
    with the ones obtained with meshlab
    """
    filename = binary_ascii_path / 'HalfDonut.stl'
    mesh = stl.StlMesh(str(filename), speedups=speedups)
    volume, mass, cog, inertia = mesh.get_mass_properties_with_density(1.23)

    assert close([mass], [2.882083302268982])
    assert close([volume], [2.343149026234945])
    assert close(cog, [1.500001, 0.209472, 1.500001])
    print('inertia')
    np.set_printoptions(suppress=True)
    print(inertia)
    assert close(
        inertia,
        [
            [+1.71022851, +0.00000001, -0.00000011],
            [+0.00000001, +3.32226227, +0.00000002],
            [-0.00000011, +0.00000002, +1.71022859],
        ],
    )


@pytest.mark.parametrize(
    'filename, expected_result',
    [
        ('Cube.stl', True),
        ('HalfDonut.stl', False),
        ('Moon.stl', False),
        ('Star.stl', False),
    ],
)
def test_is_convex(binary_ascii_path, speedups, filename, expected_result):
    """Check the is_convex() method on various STL files."""
    filepath = binary_ascii_path / filename
    mesh = stl.StlMesh(str(filepath), speedups=speedups)
    assert mesh.is_convex() == expected_result


def test_mass_properties_open_mesh_warns_and_returns(caplog):
    """An open mesh warns instead of raising and still returns values."""
    data = np.zeros(1, dtype=Mesh.dtype)
    data['vectors'][0] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    open_mesh = Mesh(data, remove_empty_areas=False)

    with caplog.at_level(logging.WARNING):
        volume, cog, inertia = open_mesh.get_mass_properties()

    assert 'mesh is not closed' in caplog.text
    assert np.isclose(volume, 0)
    assert cog.shape == (3,)
    assert inertia.shape == (3, 3)
