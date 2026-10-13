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
    ('length', 'thickness', 'distance', 'cube_distance'),
    [
        (20, 2e-8, 500000, 0),
        (1e10, 3.16e-14, 1e15, 0),
        (1e6, 2e-9, 0, 1e6),
        (1e15, 3.162e-14, -1e15, 0),
    ],
)
def test_mass_properties_with_distant_small_component(
    binary_path: Path,
    density: float | None,
    tiny_first: bool,
    axis: int,
    length: float,
    thickness: float,
    distance: float,
    cube_distance: float,
) -> None:
    """A distant low-volume component cannot choose the integration origin."""
    cube = Mesh.from_file(str(binary_path / 'Cube.stl'))
    tiny_data = cube.data.copy()
    cube_offset = np.zeros(3)
    cube_offset[axis] = cube_distance
    cube.translate(cube_offset)
    sides = np.full(3, thickness)
    sides[axis] = length
    offset = np.zeros(3)
    offset[axis] = distance
    if distance < 0:
        offset[np.arange(3) != axis] = -thickness
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
    cube_cog = np.ones(3) + cube_offset
    expected_cog = (8 * cube_cog + tiny_volume * tiny_cog) / expected_volume
    expected_inertia = np.eye(3) * (16 / 3)
    expected_inertia += np.diag(
        tiny_volume / 12 * (np.sum(sizes**2) - sizes**2)
    )
    for volume, center in [(8, cube_cog), (tiny_volume, tiny_cog)]:
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
    cog_tolerance = max(1e-10, 2 * np.spacing(np.abs(expected_cog).max()))
    np.testing.assert_allclose(cog, expected_cog, rtol=0, atol=cog_tolerance)
    np.testing.assert_allclose(
        inertia, expected_inertia, rtol=1e-12, atol=1e-10
    )
    np.testing.assert_array_equal(compound.data, before)


@pytest.mark.parametrize('density', [None, 0.7, 2.3])
def test_point_touching_signed_shell_moments(
    binary_path: Path, density: float | None
) -> None:
    """Signed moments survive cancellation in two point-touching shells."""
    original = Mesh.from_file(str(binary_path / 'Cube.stl'))
    parts = [original.data.copy() for _ in range(3)]
    parts[1]['vectors'] = parts[1]['vectors'][:, ::-1] + 2
    parts[2]['vectors'] += 10
    signed = Mesh(np.concatenate(parts))
    assert signed.check(exact=True)
    before = signed.data.copy()
    if density is None:
        volume, cog, inertia = signed.get_mass_properties()
        scale = 1.0
    else:
        volume, mass, cog, inertia = signed.get_mass_properties_with_density(
            density
        )
        np.testing.assert_allclose(mass, 8 * density)
        scale = density
    expected_cog = np.full(3, 9.0)
    expected_inertia = np.eye(3) * (16 / 3)
    for sign, center in [(1, 1), (-1, 3), (1, 11)]:
        shift = np.full(3, center) - expected_cog
        expected_inertia += (
            sign * 8 * (np.eye(3) * (shift @ shift) - np.outer(shift, shift))
        )
    np.testing.assert_allclose(volume, 8)
    np.testing.assert_allclose(cog, expected_cog)
    np.testing.assert_allclose(inertia, expected_inertia * scale, atol=1e-10)
    np.testing.assert_array_equal(signed.data, before)


def test_zero_signed_volume_retains_first_and_second_moments(
    binary_path: Path,
) -> None:
    """A closed signed surface can have zero volume and nonzero moments."""
    # Two tetrahedra share a base and have equal heights, but their apexes
    # differ. The boundary has zero signed volume without zero moments.
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 1]]
    )
    faces = [[0, 1, 3], [1, 2, 3], [2, 0, 3], [1, 0, 4], [2, 1, 4], [0, 2, 4]]
    data = np.zeros(len(faces), dtype=Mesh.dtype)
    data['vectors'] = vertices[faces]
    cube = Mesh.from_file(str(binary_path / 'Cube.stl'))
    cube.translate([10, 10, 10])
    signed = Mesh(np.concatenate([data, cube.data]))
    assert signed.check(exact=True)
    first = np.full(3, 8 * 11.0)
    second = np.full((3, 3), 8 * 11.0**2)
    second += np.eye(3) * (8 / 3)
    # Independent simplex integrals: E[x] = sum(vertices)/4 and
    # E[xx^T] = (sum(vv^T) + sum(v)sum(v)^T)/20.
    for sign, apex in [(1, 3), (-1, 4)]:
        tetrahedron = vertices[[0, 1, 2, apex]]
        total = tetrahedron.sum(axis=0)
        first += sign / 6 * total / 4
        second += (
            sign
            / 6
            * (tetrahedron.T @ tetrahedron + np.outer(total, total))
            / 20
        )
    expected_cog = first / 8
    central = second - 8 * np.outer(expected_cog, expected_cog)
    expected_inertia = np.trace(central) * np.eye(3) - central
    volume, cog, inertia = signed.get_mass_properties()
    np.testing.assert_allclose(volume, 8)
    np.testing.assert_allclose(cog, expected_cog, atol=1e-12)
    np.testing.assert_allclose(inertia, expected_inertia, atol=1e-10)


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


@pytest.mark.parametrize('density', [None, 0.7, 2.3])
@pytest.mark.parametrize('axis', [0, 1, 2])
def test_mass_properties_extreme_edge_connected_appendage(
    density: float | None, axis: int
) -> None:
    """A conforming thin appendage preserves every central tensor entry."""
    length, width = float(np.float32(1e15)), float(np.float32(3.162e-14))
    vertices = np.array(
        [
            [-length, 0, 0],
            [-length, 0, width],
            [-length, width, 0],
            [-length, width, width],
            [0, 0, 0],
            [0, 0, width],
            [0, 0, 2],
            [0, width, 0],
            [0, width, width],
            [0, 2, 0],
            [0, 2, 2],
            [2, 0, 0],
            [2, 0, 2],
            [2, 2, 0],
            [2, 2, 2],
        ]
    )
    faces = [
        [14, 13, 9],
        [9, 10, 14],
        [12, 11, 13],
        [13, 14, 12],
        [6, 5, 11],
        [5, 4, 11],
        [11, 12, 6],
        [14, 10, 6],
        [6, 12, 14],
        [4, 7, 11],
        [7, 9, 11],
        [9, 13, 11],
        [10, 5, 6],
        [10, 8, 5],
        [10, 7, 8],
        [10, 9, 7],
        [3, 2, 0],
        [0, 1, 3],
        [8, 7, 2],
        [2, 3, 8],
        [1, 0, 4],
        [4, 5, 1],
        [8, 3, 1],
        [1, 5, 8],
        [4, 0, 2],
        [2, 7, 4],
    ]
    data = np.zeros(len(faces), dtype=Mesh.dtype)
    data['vectors'] = np.roll(vertices[faces], axis, axis=2)
    joined = Mesh(data)
    before = joined.data.copy()
    assert joined.check(exact=True)
    assert len(np.unique(joined._get_mass_properties_components()[0])) == 1
    tiny_volume = length * width**2
    cube_center = np.ones(3)
    tiny_center = np.array([-length / 2, width / 2, width / 2])
    expected_volume = 8 + tiny_volume
    expected_cog = (
        8 * cube_center + tiny_volume * tiny_center
    ) / expected_volume
    expected_inertia = np.eye(3) * (16 / 3)
    expected_inertia += (
        tiny_volume
        / 12
        * np.diag([2 * width**2, length**2 + width**2, length**2 + width**2])
    )
    for volume, center in [(8, cube_center), (tiny_volume, tiny_center)]:
        offset = center - expected_cog
        expected_inertia += volume * (
            np.eye(3) * (offset @ offset) - np.outer(offset, offset)
        )
    if density is None:
        volume, cog, inertia = joined.get_mass_properties()
        scale = 1.0
    else:
        volume, mass, cog, inertia = joined.get_mass_properties_with_density(
            density
        )
        np.testing.assert_allclose(mass, expected_volume * density)
        scale = density
    np.testing.assert_allclose(volume, expected_volume, rtol=1e-12)
    np.testing.assert_allclose(
        cog, np.roll(expected_cog, axis), rtol=1e-12, atol=1e-10
    )
    np.testing.assert_allclose(
        inertia,
        np.roll(expected_inertia, (axis, axis), axis=(0, 1)) * scale,
        rtol=1e-12,
        atol=1e-10,
    )
    np.testing.assert_array_equal(joined.data, before)


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


@pytest.mark.parametrize('density', [None, 0.7, 2.3])
@pytest.mark.parametrize('axis', [0, 1, 2])
def test_mass_properties_edge_connected_broad_plate(
    density: float | None, axis: int
) -> None:
    """A negligible broad plate keeps the integration frame near the body."""
    length = float(np.float32(1e15))
    end = float(np.float32(1.01e15))
    width = float(np.float32(1e13))
    thickness = float(np.float32(1e-28))
    vertices = np.array(
        [
            [0, 0, 0],
            [0, 0, thickness],
            [0, 0, 2],
            [0, 1, 0],
            [0, 1, thickness],
            [0, 1, 2],
            [0, 2, 0],
            [0, 2, thickness],
            [0, 2, 2],
            [2, 0, 0],
            [2, 0, thickness],
            [2, 0, 2],
            [2, 1, 0],
            [2, 1, thickness],
            [2, 1, 2],
            [2, 2, 0],
            [2, 2, thickness],
            [2, 2, 2],
            [length, 0, 0],
            [length, 0, thickness],
            [length, 1, 0],
            [length, 1, thickness],
            [length, 2, 0],
            [length, 2, thickness],
            [length, width, 0],
            [length, width, thickness],
            [end, 0, 0],
            [end, 0, thickness],
            [end, 1, 0],
            [end, 1, thickness],
            [end, 2, 0],
            [end, 2, thickness],
            [end, width, 0],
            [end, width, thickness],
        ]
    )
    faces = [
        [1, 4, 3],
        [1, 3, 0],
        [9, 10, 1],
        [9, 1, 0],
        [3, 12, 9],
        [3, 9, 0],
        [2, 5, 4],
        [2, 4, 1],
        [10, 13, 14],
        [10, 14, 11],
        [10, 11, 2],
        [10, 2, 1],
        [2, 11, 14],
        [2, 14, 5],
        [4, 7, 6],
        [4, 6, 3],
        [12, 15, 16],
        [12, 16, 13],
        [6, 7, 16],
        [6, 16, 15],
        [6, 15, 12],
        [6, 12, 3],
        [5, 8, 7],
        [5, 7, 4],
        [13, 16, 17],
        [13, 17, 14],
        [7, 8, 17],
        [7, 17, 16],
        [5, 14, 17],
        [5, 17, 8],
        [18, 19, 10],
        [18, 10, 9],
        [12, 13, 21],
        [12, 21, 20],
        [12, 20, 18],
        [12, 18, 9],
        [10, 19, 21],
        [10, 21, 13],
        [26, 28, 29],
        [26, 29, 27],
        [26, 27, 19],
        [26, 19, 18],
        [20, 28, 26],
        [20, 26, 18],
        [19, 27, 29],
        [19, 29, 21],
        [21, 23, 22],
        [21, 22, 20],
        [28, 30, 31],
        [28, 31, 29],
        [22, 30, 28],
        [22, 28, 20],
        [21, 29, 31],
        [21, 31, 23],
        [23, 25, 24],
        [23, 24, 22],
        [30, 32, 33],
        [30, 33, 31],
        [24, 25, 33],
        [24, 33, 32],
        [24, 32, 30],
        [24, 30, 22],
        [23, 31, 33],
        [23, 33, 25],
    ]
    data = np.zeros(len(faces), dtype=Mesh.dtype)
    data['vectors'] = np.roll(vertices[faces], axis, axis=2)
    joined = Mesh(data, calculate_normals=False)
    before = joined.data.copy()
    assert joined.check(exact=True)
    assert len(np.unique(joined._get_mass_properties_components()[0])) == 1
    sides = np.array(
        [
            [2, 2, 2],
            [length - 2, 1, thickness],
            [end - length, width, thickness],
        ]
    )
    centers = np.array(
        [
            [1, 1, 1],
            [(length + 2) / 2, 0.5, thickness / 2],
            [(end + length) / 2, width / 2, thickness / 2],
        ]
    )
    volumes = sides.prod(axis=1)
    expected_volume = volumes.sum()
    expected_cog = (volumes[:, None] * centers).sum(axis=0) / expected_volume
    expected_inertia = np.zeros((3, 3))
    for component_volume, dimensions, center in zip(
        volumes, sides, centers, strict=True
    ):
        offset = center - expected_cog
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            expected_inertia[i, i] += component_volume * (
                (dimensions[j] ** 2 + dimensions[k] ** 2) / 12
                + offset[j] ** 2
                + offset[k] ** 2
            )
            for j in range(3):
                if i != j:
                    expected_inertia[i, j] -= (
                        component_volume * offset[i] * offset[j]
                    )
    if density is None:
        volume, cog, inertia = joined.get_mass_properties()
        scale = 1.0
    else:
        volume, mass, cog, inertia = joined.get_mass_properties_with_density(
            density
        )
        np.testing.assert_allclose(mass, expected_volume * density, rtol=1e-10)
        scale = density
    np.testing.assert_allclose(volume, expected_volume, rtol=1e-10)
    np.testing.assert_allclose(
        cog, np.roll(expected_cog, axis), rtol=1e-10, atol=1e-8
    )
    np.testing.assert_allclose(
        inertia,
        np.roll(expected_inertia, (axis, axis), axis=(0, 1)) * scale,
        rtol=1e-10,
        atol=1e-8,
    )
    np.testing.assert_array_equal(joined.data, before)
