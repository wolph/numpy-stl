import numpy
import pytest

from stl import stl


tolerance = 1e-6


def close(a, b):
    return numpy.allclose(a, b, atol=tolerance)


def test_mass_properties_for_half_donut(binary_ascii_path, speedups):
    '''
    Checks the results of method get_mass_properties() on
    STL ASCII and binary files HalfDonut.stl
    One checks the results obtained with stl
    with the ones obtained with meshlab
    '''
    filename = binary_ascii_path.join('HalfDonut.stl')
    mesh = stl.StlMesh(str(filename), speedups=speedups)
    volume, cog, inertia = mesh.get_mass_properties()
    assert close([volume], [2.343149])
    assert close(cog, [1.500001, 0.209472, 1.500001])
    assert close(inertia, [[+1.390429, +0.000000, +0.000000],
                           [+0.000000, +2.701025, +0.000000],
                           [+0.000000, +0.000000, +1.390429]])


def test_mass_properties_for_moon(binary_ascii_path, speedups):
    '''
    Checks the results of method get_mass_properties() on
    STL ASCII and binary files Moon.stl
    One checks the results obtained with stl
    with the ones obtained with meshlab
    '''
    filename = binary_ascii_path.join('Moon.stl')
    mesh = stl.StlMesh(str(filename), speedups=speedups)
    volume, cog, inertia = mesh.get_mass_properties()
    assert close([volume], [0.888723])
    assert close(cog, [0.906913, 0.170731, 1.500001])
    assert close(inertia, [[+0.562097, -0.000457, +0.000000],
                           [-0.000457, +0.656851, +0.000000],
                           [+0.000000, +0.000000, +0.112465]])


@pytest.mark.parametrize('filename', ('Star.stl', 'StarWithEmptyHeader.stl'))
def test_mass_properties_for_star(binary_ascii_path, filename, speedups):
    '''
    Checks the results of method get_mass_properties() on
    STL ASCII and binary files Star.stl and
    STL binary file StarWithEmptyHeader.stl (with no header)
    One checks the results obtained with stl
    with the ones obtained with meshlab
    '''
    filename = binary_ascii_path.join(filename)
    if not filename.exists():
        pytest.skip('STL file does not exist')
    mesh = stl.StlMesh(str(filename), speedups=speedups)
    volume, cog, inertia = mesh.get_mass_properties()
    assert close([volume], [1.416599])
    assert close(cog, [1.299040, 0.170197, 1.499999])
    assert close(inertia, [[+0.509549, +0.000000, -0.000000],
                           [+0.000000, +0.991236, +0.000000],
                           [-0.000000, +0.000000, +0.509550]])


def test_mass_properties_for_half_donut_with_density(
        binary_ascii_path, speedups):
    '''
    Checks the results of method get_mass_properties_with_density() on
    STL ASCII and binary files HalfDonut.stl
    One checks the results obtained with stl
    with the ones obtained with meshlab
    '''
    filename = binary_ascii_path.join('HalfDonut.stl')
    mesh = stl.StlMesh(str(filename), speedups=speedups)
    volume, mass, cog, inertia = mesh.get_mass_properties_with_density(1.23)

    assert close([mass], [2.882083302268982])
    assert close([volume], [2.343149026234945])
    assert close(cog, [1.500001, 0.209472, 1.500001])
    print('inertia')
    numpy.set_printoptions(suppress=True)
    print(inertia)
    assert close(inertia, [[+1.71022851, +0.00000001, -0.00000011],
                           [+0.00000001, +3.32226227, +0.00000002],
                           [-0.00000011, +0.00000002, +1.71022859]])
