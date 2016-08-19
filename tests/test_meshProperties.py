import numpy

from stl import stl


tolerance = 1e-6


def test_mass_properties_for_half_donut(ascii_path, binary_path, speedups):
    """
    Checks the results of method get_mass_properties() on
    STL ASCII and binary files HalfDonut.stl
    One checks the results obtained with stl
    with the ones obtained with meshlab
    """
    for path in (ascii_path, binary_path):
        filename = path.join('HalfDonut.stl')
        mesh = stl.StlMesh(str(filename), speedups=speedups)
        volume, cog, inertia = mesh.get_mass_properties()
        assert(abs(volume - 2.343149) < tolerance)
        assert(numpy.allclose(cog,
               numpy.array([1.500001, 0.209472, 1.500001]),
               atol=tolerance))
        assert(numpy.allclose(inertia,
               numpy.array([[+1.390429, +0.000000, +0.000000],
                            [+0.000000, +2.701025, +0.000000],
                            [+0.000000, +0.000000, +1.390429]]),
               atol=tolerance))


def test_mass_properties_for_moon(ascii_path, binary_path, speedups):
    """
    Checks the results of method get_mass_properties() on
    STL ASCII and binary files Moon.stl
    One checks the results obtained with stl
    with the ones obtained with meshlab
    """
    for path in (ascii_path, binary_path):
        filename = path.join('Moon.stl')
        mesh = stl.StlMesh(str(filename), speedups=speedups)
        volume, cog, inertia = mesh.get_mass_properties()
        assert(abs(volume - 0.888723) < tolerance)
        assert(numpy.allclose(cog,
               numpy.array([0.906913, 0.170731, 1.500001]),
               atol=tolerance))
        assert(numpy.allclose(inertia,
               numpy.array([[+0.562097, -0.000457, +0.000000],
                            [-0.000457, +0.656851, +0.000000],
                            [+0.000000, +0.000000, +0.112465]]),
               atol=tolerance))


def test_mass_properties_for_star(ascii_path, binary_path, speedups):
    """
    Checks the results of method get_mass_properties() on
    STL ASCII and binary files Star.stl and
    STL binary file StarWithEmptyHeader.stl (with no header)
    One checks the results obtained with stl
    with the ones obtained with meshlab
    """
    for filename in (ascii_path.join('Star.stl'),
                     binary_path.join('Star.stl'),
                     binary_path.join('StarWithEmptyHeader.stl')):
        mesh = stl.StlMesh(str(filename), speedups=speedups)
        volume, cog, inertia = mesh.get_mass_properties()
        assert(abs(volume - 1.416599) < tolerance)
        assert(numpy.allclose(cog,
               numpy.array([1.299040, 0.170197, 1.499999]),
               atol=tolerance))
        assert(numpy.allclose(inertia,
               numpy.array([[+0.509549, +0.000000, -0.000000],
                            [+0.000000, +0.991236, +0.000000],
                            [-0.000000, +0.000000, +0.509550]]),
               atol=tolerance))
