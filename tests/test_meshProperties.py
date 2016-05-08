import numpy

from stl import stl


tolerance = 1e-6


def test_mass_properties_for_HalfDonut():
    # One checks the results obtained with stl
    # with the ones obtained with meshlab
    ascii_file = 'tests/stl_ascii/HalfDonut.stl'
    mesh = stl.StlMesh(ascii_file)
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


def test_mass_properties_for_Moon():
    # One checks the results obtained with stl
    # with the ones obtained with meshlab
    ascii_file = 'tests/stl_ascii/Moon.stl'
    mesh = stl.StlMesh(ascii_file)
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


def test_mass_properties_for_Star():
    # One checks the results obtained with stl
    # with the ones obtained with meshlab
    ascii_file = 'tests/stl_ascii/Star.stl'
    mesh = stl.StlMesh(ascii_file)
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
