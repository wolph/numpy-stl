import numpy

from stl.mesh import Mesh


def test_units_1d():
    data = numpy.zeros(1, dtype=Mesh.dtype)
    data['vectors'][0] = numpy.array([[0, 0, 0],
                                      [1, 0, 0],
                                      [2, 0, 0]])

    mesh = Mesh(data, remove_empty_areas=False)
    mesh.update_units()

    assert mesh.areas == 0
    assert (mesh.normals == [0, 0, 0]).all()
    assert (mesh.units == [0, 0, 0]).all()


def test_units_2d():
    data = numpy.zeros(2, dtype=Mesh.dtype)
    data['vectors'][0] = numpy.array([[0, 0, 0],
                                      [1, 0, 0],
                                      [0, 1, 0]])
    data['vectors'][1] = numpy.array([[1, 0, 0],
                                      [0, 1, 0],
                                      [1, 1, 0]])

    mesh = Mesh(data, remove_empty_areas=False)
    mesh.update_units()

    assert (mesh.areas == [1, 1]).all()
    assert (mesh.normals == [[0, 0, 1.],
                             [0, 0, -1.]]).all()

    assert (mesh.units == [[0, 0, 0.5],
                           [0, 0, -0.5]]).all()


def test_units_3d():
    data = numpy.zeros(1, dtype=Mesh.dtype)
    data['vectors'][0] = numpy.array([[0, 0, 0],
                                      [1, 0, 0],
                                      [0, 1, 1.]])

    mesh = Mesh(data, remove_empty_areas=False)
    mesh.update_units()

    assert mesh.areas == 2 ** .5
    assert (mesh.normals == [0, -1, 1.]).all()

    units = mesh.units[0]
    assert units[0] == 0
    # Due to floating point errors
    assert (units[1] + .25 * 2 ** .5) < 0.0001
    assert (units[2] - .25 * 2 ** .5) < 0.0001

