import math
import numpy

from stl.mesh import Mesh


def test_rotation():
    # Create 6 faces of a cube
    data = numpy.zeros(6, dtype=Mesh.dtype)

    # Top of the cube
    data['vectors'][0] = numpy.array([[0, 1, 1],
                                      [1, 0, 1],
                                      [0, 0, 1]])
    data['vectors'][1] = numpy.array([[1, 0, 1],
                                      [0, 1, 1],
                                      [1, 1, 1]])
    # Right face
    data['vectors'][2] = numpy.array([[1, 0, 0],
                                      [1, 0, 1],
                                      [1, 1, 0]])
    data['vectors'][3] = numpy.array([[1, 1, 1],
                                      [1, 0, 1],
                                      [1, 1, 0]])
    # Left face
    data['vectors'][4] = numpy.array([[0, 0, 0],
                                      [1, 0, 0],
                                      [1, 0, 1]])
    data['vectors'][5] = numpy.array([[0, 0, 0],
                                      [0, 0, 1],
                                      [1, 0, 1]])

    mesh = Mesh(data, remove_empty_areas=False)

    # Since the cube faces are from 0 to 1 we can move it to the middle by
    # substracting .5
    data['vectors'] -= .5

    # Rotate 90 degrees over the X axis followed by the Y axis followed by the
    # X axis
    mesh.rotate([0.5, 0.0, 0.0], math.radians(90))
    mesh.rotate([0.0, 0.5, 0.0], math.radians(90))
    mesh.rotate([0.5, 0.0, 0.0], math.radians(90))

    # Since the cube faces are from 0 to 1 we can move it to the middle by
    # substracting .5
    data['vectors'] += .5

    assert (mesh.vectors == numpy.array([
        [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 0, 0], [1, 1, 0]],
        [[0, 1, 1], [0, 1, 0], [1, 1, 1]],
        [[1, 1, 0], [0, 1, 0], [1, 1, 1]],
        [[0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 1], [0, 0, 0], [0, 1, 0]],
    ])).all()


def test_rotation_over_point():
    # Create a single face
    data = numpy.zeros(1, dtype=Mesh.dtype)

    data['vectors'][0] = numpy.array([[1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]])

    mesh = Mesh(data, remove_empty_areas=False)

    mesh.rotate([1, 0, 0], math.radians(180), point=[1, 2, 3])
    assert (mesh.vectors == numpy.array([[1, -4, -6],
                                         [0, -5, -6],
                                         [0, -4, -7]])).all()


def test_no_rotation():
    # Create a single face
    data = numpy.zeros(1, dtype=Mesh.dtype)

    data['vectors'][0] = numpy.array([[0, 1, 1],
                                      [1, 0, 1],
                                      [0, 0, 1]])

    mesh = Mesh(data, remove_empty_areas=False)

    # Rotate by 0 degrees
    mesh.rotate([0.5, 0.0, 0.0], math.radians(0))
    assert (mesh.vectors == numpy.array([
        [[0, 1, 1], [1, 0, 1], [0, 0, 1]]])).all()

    # Use a zero rotation matrix
    mesh.rotate([0.0, 0.0, 0.0], math.radians(90))
    assert (mesh.vectors == numpy.array([
        [[0, 1, 1], [1, 0, 1], [0, 0, 1]]])).all()


def test_no_translation():
    # Create a single face
    data = numpy.zeros(1, dtype=Mesh.dtype)
    data['vectors'][0] = numpy.array([[0, 1, 1],
                                      [1, 0, 1],
                                      [0, 0, 1]])

    mesh = Mesh(data, remove_empty_areas=False)
    assert (mesh.vectors == numpy.array([
        [[0, 1, 1], [1, 0, 1], [0, 0, 1]]])).all()

    # Translate mesh with a zero vector
    mesh.translate([0.0, 0.0, 0.0])
    assert (mesh.vectors == numpy.array([
        [[0, 1, 1], [1, 0, 1], [0, 0, 1]]])).all()


def test_translation():
    # Create a single face
    data = numpy.zeros(1, dtype=Mesh.dtype)
    data['vectors'][0] = numpy.array([[0, 1, 1],
                                      [1, 0, 1],
                                      [0, 0, 1]])

    mesh = Mesh(data, remove_empty_areas=False)
    assert (mesh.vectors == numpy.array([
        [[0, 1, 1], [1, 0, 1], [0, 0, 1]]])).all()

    # Translate mesh with vector [1, 2, 3]
    mesh.translate([1.0, 2.0, 3.0])
    assert (mesh.vectors == numpy.array([
        [[1, 3, 4], [2, 2, 4], [1, 2, 4]]])).all()


def test_no_transformation():
    # Create a single face
    data = numpy.zeros(1, dtype=Mesh.dtype)
    data['vectors'][0] = numpy.array([[0, 1, 1],
                                      [1, 0, 1],
                                      [0, 0, 1]])

    mesh = Mesh(data, remove_empty_areas=False)
    assert (mesh.vectors == numpy.array([
        [[0, 1, 1], [1, 0, 1], [0, 0, 1]]])).all()

    # Transform mesh with identity matrix
    mesh.transform(numpy.eye(4))
    assert (mesh.vectors == numpy.array([
        [[0, 1, 1], [1, 0, 1], [0, 0, 1]]])).all()
    assert numpy.all(mesh.areas == 0.5)


def test_transformation():
    # Create a single face
    data = numpy.zeros(1, dtype=Mesh.dtype)
    data['vectors'][0] = numpy.array([[0, 1, 1],
                                      [1, 0, 1],
                                      [0, 0, 1]])

    mesh = Mesh(data, remove_empty_areas=False)
    assert (mesh.vectors == numpy.array([
        [[0, 1, 1], [1, 0, 1], [0, 0, 1]]])).all()

    # Transform mesh with identity matrix
    tr = numpy.zeros((4, 4))
    tr[0:3, 0:3] = Mesh.rotation_matrix([0, 0, 1], 0.5 * numpy.pi)
    tr[0:3, 3] = [1, 2, 3]
    mesh.transform(tr)
    assert (mesh.vectors == numpy.array([
        [[0, 2, 4], [1, 3, 4], [1, 2, 4]]])).all()
    assert numpy.all(mesh.areas == 0.5)
