numpy-stl
==============================================================================


.. image:: https://travis-ci.org/WoLpH/numpy-stl.svg?branch=master
    :alt: numpy-stl test status 
    :target: https://travis-ci.org/WoLpH/numpy-stl

.. image:: https://badge.fury.io/py/numpy-stl.svg
    :alt: numpy-stl Pypi version 
    :target: https://pypi.python.org/pypi/numpy-stl

.. image:: https://coveralls.io/repos/WoLpH/numpy-stl/badge.svg?branch=master
    :alt: numpy-stl code coverage 
    :target: https://coveralls.io/r/WoLpH/numpy-stl?branch=master

.. image:: https://img.shields.io/pypi/pyversions/numpy-stl.svg

Simple library to make working with STL files (and 3D objects in general) fast
and easy.

Due to all operations heavily relying on `numpy` this is one of the fastest
STL editing libraries for Python available.

Links
-----

 - The source: https://github.com/WoLpH/numpy-stl
 - Project page: https://pypi.python.org/pypi/numpy-stl
 - Reporting bugs: https://github.com/WoLpH/numpy-stl/issues
 - Documentation: http://numpy-stl.readthedocs.org/en/latest/
 - My blog: https://wol.ph/

Requirements for installing:
------------------------------------------------------------------------------

 - `numpy`_ any recent version
 - `python-utils`_ version 1.6 or greater

Installation:
------------------------------------------------------------------------------

`pip install numpy-stl`

Initial usage:
------------------------------------------------------------------------------

 - `stl2bin your_ascii_stl_file.stl new_binary_stl_file.stl`
 - `stl2ascii your_binary_stl_file.stl new_ascii_stl_file.stl`
 - `stl your_ascii_stl_file.stl new_binary_stl_file.stl`

Contributing:
------------------------------------------------------------------------------

Contributions are always welcome. Please view the guidelines to get started:
https://github.com/WoLpH/numpy-stl/blob/develop/CONTRIBUTING.rst

Quickstart
------------------------------------------------------------------------------

.. code-block:: python

    import numpy
    from stl import mesh

    # Using an existing stl file:
    your_mesh = mesh.Mesh.from_file('some_file.stl')

    # Or creating a new mesh (make sure not to overwrite the `mesh` import by
    # naming it `mesh`):
    VERTICE_COUNT = 100
    data = numpy.zeros(VERTICE_COUNT, dtype=mesh.Mesh.dtype)
    your_mesh = mesh.Mesh(data, remove_empty_areas=False)

    # The mesh normals (calculated automatically)
    your_mesh.normals
    # The mesh vectors
    your_mesh.v0, your_mesh.v1, your_mesh.v2
    # Accessing individual points (concatenation of v0, v1 and v2 in triplets)
    assert (your_mesh.points[0][0:3] == your_mesh.v0[0]).all()
    assert (your_mesh.points[0][3:6] == your_mesh.v1[0]).all()
    assert (your_mesh.points[0][6:9] == your_mesh.v2[0]).all()
    assert (your_mesh.points[1][0:3] == your_mesh.v0[1]).all()

    your_mesh.save('new_stl_file.stl')

Plotting using `matplotlib`_ is equally easy:
------------------------------------------------------------------------------

.. code-block:: python

    from stl import mesh
    from mpl_toolkits import mplot3d
    from matplotlib import pyplot

    # Create a new plot
    figure = pyplot.figure()
    axes = mplot3d.Axes3D(figure)

    # Load the STL files and add the vectors to the plot
    your_mesh = mesh.Mesh.from_file('tests/stl_binary/HalfDonut.stl')
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

    # Auto scale to the mesh size
    scale = your_mesh.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)

    # Show the plot to the screen
    pyplot.show()

.. _numpy: http://numpy.org/
.. _matplotlib: http://matplotlib.org/
.. _python-utils: https://github.com/WoLpH/python-utils

Modifying Mesh objects
------------------------------------------------------------------------------

.. code-block:: python

    from stl import mesh
    import math
    import numpy

    # Create 3 faces of a cube
    data = numpy.zeros(6, dtype=mesh.Mesh.dtype)

    # Top of the cube
    data['vectors'][0] = numpy.array([[0, 1, 1],
                                      [1, 0, 1],
                                      [0, 0, 1]])
    data['vectors'][1] = numpy.array([[1, 0, 1],
                                      [0, 1, 1],
                                      [1, 1, 1]])
    # Front face
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

    # Since the cube faces are from 0 to 1 we can move it to the middle by
    # substracting .5
    data['vectors'] -= .5

    # Generate 4 different meshes so we can rotate them later
    meshes = [mesh.Mesh(data.copy()) for _ in range(4)]

    # Rotate 90 degrees over the Y axis
    meshes[0].rotate([0.0, 0.5, 0.0], math.radians(90))

    # Translate 2 points over the X axis
    meshes[1].x += 2

    # Rotate 90 degrees over the X axis
    meshes[2].rotate([0.5, 0.0, 0.0], math.radians(90))
    # Translate 2 points over the X and Y points
    meshes[2].x += 2
    meshes[2].y += 2

    # Rotate 90 degrees over the X and Y axis
    meshes[3].rotate([0.5, 0.0, 0.0], math.radians(90))
    meshes[3].rotate([0.0, 0.5, 0.0], math.radians(90))
    # Translate 2 points over the Y axis
    meshes[3].y += 2


    # Optionally render the rotated cube faces
    from matplotlib import pyplot
    from mpl_toolkits import mplot3d

    # Create a new plot
    figure = pyplot.figure()
    axes = mplot3d.Axes3D(figure)

    # Render the cube faces
    for m in meshes:
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(m.vectors))

    # Auto scale to the mesh size
    scale = numpy.concatenate([m.points for m in meshes]).flatten()
    axes.auto_scale_xyz(scale, scale, scale)

    # Show the plot to the screen
    pyplot.show()

Extending Mesh objects
------------------------------------------------------------------------------

.. code-block:: python

    from stl import mesh
    import math
    import numpy

    # Create 3 faces of a cube
    data = numpy.zeros(6, dtype=mesh.Mesh.dtype)

    # Top of the cube
    data['vectors'][0] = numpy.array([[0, 1, 1],
                                      [1, 0, 1],
                                      [0, 0, 1]])
    data['vectors'][1] = numpy.array([[1, 0, 1],
                                      [0, 1, 1],
                                      [1, 1, 1]])
    # Front face
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

    # Since the cube faces are from 0 to 1 we can move it to the middle by
    # substracting .5
    data['vectors'] -= .5

    cube_back = mesh.Mesh(data.copy())
    cube_front = mesh.Mesh(data.copy())

    # Rotate 90 degrees over the X axis followed by the Y axis followed by the
    # X axis
    cube_back.rotate([0.5, 0.0, 0.0], math.radians(90))
    cube_back.rotate([0.0, 0.5, 0.0], math.radians(90))
    cube_back.rotate([0.5, 0.0, 0.0], math.radians(90))

    cube = mesh.Mesh(numpy.concatenate([
        cube_back.data.copy(),
        cube_front.data.copy(),
    ]))

    # Optionally render the rotated cube faces
    from matplotlib import pyplot
    from mpl_toolkits import mplot3d

    # Create a new plot
    figure = pyplot.figure()
    axes = mplot3d.Axes3D(figure)

    # Render the cube
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(cube.vectors))

    # Auto scale to the mesh size
    scale = cube_back.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)

    # Show the plot to the screen
    pyplot.show()

Creating Mesh objects from a list of vertices and faces
------------------------------------------------------------------------------

.. code-block:: python

    import numpy as np
    from stl import mesh

    # Define the 8 vertices of the cube
    vertices = np.array([\
        [-1, -1, -1],
        [+1, -1, -1],
        [+1, +1, -1],
        [-1, +1, -1],
        [-1, -1, +1],
        [+1, -1, +1],
        [+1, +1, +1],
        [-1, +1, +1]])
    # Define the 12 triangles composing the cube
    faces = np.array([\
        [0,3,1],
        [1,3,2],
        [0,4,7],
        [0,7,3],
        [4,5,6],
        [4,6,7],
        [5,1,2],
        [5,2,6],
        [2,3,6],
        [3,7,6],
        [0,1,5],
        [0,5,4]])

    # Create the mesh
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[f[j],:]

    # Write the mesh to file "cube.stl"
    cube.save('cube.stl')


Evaluating Mesh properties (Volume, Center of gravity, Inertia)
------------------------------------------------------------------------------

.. code-block:: python

    import numpy as np
    from stl import mesh

    # Using an existing closed stl file:
    your_mesh = mesh.Mesh.from_file('some_file.stl')

    volume, cog, inertia = your_mesh.get_mass_properties()
    print("Volume                                  = {0}".format(volume))
    print("Position of the center of gravity (COG) = {0}".format(cog))
    print("Inertia matrix at expressed at the COG  = {0}".format(inertia[0,:]))
    print("                                          {0}".format(inertia[1,:]))
    print("                                          {0}".format(inertia[2,:]))

Combining multiple STL files
------------------------------------------------------------------------------

.. code-block:: python

    import math
    import stl
    from stl import mesh
    import numpy


    # find the max dimensions, so we can know the bounding box, getting the height,
    # width, length (because these are the step size)...
    def find_mins_maxs(obj):
        minx = obj.x.min()
        maxx = obj.x.max()
        miny = obj.y.min()
        maxy = obj.y.max()
        minz = obj.z.min()
        maxz = obj.z.max()
        return minx, maxx, miny, maxy, minz, maxz


    def translate(_solid, step, padding, multiplier, axis):
        if 'x' == axis:
            items = 0, 3, 6
        elif 'y' == axis:
            items = 1, 4, 7
        elif 'z' == axis:
            items = 2, 5, 8
        else:
            raise RuntimeError('Unknown axis %r, expected x, y or z' % axis)

        # _solid.points.shape == [:, ((x, y, z), (x, y, z), (x, y, z))]
        _solid.points[:, items] += (step * multiplier) + (padding * multiplier)


    def copy_obj(obj, dims, num_rows, num_cols, num_layers):
        w, l, h = dims
        copies = []
        for layer in range(num_layers):
            for row in range(num_rows):
                for col in range(num_cols):
                    # skip the position where original being copied is
                    if row == 0 and col == 0 and layer == 0:
                        continue
                    _copy = mesh.Mesh(obj.data.copy())
                    # pad the space between objects by 10% of the dimension being
                    # translated
                    if col != 0:
                        translate(_copy, w, w / 10., col, 'x')
                    if row != 0:
                        translate(_copy, l, l / 10., row, 'y')
                    if layer != 0:
                        translate(_copy, h, h / 10., layer, 'z')
                    copies.append(_copy)
        return copies

    # Using an existing stl file:
    main_body = mesh.Mesh.from_file('ball_and_socket_simplified_-_main_body.stl')

    # rotate along Y
    main_body.rotate([0.0, 0.5, 0.0], math.radians(90))

    minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(main_body)
    w1 = maxx - minx
    l1 = maxy - miny
    h1 = maxz - minz
    copies = copy_obj(main_body, (w1, l1, h1), 2, 2, 1)

    # I wanted to add another related STL to the final STL
    twist_lock = mesh.Mesh.from_file('ball_and_socket_simplified_-_twist_lock.stl')
    minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(twist_lock)
    w2 = maxx - minx
    l2 = maxy - miny
    h2 = maxz - minz
    translate(twist_lock, w1, w1 / 10., 3, 'x')
    copies2 = copy_obj(twist_lock, (w2, l2, h2), 2, 2, 1)
    combined = mesh.Mesh(numpy.concatenate([main_body.data, twist_lock.data] +
                                        [copy.data for copy in copies] +
                                        [copy.data for copy in copies2]))

    combined.save('combined.stl', mode=stl.Mode.ASCII)  # save as ASCII

