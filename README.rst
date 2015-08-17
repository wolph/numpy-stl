stl-numpy
==============================================================================

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
 - My blog: http://w.wol.ph/

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
    scale = your_mesh.points.flatten(-1)
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

    # Since the cube faces are from 0 to 1 we can move it to the middle by 
    # substracting .5
    data['vectors'] -= .5

    # Generate 4 different meshes so we can rotate them later
    meshes = []
    for _ in range(4):
        meshes.append(mesh.Mesh(data.copy()))

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
    scale = numpy.concatenate([m.points for m in meshes]).flatten(-1)
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
    scale = cube_back.points.flatten(-1)
    axes.auto_scale_xyz(scale, scale, scale)

    # Show the plot to the screen
    pyplot.show()
