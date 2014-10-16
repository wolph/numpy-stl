stl-numpy
------------------------------------------------------------------------------

Simple library to make working with STL files (and 3D objects in general) fast
and easy.

Due to all operations heavily relying on `numpy` this is one of the fastest
STL editing libraries for Python available.

Requirements for installing:
==============================================================================

 - `numpy_` any recent version
 - `python-utils_` version 1.6 or greater

Installation:
==============================================================================

`pip install numpy-stl`

Initial usage:
==============================================================================

 - `stl2bin your_ascii_stl_file.stl new_binary_stl_file.stl`
 - `stl2ascii your_binary_stl_file.stl new_ascii_stl_file.stl`
 - `stl your_ascii_stl_file.stl new_binary_stl_file.stl`

Quickstart
==============================================================================

.. code-block:: python

   from stl import stl
   
   mesh = stl.StlMesh('some_file.stl')
   # The mesh normals (calculated automatically)
   mesh.normals
   # The mesh vectors
   mesh.v0, mesh.v1, mesh.v2
   # Accessing individual points (concatenation of v0, v1 and v2 in triplets)
   mesh.points[0] == mesh.v0[0]
   mesh.points[1] == mesh.v1[0]
   mesh.points[2] == mesh.v2[0]
   mesh.points[3] == mesh.v0[1]
   
   mesh.save('new_stl_file.stl')
   

.. _numpy: http://numpy.org/
.. _python-utils: https://github.com/WoLpH/python-utils 

