Mesh Operations
===============

Rotation
--------

Rotate around an arbitrary axis:

.. code-block:: python

   import math
   from stl import mesh

   m = mesh.Mesh.from_file('model.stl')

   # Rotate 90 degrees around the Y axis
   m.rotate([0, 0.5, 0], math.radians(90))

   # Rotate around an axis passing through a specific point
   m.rotate([0, 0, 1], math.radians(45), point=[1, 0, 0])

Translation
-----------

Translate (move) a mesh:

.. code-block:: python

   from stl import mesh

   m = mesh.Mesh.from_file('model.stl')
   m.translate([10, 0, 5])

You can also modify coordinates directly:

.. code-block:: python

   m.x += 10  # shift all X coordinates
   m.y += 5   # shift all Y coordinates

Transformation (4x4 Matrix)
----------------------------

Apply a full 4x4 transformation matrix:

.. code-block:: python

   import numpy as np
   from stl import mesh

   m = mesh.Mesh.from_file('model.stl')

   # Rotate 90 degrees around Z and translate by [10, 0, 5]
   matrix = np.eye(4)
   matrix[:3, :3] = [
       [0.0, -1.0, 0.0],
       [1.0, 0.0, 0.0],
       [0.0, 0.0, 1.0],
   ]
   matrix[:3, 3] = [10.0, 0.0, 5.0]
   m.transform(matrix)

Combining Meshes
----------------

Concatenate mesh data arrays to combine meshes:

.. code-block:: python

   import numpy as np
   from stl import mesh

   m1 = mesh.Mesh.from_file('part1.stl')
   m2 = mesh.Mesh.from_file('part2.stl')

   combined = mesh.Mesh(np.concatenate([m1.data, m2.data]))
   combined.save('combined.stl')

Removing Duplicates
-------------------

Remove duplicate or degenerate triangles:

.. code-block:: python

   from stl import mesh, base

   m = mesh.Mesh.from_file('model.stl')

   # Remove duplicate triangles (keep one copy)
   data = mesh.Mesh.remove_duplicate_polygons(
       m.data,
       base.RemoveDuplicates.SINGLE,
   )

   # Remove zero-area triangles
   data = mesh.Mesh.remove_empty_areas(data)
   cleaned = mesh.Mesh(data)
