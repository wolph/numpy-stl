Quick Start
===========

This guide gets you productive with numpy-stl in under 5 minutes.

Loading an STL File
-------------------

.. code-block:: python

   from stl import mesh

   your_mesh = mesh.Mesh.from_file('model.stl')
   print(f'{len(your_mesh)} triangles')

Format detection (ASCII vs binary) is automatic.

Creating a Mesh from Scratch
-----------------------------

.. code-block:: python

   import numpy as np
   from stl import mesh

   # Create a single triangle
   data = np.zeros(1, dtype=mesh.Mesh.dtype)
   data['vectors'][0] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]

   triangle = mesh.Mesh(data)
   triangle.save('triangle.stl')

Inspecting Properties
---------------------

.. code-block:: python

   from stl import mesh

   m = mesh.Mesh.from_file('model.stl')

   # Vertex data
   print('First vertex of each triangle:', m.v0[:3])
   print('Bounding box:', m.min_, m.max_)
   print('Surface areas:', m.areas[:3])

Basic Transformations
---------------------

.. code-block:: python

   import math
   from stl import mesh

   m = mesh.Mesh.from_file('model.stl')

   # Rotate 90 degrees around the Z axis
   m.rotate([0, 0, 1], math.radians(90))

   # Translate by [10, 0, 0]
   m.translate([10, 0, 0])

   m.save('transformed.stl')

Saving in Different Formats
----------------------------

.. code-block:: python

   import stl
   from stl import mesh

   m = mesh.Mesh.from_file('model.stl')

   # Save as binary (default, smaller file)
   m.save('output.stl')

   # Save as ASCII (human-readable)
   m.save('output.stl', mode=stl.Mode.ASCII)

Next Steps
----------

- :doc:`../guide/reading-writing` -- Full I/O documentation
- :doc:`../guide/mesh-operations` -- Rotations, translations, combining meshes
- :doc:`../guide/properties` -- Mass properties, convexity, surface area
- :doc:`../guide/cli` -- Command-line tools
