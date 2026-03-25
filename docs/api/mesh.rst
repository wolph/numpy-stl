stl.Mesh
========

.. autoclass:: stl.Mesh
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: dtype

   .. autoattribute:: dtype
      :annotation: = numpy.dtype([('normals', '<f4', (3,)), ('vectors', '<f4', (3, 3)), ('attr', '<u2', (1,))])

      The structured NumPy dtype used for mesh data storage.
      Each record contains: normals (3x float32), vectors
      (3x3 float32), and attr (1x uint16).
