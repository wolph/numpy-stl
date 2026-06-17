API Reference
=============

.. toctree::
   :maxdepth: 2

   mesh

The primary entry point is :class:`stl.Mesh`, which inherits
from :class:`~stl.stl.BaseStl` and :class:`~stl.base.BaseMesh`.

Enumerations
------------

.. autoclass:: stl.Mode
   :members:
   :undoc-members:

.. autoclass:: stl.Dimension
   :members:
   :undoc-members:

.. autoclass:: stl.RemoveDuplicates
   :members:
   :undoc-members:

Constants
---------

.. data:: stl.HEADER_SIZE

   Size of the binary STL header in bytes (80).

.. data:: stl.COUNT_SIZE

   Size of the triangle count field in bytes (4).

.. data:: stl.MAX_COUNT

   Maximum number of triangles in a binary STL file (100,000,000).
