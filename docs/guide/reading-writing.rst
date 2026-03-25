Reading and Writing STL Files
=============================

numpy-stl supports both ASCII and binary STL formats. Format
detection is automatic when loading.

Loading a Single File
---------------------

.. code-block:: python

   from stl import mesh

   # Auto-detect format
   m = mesh.Mesh.from_file('model.stl')

   # Force a specific format
   m = mesh.Mesh.from_file('model.stl', mode=stl.Mode.ASCII)
   m = mesh.Mesh.from_file('model.stl', mode=stl.Mode.BINARY)

Loading from a File Handle
--------------------------

.. code-block:: python

   from stl import mesh

   with open('model.stl', 'rb') as fh:
       m = mesh.Mesh.from_file('model.stl', fh=fh)

Loading Multiple Solids (ASCII Only)
-------------------------------------

An ASCII STL file can contain multiple ``solid`` blocks.
Use :meth:`~stl.mesh.Mesh.from_multi_file` to load each as
a separate Mesh:

.. code-block:: python

   from stl import mesh

   for m in mesh.Mesh.from_multi_file('multi.stl'):
       print(f'Solid: {m.name}, {len(m)} triangles')

.. note::
   Multi-solid loading only works with ASCII STL files.
   Binary STL files always contain a single solid.

Combining Multiple Files
-------------------------

:meth:`~stl.mesh.Mesh.from_files` merges multiple STL files
into a single mesh:

.. code-block:: python

   from stl import mesh

   combined = mesh.Mesh.from_files(['part1.stl', 'part2.stl'])

Reading 3MF Files
-----------------

Experimental support for reading 3MF files (read-only):

.. code-block:: python

   import pathlib
   from stl import mesh

   for m in mesh.Mesh.from_3mf_file(pathlib.Path('model.3mf')):
       print(f'{len(m)} triangles')

Saving
------

.. code-block:: python

   import stl
   from stl import mesh

   m = mesh.Mesh.from_file('input.stl')

   # Binary (default, compact)
   m.save('output.stl')

   # ASCII (human-readable)
   m.save('output.stl', mode=stl.Mode.ASCII)

.. warning::
   The ``save`` method requires a binary file handle (``'wb'`` mode)
   even when saving ASCII format. If you pass a text-mode handle,
   a ``TypeError`` is raised.
