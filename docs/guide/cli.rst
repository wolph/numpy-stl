Command-Line Tools
==================

numpy-stl provides three CLI commands, installed automatically
with the package.

stl
---

Convert between ASCII and binary STL formats:

.. code-block:: bash

   # Auto-detect input, write binary output
   stl input.stl output.stl

   # Force ASCII output
   stl input.stl output.stl -a

   # Force binary output
   stl input.stl output.stl -b

stl2ascii
---------

Convert a binary STL file to ASCII format:

.. code-block:: bash

   stl2ascii input.stl output.stl

stl2bin
-------

Convert an ASCII STL file to binary format:

.. code-block:: bash

   stl2bin input.stl output.stl

Options
-------

All commands support ``-n`` to keep the normals stored in the input file
instead of recalculating them on output:

.. code-block:: bash

   stl input.stl output.stl -n

All commands also support ``-s`` to force the pure-Python reader/writer
and disable optional speedups:

.. code-block:: bash

   stl input.stl output.stl -s
