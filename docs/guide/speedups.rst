Performance and Speedups
========================

numpy-stl is fast by default because all mesh operations use
vectorized NumPy operations. For even faster ASCII file I/O,
optional Cython-compiled speedups are available.

How Speedups Work
-----------------

The ``speedups`` package provides a C-compiled replacement for
the ASCII STL reader and writer. When installed, numpy-stl
auto-detects and uses it transparently.

Installing Speedups
-------------------

.. code-block:: bash

   pip install numpy-stl[fast]

This installs the ``speedups`` package as an extra dependency.

Checking Speedups Status
------------------------

.. code-block:: python

   from stl._compat import has_speedups

   print(has_speedups())  # True if speedups installed

When Speedups Are Disabled
--------------------------

Speedups are automatically disabled for non-seekable streams
(e.g., ``stdin``, ``StringIO``). The library falls back to
pure Python in these cases.

For the CLI tools, pass ``-s`` / ``--disable-speedups`` to force the
pure-Python implementation even when the optional ``speedups`` package
is installed.

.. note::
   When speedups are enabled, STL solid names are automatically
   converted to lowercase. This is a known limitation of the
   Cython implementation.
