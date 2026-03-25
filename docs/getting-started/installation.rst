Installation
============

Basic Install
-------------

.. code-block:: bash

   pip install numpy-stl

This installs numpy-stl with its required dependencies (NumPy and
python-utils).

Optional Speedups
-----------------

For faster ASCII STL parsing, install with the ``fast`` extra:

.. code-block:: bash

   pip install numpy-stl[fast]

This installs the ``speedups`` package, a Cython-compiled extension
that accelerates ASCII file I/O. The library works identically
without it -- pure Python is the default.

Requirements
------------

- Python 3.10 or later
- NumPy (1.x or 2.x)
- python-utils >= 3.4.5

Development Install
-------------------

See the `Contributing Guide <https://github.com/WoLpH/numpy-stl/blob/develop/CONTRIBUTING.md>`_
for development setup instructions.
