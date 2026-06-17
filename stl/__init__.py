'''numpy-stl: fast STL file handling powered by NumPy.

Read, write, and manipulate STL files with vectorized
array operations.

Quick start::

    from stl import mesh
    m = mesh.Mesh.from_file('model.stl')
    print(len(m), 'triangles')
'''
from .base import Dimension, RemoveDuplicates
from .mesh import Mesh
from .stl import BUFFER_SIZE, COUNT_SIZE, HEADER_SIZE, MAX_COUNT, Mode

__all__ = [
    'BUFFER_SIZE',
    'COUNT_SIZE',
    'HEADER_SIZE',
    'MAX_COUNT',
    'Dimension',
    'Mesh',
    'Mode',
    'RemoveDuplicates',
]
