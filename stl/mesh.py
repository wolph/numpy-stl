from . import stl


class Mesh(stl.BaseStl):
    """Primary user-facing class for STL mesh operations.

    Inherits all functionality from
    :class:`~stl.stl.BaseStl` and
    :class:`~stl.base.BaseMesh`. Use class methods
    like :meth:`from_file` and :meth:`from_multi_file`
    to load STL files, or pass a structured NumPy array
    to the constructor.

    Example:
        >>> import numpy as np
        >>> from stl import mesh
        >>> data = np.zeros(1, dtype=mesh.Mesh.dtype)
        >>> m = mesh.Mesh(data, remove_empty_areas=False)
        >>> len(m)
        1
    """
