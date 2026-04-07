"""PLY (Polygon File Format) reader.

Supports ASCII, binary little-endian, and binary big-endian
PLY files. Binary formats raise ``ValueError`` until the
binary reader is implemented (Task 3).
"""

from __future__ import annotations

from typing import IO, Any

import numpy as np

# Mapping from PLY type names to (struct format, numpy dtype, byte size).
# Covers both the verbose and short type names from the PLY spec.
_PLY_TYPES: dict[str, tuple[str, type[Any], int]] = {
    'char': ('b', np.int8, 1),
    'int8': ('b', np.int8, 1),
    'uchar': ('B', np.uint8, 1),
    'uint8': ('B', np.uint8, 1),
    'short': ('h', np.int16, 2),
    'int16': ('h', np.int16, 2),
    'ushort': ('H', np.uint16, 2),
    'uint16': ('H', np.uint16, 2),
    'int': ('i', np.int32, 4),
    'int32': ('i', np.int32, 4),
    'uint': ('I', np.uint32, 4),
    'uint32': ('I', np.uint32, 4),
    'float': ('f', np.float32, 4),
    'float32': ('f', np.float32, 4),
    'double': ('d', np.float64, 8),
    'float64': ('d', np.float64, 8),
}


class _Property:
    """A single PLY property definition."""

    __slots__ = ('count_type', 'is_list', 'item_type', 'name', 'type_name')

    def __init__(
        self,
        name: str,
        type_name: str,
        is_list: bool = False,
        count_type: str = '',
        item_type: str = '',
    ) -> None:
        self.name = name
        self.type_name = type_name
        self.is_list = is_list
        self.count_type = count_type
        self.item_type = item_type


class _Element:
    """A PLY element definition (e.g. vertex, face)."""

    __slots__ = ('count', 'name', 'properties')

    def __init__(self, name: str, count: int) -> None:
        self.name = name
        self.count = count
        self.properties: list[_Property] = []


def _parse_header(
    fh: IO[bytes],
) -> tuple[str, list[_Element], str]:
    """Parse the PLY header from an open binary file.

    Returns:
        A tuple of (format_string, elements, object_name).
        format_string is one of 'ascii', 'binary_little_endian',
        or 'binary_big_endian'.  object_name is taken from
        the first ``obj_info`` or ``comment`` line, or
        defaults to 'ply'.
    """
    magic = fh.readline().strip()
    if magic != b'ply':
        raise ValueError(f'Not a PLY file (expected "ply", got {magic!r})')

    format_str = ''
    elements: list[_Element] = []
    obj_name = 'ply'
    current: _Element | None = None

    while True:
        raw = fh.readline()
        if not raw:
            raise ValueError('Unexpected end of file while parsing header')
        line = raw.decode('ascii', errors='replace').strip()

        if line == 'end_header':
            break

        parts = line.split()
        if not parts:
            continue

        keyword = parts[0]

        if keyword == 'format':
            format_str = parts[1]
        elif keyword == 'element':
            current = _Element(parts[1], int(parts[2]))
            elements.append(current)
        elif keyword == 'property':
            if current is None:
                raise ValueError('Property before any element')
            if parts[1] == 'list':
                prop = _Property(
                    name=parts[4],
                    type_name='list',
                    is_list=True,
                    count_type=parts[2],
                    item_type=parts[3],
                )
            else:
                prop = _Property(
                    name=parts[2],
                    type_name=parts[1],
                )
            current.properties.append(prop)
        elif keyword == 'obj_info':
            obj_name = ' '.join(parts[1:])
        # comment and other lines are ignored

    if not format_str:
        raise ValueError('No format line found in PLY header')

    return format_str, elements, obj_name


def _find_elements(
    elements: list[_Element],
) -> tuple[_Element, _Element]:
    """Find the vertex and face elements.

    Returns:
        (vertex_element, face_element)

    Raises:
        ValueError: If vertex or face element is missing.
    """
    vertex: _Element | None = None
    face: _Element | None = None

    for elem in elements:
        if elem.name == 'vertex':
            vertex = elem
        elif elem.name == 'face':
            face = elem

    if vertex is None:
        raise ValueError('PLY file has no vertex element')
    if face is None:
        raise ValueError('PLY file has no face element')

    return vertex, face


def _find_xyz_indices(
    vertex_elem: _Element,
) -> tuple[int, int, int]:
    """Find the indices of x, y, z properties.

    Returns:
        (x_index, y_index, z_index)

    Raises:
        ValueError: If any of x, y, z is missing.
    """
    xi = yi = zi = -1
    for i, prop in enumerate(vertex_elem.properties):
        if prop.name == 'x':
            xi = i
        elif prop.name == 'y':
            yi = i
        elif prop.name == 'z':
            zi = i

    if xi < 0 or yi < 0 or zi < 0:
        raise ValueError('Vertex element missing x, y, or z property')
    return xi, yi, zi


def _read_ascii(
    fh: IO[bytes],
    elements: list[_Element],
) -> tuple[np.ndarray, list[list[int]]]:
    """Read ASCII PLY data after the header.

    Returns:
        (vertices, faces) where vertices is (N, 3) float32
        and faces is a list of index lists (variable length).
    """
    vertex_elem, face_elem = _find_elements(elements)
    xi, yi, zi = _find_xyz_indices(vertex_elem)

    # Read all elements in order
    vertices = np.empty((vertex_elem.count, 3), dtype=np.float32)
    faces: list[list[int]] = []

    for elem in elements:
        for row_idx in range(elem.count):
            raw = fh.readline()
            if not raw:
                raise ValueError(
                    f'Unexpected EOF reading {elem.name} row {row_idx}'
                )
            line = raw.decode('ascii', errors='replace')
            parts = line.split()

            if elem is vertex_elem:
                vertices[row_idx] = [
                    float(parts[xi]),
                    float(parts[yi]),
                    float(parts[zi]),
                ]
            elif elem is face_elem:
                # First value is the vertex count,
                # followed by vertex indices
                n = int(parts[0])
                indices = [int(parts[j + 1]) for j in range(n)]
                faces.append(indices)
            # Other elements: skip (already consumed)

    return vertices, faces


def _triangulate(
    faces: list[list[int]],
) -> list[tuple[int, int, int]]:
    """Fan-triangulate polygon faces into triangles.

    Each face with N vertices is split into N-2 triangles
    using fan triangulation from vertex 0.

    Returns:
        List of (i, j, k) triangle index tuples.
    """
    triangles: list[tuple[int, int, int]] = []
    for face in faces:
        if len(face) < 3:
            continue
        v0 = face[0]
        for i in range(1, len(face) - 1):
            triangles.append((v0, face[i], face[i + 1]))
    return triangles


def _build_mesh_data(
    vertices: np.ndarray,
    triangles: list[tuple[int, int, int]],
    mesh_dtype: np.dtype,  # type: ignore[type-arg]
) -> np.ndarray:
    """Build the structured numpy array for the mesh.

    Args:
        vertices: (N, 3) float32 vertex positions.
        triangles: List of (i, j, k) index tuples.
        mesh_dtype: The mesh dtype (normals, vectors, attr).

    Returns:
        Structured 1-D numpy array with the mesh dtype.
    """
    count = len(triangles)
    data = np.zeros(count, dtype=mesh_dtype)
    for i, (a, b, c) in enumerate(triangles):
        data['vectors'][i][0] = vertices[a]
        data['vectors'][i][1] = vertices[b]
        data['vectors'][i][2] = vertices[c]
    return data


def read_ply(
    fh: IO[bytes],
    mesh_dtype: np.dtype,  # type: ignore[type-arg]
) -> tuple[np.ndarray, str]:
    """Read a PLY file and return mesh data.

    Args:
        fh: Open binary file handle positioned at the
            start of the PLY file.
        mesh_dtype: The structured dtype for the mesh
            (normals, vectors, attr).

    Returns:
        (data, name) where data is a structured numpy
        array and name is the object name from the header.

    Raises:
        ValueError: If the file is not a valid PLY file
            or uses an unsupported binary format.
    """
    format_str, elements, obj_name = _parse_header(fh)

    if format_str == 'ascii':
        vertices, faces = _read_ascii(fh, elements)
    elif format_str in (
        'binary_little_endian',
        'binary_big_endian',
    ):
        raise ValueError(
            f'Binary PLY format ({format_str}) is not yet supported'
        )
    else:
        raise ValueError(f'Unknown PLY format: {format_str!r}')

    triangles = _triangulate(faces)
    data = _build_mesh_data(vertices, triangles, mesh_dtype)
    return data, obj_name
