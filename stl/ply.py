"""PLY (Polygon File Format) reader.

Supports ASCII, binary little-endian, and binary big-endian
PLY files. Binary formats raise ``ValueError`` until the
binary reader is implemented (Task 3).
"""

from __future__ import annotations

import struct
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


def _parse_property(
    parts: list[str],
) -> _Property:
    """Parse a PLY property line into a _Property."""
    if parts[1] == 'list':
        return _Property(
            name=parts[4],
            type_name='list',
            is_list=True,
            count_type=parts[2],
            item_type=parts[3],
        )
    return _Property(
        name=parts[2],
        type_name=parts[1],
    )


def _parse_header(
    fh: IO[bytes],
) -> tuple[str, list[_Element], str]:
    """Parse the PLY header from an open binary file.

    Returns:
        A tuple of (format_string, elements, object_name).
        format_string is one of 'ascii',
        'binary_little_endian', or 'binary_big_endian'.
    """
    magic = fh.readline().strip()
    if magic != b'ply':
        raise ValueError(f'Not a PLY file (expected "ply", got {magic!r})')

    format_str = ''
    elements: list[_Element] = []
    obj_name = 'ply'
    current: _Element | None = None

    for raw in iter(fh.readline, b''):
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
        elif keyword == 'property' and current is not None:
            current.properties.append(_parse_property(parts))
        elif keyword == 'obj_info':
            obj_name = ' '.join(parts[1:])
    else:
        raise ValueError('Unexpected end of file while parsing header')

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


def _element_row_size(element: _Element) -> int:
    """Return the fixed byte size of one binary row of an element.

    Raises:
        ValueError: If the element has list properties; their rows
            have no fixed binary size.
    """
    row_size = 0
    for prop in element.properties:
        if prop.is_list:
            raise ValueError(
                f'Element {element.name!r} has list properties; '
                'its rows have no fixed binary size'
            )
        row_size += _PLY_TYPES[prop.type_name][2]
    return row_size


def _skip_element_rows(fh: IO[bytes], element: _Element) -> None:
    """Skip all binary rows of a fixed-size element, validating EOF.

    Raises:
        ValueError: If the element has list properties or the file
            ends before all rows are consumed.
    """
    size = element.count * _element_row_size(element)
    if len(fh.read(size)) != size:
        raise ValueError(f'Unexpected EOF skipping element {element.name!r}')


def _find_face_list_index(face_elem: _Element) -> int:
    """Return the position of the first list property of a face.

    Raises:
        ValueError: If the face element has no list property.
    """
    for i, prop in enumerate(face_elem.properties):
        if prop.is_list:
            return i
    raise ValueError('Face element has no list property')


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
    # The vertex index list may be preceded by scalar face properties;
    # its count value sits at this field offset.
    list_index = _find_face_list_index(face_elem)

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
                n = int(parts[list_index])
                indices = [int(parts[list_index + 1 + j]) for j in range(n)]
                faces.append(indices)
            # Other elements: skip (already consumed)

    return vertices, faces


def _read_binary_vertices(
    raw: bytes,
    vertex_elem: _Element,
    endian: str,
    xi: int,
    yi: int,
    zi: int,
) -> np.ndarray:
    """Unpack binary vertex data into (N, 3) float32."""
    vertex_fmt = endian
    vertex_size = 0
    for prop in vertex_elem.properties:
        fmt_char, _, size = _PLY_TYPES[prop.type_name]
        vertex_fmt += fmt_char
        vertex_size += size

    n_verts = vertex_elem.count
    vertices = np.empty((n_verts, 3), dtype=np.float32)
    for i in range(n_verts):
        vals = struct.unpack_from(vertex_fmt, raw, i * vertex_size)
        vertices[i] = [vals[xi], vals[yi], vals[zi]]
    return vertices


def _read_binary_faces(
    fh: IO[bytes],
    face_elem: _Element,
    endian: str,
) -> list[list[int]]:
    """Read binary face data.

    Every property of each face row is consumed in declared order;
    scalar properties and extra list properties around the vertex
    index list are read and discarded.
    """
    list_prop = face_elem.properties[_find_face_list_index(face_elem)]

    faces: list[list[int]] = []
    for _ in range(face_elem.count):
        indices: list[int] = []
        for prop in face_elem.properties:
            if prop.is_list:
                count_fmt_char, _, count_size = _PLY_TYPES[prop.count_type]
                item_fmt_char, _, item_size = _PLY_TYPES[prop.item_type]
                count_raw = fh.read(count_size)
                if len(count_raw) != count_size:
                    raise ValueError('Unexpected EOF reading face')
                n = struct.unpack(endian + count_fmt_char, count_raw)[0]
                item_raw = fh.read(n * item_size)
                if len(item_raw) != n * item_size:
                    raise ValueError('Unexpected EOF reading face indices')
                if prop is list_prop:
                    indices = list(
                        struct.unpack(endian + item_fmt_char * n, item_raw)
                    )
            else:
                # Skip scalar properties around the index list.
                scalar_size = _PLY_TYPES[prop.type_name][2]
                if len(fh.read(scalar_size)) != scalar_size:
                    raise ValueError(
                        'Unexpected EOF reading face scalar property'
                    )
        faces.append(indices)
    return faces


def _skip_binary_elements(
    fh: IO[bytes],
    elements: list[_Element],
    vertex_elem: _Element,
    face_elem: _Element,
) -> None:
    """Skip binary elements between vertex and face."""
    found_vertex = False
    for elem in elements:
        if elem is vertex_elem:
            found_vertex = True
            continue
        if elem is face_elem:
            break
        if not found_vertex:
            continue
        _skip_element_rows(fh, elem)


def _read_binary(
    fh: IO[bytes],
    elements: list[_Element],
    format_str: str,
) -> tuple[np.ndarray, list[list[int]]]:
    """Read binary PLY data after the header.

    Returns:
        (vertices, faces) where vertices is (N, 3)
        float32 and faces is a list of index lists.
    """
    endian = '<' if format_str == 'binary_little_endian' else '>'
    vertex_elem, face_elem = _find_elements(elements)
    xi, yi, zi = _find_xyz_indices(vertex_elem)

    # Skip elements declared before the vertex element; their data
    # precedes the vertex data in the binary stream.
    for elem in elements[: elements.index(vertex_elem)]:
        _skip_element_rows(fh, elem)

    # Compute vertex row size for bulk read.
    vertex_size = _element_row_size(vertex_elem)
    n_verts = vertex_elem.count
    raw = fh.read(n_verts * vertex_size)
    if len(raw) != n_verts * vertex_size:
        raise ValueError('Unexpected EOF reading vertex data')

    vertices = _read_binary_vertices(raw, vertex_elem, endian, xi, yi, zi)
    _skip_binary_elements(fh, elements, vertex_elem, face_elem)
    faces = _read_binary_faces(fh, face_elem, endian)
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
        triangles.extend(
            (v0, face[i], face[i + 1]) for i in range(1, len(face) - 1)
        )
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
    n_vertices = len(vertices)
    data = np.zeros(count, dtype=mesh_dtype)
    for i, (a, b, c) in enumerate(triangles):
        for index in (a, b, c):
            # Negative indices must not wrap around silently.
            if not 0 <= index < n_vertices:
                raise ValueError(
                    f'Face {i} references vertex index {index}, which is '
                    f'out of range (0..{n_vertices - 1})'
                )
        data['vectors'][i][0] = vertices[a]
        data['vectors'][i][1] = vertices[b]
        data['vectors'][i][2] = vertices[c]
    return data


def _deduplicate_vertices(
    data: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Extract unique vertices and face indices.

    Args:
        data: Mesh.dtype structured array.

    Returns:
        (vertices, faces) where vertices is (N, 3)
        float32 and faces is a list of (i, j, k) tuples.
    """
    all_verts = data['vectors'].reshape(-1, 3)
    unique_verts, inverse = np.unique(all_verts, axis=0, return_inverse=True)
    faces: list[tuple[int, int, int]] = []
    for i in range(len(data)):
        base = i * 3
        faces.append(
            (
                int(inverse[base]),
                int(inverse[base + 1]),
                int(inverse[base + 2]),
            )
        )
    return unique_verts, faces


def write_ply(
    fh: IO[bytes],
    data: np.ndarray,
    name: str = '',
    mode: str = 'binary_little_endian',
) -> None:
    """Write mesh data to PLY format.

    Args:
        fh: Binary file handle.
        data: Mesh.dtype structured array.
        name: Optional object name.
        mode: 'ascii', 'binary_little_endian', or
              'binary_big_endian'.
    """
    valid_modes = (
        'ascii',
        'binary_little_endian',
        'binary_big_endian',
    )
    if mode not in valid_modes:
        raise ValueError(
            f'Unknown PLY mode {mode!r}, expected one of {valid_modes}'
        )

    vertices, faces = _deduplicate_vertices(data)

    lines = [
        'ply',
        f'format {mode} 1.0',
    ]
    if name:
        lines.append(f'obj_info {name}')
    lines.extend(
        [
            f'element vertex {len(vertices)}',
            'property float x',
            'property float y',
            'property float z',
            f'element face {len(faces)}',
            'property list uchar int vertex_indices',
            'end_header',
        ]
    )
    header = '\n'.join(lines) + '\n'
    fh.write(header.encode('ascii'))

    if mode == 'ascii':
        for v in vertices:
            fh.write(f'{v[0]} {v[1]} {v[2]}\n'.encode('ascii'))
        for face in faces:
            fh.write(f'3 {face[0]} {face[1]} {face[2]}\n'.encode('ascii'))
    else:
        endian = '<' if mode == 'binary_little_endian' else '>'
        for v in vertices:
            fh.write(
                struct.pack(
                    f'{endian}fff',
                    float(v[0]),
                    float(v[1]),
                    float(v[2]),
                )
            )
        for face in faces:
            fh.write(struct.pack('B', 3))
            fh.write(
                struct.pack(
                    f'{endian}iii',
                    face[0],
                    face[1],
                    face[2],
                )
            )


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
        vertices, faces = _read_binary(fh, elements, format_str)
    else:
        raise ValueError(f'Unknown PLY format: {format_str!r}')

    triangles = _triangulate(faces)
    data = _build_mesh_data(vertices, triangles, mesh_dtype)
    return data, obj_name
