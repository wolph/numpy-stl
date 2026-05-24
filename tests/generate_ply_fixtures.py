"""Generate binary PLY test fixtures from the ASCII cube definition."""

import pathlib
import struct

VERTICES = [
    (-1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0),
    (1.0, 1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0),
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0),
]

FACES = [
    (0, 3, 1),
    (1, 3, 2),
    (0, 4, 7),
    (0, 7, 3),
    (4, 5, 6),
    (4, 6, 7),
    (5, 1, 2),
    (5, 2, 6),
    (2, 3, 6),
    (3, 7, 6),
    (0, 1, 5),
    (0, 5, 4),
]

BASE = pathlib.Path(__file__).parent


def write_binary_ply(path: pathlib.Path, endian: str) -> None:
    fmt_char = '<' if endian == 'little' else '>'
    format_name = (
        'binary_little_endian' if endian == 'little' else 'binary_big_endian'
    )

    header = (
        f'ply\n'
        f'format {format_name} 1.0\n'
        f'comment Cube test fixture\n'
        f'element vertex {len(VERTICES)}\n'
        f'property float x\n'
        f'property float y\n'
        f'property float z\n'
        f'element face {len(FACES)}\n'
        f'property list uchar int vertex_indices\n'
        f'end_header\n'
    ).encode('ascii')

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(header)
        for v in VERTICES:
            fh.write(struct.pack(f'{fmt_char}fff', *v))
        for face in FACES:
            fh.write(struct.pack('B', len(face)))
            for idx in face:
                fh.write(struct.pack(f'{fmt_char}i', idx))


write_binary_ply(BASE / 'ply_binary' / 'Cube.ply', 'little')
write_binary_ply(BASE / 'ply_binary' / 'CubeBigEndian.ply', 'big')
print('Generated binary PLY fixtures')
