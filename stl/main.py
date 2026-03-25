import argparse
import random
import sys

from . import stl


def _get_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        'infile',
        nargs='?',
        type=argparse.FileType('rb'),
        default=sys.stdin,
        help='STL file to read',
    )
    parser.add_argument(
        'outfile',
        nargs='?',
        type=argparse.FileType('wb'),
        default=sys.stdout,
        help='STL file to write',
    )
    parser.add_argument('--name', nargs='?', help='Name of the mesh')
    parser.add_argument(
        '-n',
        '--use-file-normals',
        action='store_true',
        help='Read the normals from the file instead of recalculating them',
    )
    parser.add_argument(
        '-r',
        '--remove-empty-areas',
        action='store_true',
        help='Remove areas with 0 surface areas to prevent errors during '
        'normal calculation',
    )
    return parser


def _get_name(args: argparse.Namespace) -> str:
    names = [
        args.name,
        getattr(args.outfile, 'name', None),
        getattr(args.infile, 'name', None),
    ]

    for name in names:
        if not isinstance(name, str):
            continue
        elif name.startswith('<'):  # pragma: no cover
            continue
        elif r'\AppData\Local\Temp' in name:  # pragma: no cover
            # Windows temp file
            continue
        else:
            return name

    return 'numpy-stl-%06d' % random.randint(0, 1_000_000)  # noqa: UP031


def main() -> None:
    '''CLI entry point for the ``stl`` command.

    Converts between ASCII and binary STL formats.
    Supports ``-a`` (force ASCII), ``-b`` (force binary),
    and ``-n`` (recalculate normals).
    '''
    parser = _get_parser('Convert STL files from ascii to binary and back')
    parser.add_argument(
        '-a',
        '--ascii',
        action='store_true',
        help='Write ASCII file (default is binary)',
    )
    parser.add_argument(
        '-b',
        '--binary',
        action='store_true',
        help='Force binary file (for TTYs)',
    )

    args = parser.parse_args()
    name = _get_name(args)
    stl_file = stl.StlMesh(
        filename=name,
        fh=args.infile,
        calculate_normals=False,
        remove_empty_areas=args.remove_empty_areas,
        speedups=True,
    )

    if args.binary:
        mode = stl.BINARY
    elif args.ascii:
        mode = stl.ASCII
    else:
        mode = stl.AUTOMATIC

    stl_file.save(
        name, args.outfile, mode=mode, update_normals=not args.use_file_normals
    )


def to_ascii() -> None:
    '''CLI entry point for the ``stl2ascii`` command.

    Converts an STL file to ASCII format.
    '''
    parser = _get_parser('Convert STL files to ASCII (text) format')
    args = parser.parse_args()
    name = _get_name(args)
    stl_file = stl.StlMesh(
        filename=name,
        fh=args.infile,
        calculate_normals=False,
        remove_empty_areas=args.remove_empty_areas,
        speedups=True,
    )
    stl_file.save(
        name,
        args.outfile,
        mode=stl.ASCII,
        update_normals=not args.use_file_normals,
    )


def to_binary() -> None:
    '''CLI entry point for the ``stl2bin`` command.

    Converts an STL file to binary format.
    '''
    parser = _get_parser('Convert STL files to binary format')
    args = parser.parse_args()
    name = _get_name(args)
    stl_file = stl.StlMesh(
        filename=name,
        fh=args.infile,
        calculate_normals=False,
        remove_empty_areas=args.remove_empty_areas,
        speedups=True,
    )
    stl_file.save(
        name,
        args.outfile,
        mode=stl.BINARY,
        update_normals=not args.use_file_normals,
    )
