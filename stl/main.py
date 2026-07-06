import argparse
import random
import sys
import typing

from . import stl

if typing.TYPE_CHECKING:  # pragma: no cover
    from typing import IO


def _get_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        'infile',
        nargs='?',
        default='-',
        help="STL file to read ('-' or omitted reads from stdin)",
    )
    parser.add_argument(
        'outfile',
        nargs='?',
        default='-',
        help="STL file to write ('-' or omitted writes to stdout)",
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
    parser.add_argument(
        '-s',
        '--disable-speedups',
        action='store_true',
        help='Disable Cython speedups',
    )
    return parser


# The std stream defaults use the underlying binary buffers: STL
# data is binary and the text wrappers would corrupt it (or raise).
def _open_infile(path: str) -> 'IO[bytes]':
    if path == '-':
        return sys.stdin.buffer
    return open(path, 'rb')


def _open_outfile(path: str) -> 'IO[bytes]':
    if path == '-':
        return sys.stdout.buffer
    return open(path, 'wb')


def _get_name(args: argparse.Namespace) -> str:
    names: list[str | None] = [args.name, args.outfile, args.infile]

    for name in names:
        if not isinstance(name, str):
            continue
        elif name == '-':
            continue
        elif r'\AppData\Local\Temp' in name:  # pragma: no cover
            # Windows temp file
            continue
        else:
            return name

    return 'numpy-stl-%06d' % random.randint(0, 1_000_000)  # noqa: UP031


def _convert(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    mode: 'stl.Mode',
) -> None:
    """Load the input mesh and save it in the requested mode.

    The input is opened and fully loaded before the output is opened,
    so a bad input path never truncates an existing output file.
    """
    name: str = _get_name(args)

    try:
        infile: IO[bytes] = _open_infile(args.infile)
    except OSError as exc:
        parser.error(f"can't open {args.infile!r}: {exc}")

    try:
        stl_file = stl.StlMesh(
            filename=name,
            fh=infile,
            calculate_normals=False,
            remove_empty_areas=args.remove_empty_areas,
            speedups=not args.disable_speedups,
        )
    finally:
        if infile is not sys.stdin.buffer:
            infile.close()

    try:
        outfile: IO[bytes] = _open_outfile(args.outfile)
    except OSError as exc:
        parser.error(f"can't open {args.outfile!r}: {exc}")

    try:
        stl_file.save(
            name,
            outfile,
            mode=mode,
            update_normals=not args.use_file_normals,
        )
    finally:
        if outfile is not sys.stdout.buffer:
            outfile.close()


def main() -> None:
    """CLI entry point for the ``stl`` command.

    Converts between ASCII and binary STL formats.
    Supports ``-a`` (force ASCII), ``-b`` (force binary),
    ``-n`` (keep file normals), and ``-s`` (disable speedups).
    """
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

    mode: stl.Mode
    if args.binary:
        mode = stl.BINARY
    elif args.ascii:
        mode = stl.ASCII
    else:
        mode = stl.AUTOMATIC

    _convert(parser, args, mode)


def to_ascii() -> None:
    """CLI entry point for the ``stl2ascii`` command.

    Converts an STL file to ASCII format.
    Supports ``-n`` (keep file normals) and ``-s`` (disable speedups).
    """
    parser = _get_parser('Convert STL files to ASCII (text) format')
    _convert(parser, parser.parse_args(), stl.ASCII)


def to_binary() -> None:
    """CLI entry point for the ``stl2bin`` command.

    Converts an STL file to binary format.
    Supports ``-n`` (keep file normals) and ``-s`` (disable speedups).
    """
    parser = _get_parser('Convert STL files to binary format')
    _convert(parser, parser.parse_args(), stl.BINARY)
