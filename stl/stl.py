import datetime
import enum
import io
import os
import struct
import zipfile
from collections.abc import Generator
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Literal as L,  # noqa: N817
    cast,
)
from xml.etree import ElementTree as ET

import numpy as np

from . import (
    __about__ as metadata,
    base,
)
from ._compat import (
    ascii_read as _ascii_read,
    ascii_write as _ascii_write,
)
from .utils import b

if TYPE_CHECKING:
    from typing import Protocol, type_check_only

    from _typeshed import SupportsWrite
    from typing_extensions import Buffer, Self, Writer

    from .base import _data_1d

    _Name = bytes | str

    @type_check_only
    class _StatefulWriter(Writer[Buffer], Protocol):
        def tell(self) -> int: ...


class Mode(enum.IntEnum):
    """STL file format mode for loading and saving.

    Use :attr:`AUTOMATIC` for auto-detection (default),
    :attr:`ASCII` to force ASCII format, or
    :attr:`BINARY` to force binary format.
    """

    #: Automatically detect whether the output is a TTY, if so, write ASCII
    #: otherwise write BINARY
    AUTOMATIC = 0
    #: Force writing ASCII
    ASCII = 1
    #: Force writing BINARY
    BINARY = 2


# For backwards compatibility, leave the original references
AUTOMATIC: L[Mode.AUTOMATIC] = Mode.AUTOMATIC
ASCII: L[Mode.ASCII] = Mode.ASCII
BINARY: L[Mode.BINARY] = Mode.BINARY

#: Amount of bytes to read while using buffered reading
BUFFER_SIZE: int = 4096
#: The amount of bytes in the header field
HEADER_SIZE: int = 80
#: The amount of bytes in the count field
COUNT_SIZE: int = 4
#: The maximum amount of triangles we can read from binary files
MAX_COUNT: float = 1e8
#: The header format, can be safely monkeypatched. Limited to 80 characters
HEADER_FORMAT: str = '{package_name} ({version}) {now} {name}'


class BaseStl(base.BaseMesh):
    @classmethod
    def load(
        cls,
        fh: IO[Any],
        mode: Mode = AUTOMATIC,
        speedups: bool = True,
    ) -> 'tuple[bytes, _data_1d] | Any':
        """Load mesh data from an open STL file handle.

        Auto-detects binary vs ASCII format unless
        ``mode`` is explicitly set.

        Args:
            fh: Open binary file handle.
            mode: Force a specific format or use
                :attr:`Mode.AUTOMATIC` (default).
            speedups: Use Cython speedups for ASCII
                parsing. Defaults to True.

        Returns:
            A (name, data) tuple, or None if the file
            is empty.
        """
        header = fh.read(HEADER_SIZE)
        if not header:
            return None

        if isinstance(header, str):  # pragma: no branch
            header = b(header)

        if mode is AUTOMATIC:
            if header.lstrip().lower().startswith(b'solid'):
                try:
                    name, data = cls._load_ascii(fh, header, speedups=speedups)
                except RuntimeError as exception:
                    (recoverable, _) = exception.args
                    # If we didn't read beyond the header the stream is still
                    # readable through the binary reader
                    if recoverable:
                        name, data = cls._load_binary(
                            fh, header, check_size=False
                        )
                    else:
                        # Apparently we've read beyond the header. Let's try
                        # seeking :)
                        # Note that this fails when reading from stdin, we
                        # can't recover from that.
                        fh.seek(HEADER_SIZE)

                        # Since we know this is a seekable file now and we're
                        # not 100% certain it's binary, check the size while
                        # reading
                        name, data = cls._load_binary(
                            fh, header, check_size=True
                        )
            else:
                name, data = cls._load_binary(fh, header)
        elif mode is ASCII:
            name, data = cls._load_ascii(fh, header, speedups=speedups)
        else:
            name, data = cls._load_binary(fh, header)

        return name, data

    @classmethod
    def _load_binary(
        cls,
        fh: IO[bytes],
        header: bytes,
        check_size: bool = False,
    ) -> tuple[bytes, '_data_1d']:
        # Read the triangle count
        count_data = fh.read(COUNT_SIZE)
        if len(count_data) != COUNT_SIZE:
            count = 0
        else:
            (count,) = struct.unpack('<i', b(count_data))
        # raise RuntimeError()
        assert count < MAX_COUNT, (
            f'File too large, got {count} triangles which '
            f'exceeds the maximum of {MAX_COUNT}'
        )

        if check_size:
            try:
                # Check the size of the file
                fh.seek(0, os.SEEK_END)
                raw_size = fh.tell() - HEADER_SIZE - COUNT_SIZE
                expected_count = int(raw_size / cls.dtype.itemsize)
                assert expected_count == count, (
                    f'Expected {expected_count} vectors but header indicates '
                    f'{count}'
                )
                fh.seek(HEADER_SIZE + COUNT_SIZE)
            except OSError:  # pragma: no cover
                pass

        name = header.strip()

        # Read the rest of the binary data
        try:
            return name, np.fromfile(fh, dtype=cls.dtype, count=count)
        except io.UnsupportedOperation:
            data = np.frombuffer(fh.read(), dtype=cls.dtype, count=count)
            # Copy to make the buffer writable
            return name, data.copy()

    @staticmethod
    def _ascii_reader(  # noqa: C901
        fh: IO[bytes], header: bytes
    ) -> Generator[
        'bytes | tuple[list[float], tuple[bytes, bytes, bytes], int]',
        None,
        None,
    ]:
        if b'\n' in header:
            recoverable = [True]
        else:
            recoverable = [False]
            header += b(fh.read(BUFFER_SIZE))

        lines = b(header).split(b'\n')

        def get(prefix: '_Name' = '') -> 'bytes | list[float]':
            prefix = b(prefix).lower()

            if lines:
                raw_line = lines.pop(0)
            else:
                raise RuntimeError(recoverable[0], 'Unable to find more lines')

            if not lines:
                recoverable[0] = False

                # Read more lines and make sure we prepend any old data
                lines[:] = b(fh.read(BUFFER_SIZE)).split(b'\n')
                raw_line += lines.pop(0)

            raw_line = raw_line.strip()
            line = raw_line.lower()
            if line == b(''):
                return get(prefix)

            if prefix:
                if line.startswith(prefix):
                    values = line.replace(prefix, b(''), 1).strip().split()
                elif line.startswith((b('endsolid'), b('end solid'))):
                    # go back to the beginning of new solid part
                    size_unprocessedlines = (
                        sum(len(line) + 1 for line in lines) - 1
                    )

                    if size_unprocessedlines > 0:
                        position = fh.tell()
                        fh.seek(position - size_unprocessedlines)
                    raise StopIteration()
                else:
                    raise RuntimeError(
                        recoverable[0],
                        f'{line!r} should start with {prefix!r}',
                    )

                if len(values) == 3:
                    return [float(v) for v in values]
                else:  # pragma: no cover
                    raise RuntimeError(
                        recoverable[0], f'Incorrect value {line!r}'
                    )
            else:
                return b(raw_line)

        line = get()
        if not lines:
            raise RuntimeError(
                recoverable[0], 'No lines found, impossible to read'
            )

        # Yield the name
        yield cast('bytes', line[5:]).strip()

        while True:
            # Read from the header lines first, until that point we can recover
            # and go to the binary option. After that we cannot due to
            # unseekable files such as sys.stdin
            #
            # Numpy doesn't support any non-file types so wrapping with a
            # buffer and/or StringIO does not work.
            try:
                normals = cast('list[float]', get('facet normal'))
                assert cast('bytes', get()).lower() == b('outer loop')
                v0 = cast('bytes', get('vertex'))
                v1 = cast('bytes', get('vertex'))
                v2 = cast('bytes', get('vertex'))
                assert cast('bytes', get()).lower() == b('endloop')
                assert cast('bytes', get()).lower() == b('endfacet')
                attrs = 0
                yield (normals, (v0, v1, v2), attrs)
            except AssertionError as e:  # pragma: no cover  # noqa: PERF203
                raise RuntimeError(recoverable[0], e) from e
            except StopIteration:
                return

    @classmethod
    def _load_ascii(
        cls,
        fh: IO[bytes],
        header: bytes,
        speedups: bool = True,
    ) -> tuple[bytes, '_data_1d']:
        # Speedups does not support non file-based streams
        try:
            fh.fileno()
        except io.UnsupportedOperation:
            speedups = False
        if _ascii_read is not None and speedups:
            return _ascii_read(fh, header)
        else:
            iterator = cls._ascii_reader(fh, header)
            name = cast('bytes', next(iterator))
            return name, np.fromiter(iterator, dtype=cls.dtype)

    def save(  # noqa: C901
        self,
        filename: '_Name',
        fh: 'IO[bytes] | None' = None,
        mode: 'Mode | int' = AUTOMATIC,
        update_normals: bool = True,
    ) -> None:
        """Save the mesh to an STL file.

        If mode is :attr:`Mode.AUTOMATIC`, writes binary
        unless the output is a TTY.

        Args:
            filename: Output file path. Required even when
                ``fh`` is provided (used for STL header).
            fh: Optional pre-opened binary file handle.
            mode: Output format. Defaults to
                :attr:`Mode.AUTOMATIC`.
            update_normals: Whether to recalculate normals
                before saving. Defaults to True.

        Raises:
            TypeError: If ``fh`` is a text-mode handle.

        Example:
            >>> import numpy as np
            >>> from stl import mesh
            >>> data = np.zeros(1, dtype=mesh.Mesh.dtype)
            >>> data['vectors'][0] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
            >>> m = mesh.Mesh(data, remove_empty_areas=False)
            >>> m.save('/tmp/_numpy_stl_test.stl')

        Warning:
            Even for ASCII output, the file handle must be
            opened in binary mode (``'wb'``). A text-mode
            handle raises ``TypeError``.
        """
        assert filename, 'Filename is required for the STL headers'
        if update_normals:
            self.update_normals()

        if mode is AUTOMATIC:
            # Try to determine if the file is a TTY.
            if fh:
                try:
                    if os.isatty(fh.fileno()):  # pragma: no cover
                        write = self._write_ascii
                    else:
                        write = self._write_binary
                except OSError:
                    # If TTY checking fails then it's an io.BytesIO() (or one
                    # of its siblings from io). Assume binary.
                    write = self._write_binary
            else:
                write = self._write_binary
        elif mode is BINARY:
            write = self._write_binary
        elif mode is ASCII:
            write = self._write_ascii
        else:
            raise ValueError(f'Mode {mode!r} is invalid')

        if isinstance(fh, io.TextIOBase):
            # Provide a more helpful error if the user mistakenly
            # assumes ASCII files should be text files.
            raise TypeError(
                'File handles should be in binary mode - even when'
                ' writing an ASCII STL.'
            )

        name = self.name
        if not name:
            name = os.path.split(filename)[-1]

        try:
            if fh:
                write(fh, name)
            else:
                with open(filename, 'wb') as fh:
                    write(fh, name)
        except OSError:  # pragma: no cover
            pass

    def _write_ascii(self, fh: IO[bytes], name: '_Name') -> None:
        try:
            fh.fileno()
            speedups = self.speedups
        except io.UnsupportedOperation:
            speedups = False

        if _ascii_write is not None and speedups:
            _ascii_write(fh, b(name), self.data)
        else:

            def p(s: '_Name', file: 'SupportsWrite[bytes]') -> None:
                file.write(b(s) + b'\n')

            p(b'solid ' + b(name), file=fh)

            for row in self.data:
                # Explicitly convert each component to standard float for
                # normals and vertices to be compatible with numpy 2.x
                normals = tuple(float(n) for n in row['normals'])
                vectors = row['vectors']
                p('facet normal {:f} {:f} {:f}'.format(*normals), file=fh)
                p('  outer loop', file=fh)
                p(
                    '    vertex {:f} {:f} {:f}'.format(
                        *tuple(float(v) for v in vectors[0])
                    ),
                    file=fh,
                )
                p(
                    '    vertex {:f} {:f} {:f}'.format(
                        *tuple(float(v) for v in vectors[1])
                    ),
                    file=fh,
                )
                p(
                    '    vertex {:f} {:f} {:f}'.format(
                        *tuple(float(v) for v in vectors[2])
                    ),
                    file=fh,
                )
                p('  endloop', file=fh)
                p('endfacet', file=fh)

            p(b'endsolid ' + b(name), file=fh)

    def get_header(self, name: '_Name') -> str:
        """Build the 80-byte binary STL header string.

        Args:
            name: Solid name to embed in the header.

        Returns:
            Header string truncated to 80 bytes.
        """
        # Format the header
        header: str = HEADER_FORMAT.format(
            package_name=metadata.__package_name__,
            version=metadata.__version__,
            now=datetime.datetime.now(),
            name=name,
        )

        # Make it exactly 80 characters
        return header[:80].ljust(80, ' ')

    def _write_binary(
        self,
        fh: '_StatefulWriter | io.TextIOWrapper',
        name: '_Name',
    ) -> None:
        header = self.get_header(name)
        packed = struct.pack('<i', self.data.size)

        if isinstance(fh, io.TextIOWrapper):  # pragma: no cover
            fh.write(header)
            fh.write(str(packed))
        else:
            fh.write(b(header))
            fh.write(b(packed))

        if isinstance(fh, io.BufferedWriter):
            # Write to a true file.
            self.data.tofile(fh)
        else:
            # Write to a pseudo buffer.
            cast('_StatefulWriter', fh).write(self.data.data)

        # In theory this should no longer be possible but I'll leave it here
        # anyway...
        if self.data.size:  # pragma: no cover
            assert fh.tell() > 84, (
                'numpy silently refused to write our file. Note that writing '
                'to `StringIO` objects is not supported by `numpy`'
            )

    @classmethod
    def from_file(
        cls,
        filename: str,
        calculate_normals: bool = True,
        fh: 'IO[bytes] | None' = None,
        mode: Mode = Mode.AUTOMATIC,
        speedups: bool = True,
        **kwargs: Any,
    ) -> 'Self':
        """Load a mesh from an STL file.

        Reads binary or ASCII STL files. Format is
        auto-detected unless ``mode`` is explicitly set.

        Args:
            filename: Path to the STL file.
            calculate_normals: Whether to recalculate
                normals after loading. Defaults to True.
            fh: Optional pre-opened binary file handle.
                If provided, ``filename`` is used only
                for the mesh name.
            mode: Force ASCII or BINARY loading, or
                AUTOMATIC detection (default).
            speedups: Use Cython speedups for ASCII
                parsing when available. Defaults to True.
            **kwargs: Additional arguments passed to
                the Mesh constructor.

        Returns:
            A new Mesh instance containing the loaded data.

        Example:
            >>> from stl import mesh
            >>> m = mesh.Mesh.from_file('tests/stl_binary/HalfDonut.stl')
            >>> len(m.data) > 0
            True

        Note:
            When ``speedups`` is True and the speedups
            package is installed, ASCII parsing uses a
            fast C implementation. Speedups are
            automatically disabled for non-seekable
            streams (e.g., stdin).
        """
        if fh:
            name, data = cls.load(fh, mode=mode, speedups=speedups)
        else:
            with open(filename, 'rb') as fh:
                name, data = cls.load(fh, mode=mode, speedups=speedups)

        # pyrefly: ignore[bad-return]
        return cls(
            data, calculate_normals, name=name, speedups=speedups, **kwargs
        )

    @classmethod
    def from_multi_file(
        cls,
        filename: str,
        calculate_normals: bool = True,
        fh: 'IO[bytes] | None' = None,
        mode: Mode = Mode.AUTOMATIC,
        speedups: bool = True,
        **kwargs: Any,
    ) -> Generator['Self', None, None]:
        """Load multiple solids from a single STL file.

        Yields one Mesh per ``solid`` block found.

        Args:
            filename: Path to the STL file.
            calculate_normals: Whether to recalculate
                normals. Defaults to True.
            fh: Optional pre-opened binary file handle.
            mode: Format mode. Defaults to
                :attr:`Mode.AUTOMATIC`.
            speedups: Use Cython speedups when available.
            **kwargs: Additional arguments passed to
                the Mesh constructor.

        Yields:
            Mesh instances, one per solid block.

        Example:
            >>> from stl import mesh
            >>> # Single-solid file yields one mesh
            >>> solids = list(
            ...     mesh.Mesh.from_multi_file('tests/stl_ascii/HalfDonut.stl')
            ... )
            >>> len(solids) >= 1
            True

        Note:
            Multi-solid loading only works with ASCII
            STL files. Binary STL files always contain
            a single solid.
        """
        if fh:
            close = False
        else:
            fh = open(filename, 'rb')  # noqa: SIM115
            close = True

        try:
            raw_data = cls.load(fh, mode=mode, speedups=speedups)
            while raw_data:
                name, data = raw_data
                # pyrefly: ignore[invalid-yield]
                yield cls(
                    data,
                    calculate_normals,
                    name=name,
                    speedups=speedups,
                    **kwargs,
                )
                raw_data = cls.load(fh, mode=ASCII, speedups=speedups)

        finally:
            if close:
                fh.close()

    @classmethod
    def from_files(
        cls,
        filenames: 'list[str]',
        calculate_normals: bool = True,
        mode: Mode = Mode.AUTOMATIC,
        speedups: bool = True,
        **kwargs: Any,
    ) -> 'Self':
        """Load and merge multiple STL files into one mesh.

        Args:
            filenames: List of STL file paths.
            calculate_normals: Whether to recalculate
                normals. Defaults to True.
            mode: Format mode for each file.
            speedups: Use Cython speedups when available.
            **kwargs: Additional arguments passed to
                the Mesh constructor.

        Returns:
            A single Mesh with data from all files.

        Example:
            >>> from stl import mesh
            >>> m = mesh.Mesh.from_files(['tests/stl_binary/HalfDonut.stl'])
            >>> len(m.data) > 0
            True
        """
        meshes = [
            cls.from_file(
                filename,
                calculate_normals=calculate_normals,
                mode=mode,
                speedups=speedups,
                **kwargs,
            )
            for filename in filenames
        ]

        data = np.concatenate([mesh.data for mesh in meshes])
        # pyrefly: ignore[bad-return]
        return cls(data, calculate_normals=calculate_normals, **kwargs)

    @classmethod
    def from_3mf_file(
        cls,
        filename: str,
        calculate_normals: bool = True,
        **kwargs: object,
    ) -> Generator['Self', None, None]:
        """Load meshes from a 3MF file (read-only).

        Parses the 3MF ZIP archive and yields one Mesh
        per ``<mesh>`` element found.

        Args:
            filename: Path to the .3mf file.
            calculate_normals: Whether to recalculate
                normals. Defaults to True.
            **kwargs: Additional arguments.

        Yields:
            Mesh instances, one per 3MF mesh element.

        Example:
            >>> from stl import mesh
            >>> meshes = list(mesh.Mesh.from_3mf_file('tests/3mf/Moon.3mf'))
            >>> len(meshes) > 0
            True

        Note:
            3MF support is experimental and read-only.
            Not all 3MF features are supported.
        """
        with zipfile.ZipFile(filename) as zip:
            with zip.open('_rels/.rels') as rels_fh:
                model = None
                root = ET.parse(rels_fh).getroot()
                for child in root:  # pragma: no branch
                    type_ = child.attrib.get('Type', '')
                    if type_.endswith('3dmodel'):  # pragma: no branch
                        model = child.attrib.get('Target', '')
                        break

            assert model, f'No 3D model found in {filename}'
            with zip.open(model.lstrip('/')) as fh:
                root = ET.parse(fh).getroot()

                elements = root.findall('./{*}resources/{*}object/{*}mesh')
                for mesh_element in elements:  # pragma: no branch
                    triangles: list[list[list[float]]] = []
                    vertices: list[list[float]] = []

                    for element in mesh_element:
                        tag = element.tag
                        if tag.endswith('vertices'):
                            # Collect all the vertices
                            for vertice in element:
                                a = {
                                    k: float(v)
                                    for k, v in vertice.attrib.items()
                                }
                                vertices.append([a['x'], a['y'], a['z']])

                        elif tag.endswith('triangles'):  # pragma: no branch
                            # Map the triangles to the vertices and collect
                            for triangle in element:
                                a = {
                                    k: int(v)
                                    for k, v in triangle.attrib.items()
                                }
                                triangles.append(
                                    [
                                        vertices[a['v1']],
                                        vertices[a['v2']],
                                        vertices[a['v3']],
                                    ]
                                )

                    mesh = cls(np.zeros(len(triangles), dtype=cls.dtype))
                    # pyrefly: ignore[missing-attribute]
                    mesh.vectors[:] = np.array(triangles)
                    # pyrefly: ignore[invalid-yield]
                    yield mesh

    @classmethod
    def from_ply_file(
        cls,
        filename: str,
        calculate_normals: bool = True,
        fh: 'IO[bytes] | None' = None,
        **kwargs: Any,
    ) -> 'Self':
        """Load a mesh from a PLY file.

        Supports ASCII and binary PLY formats
        (little-endian and big-endian).

        Args:
            filename: Path to the .ply file.
            calculate_normals: Whether to recalculate
                normals. Defaults to True.
            fh: Optional pre-opened binary file handle.
            **kwargs: Additional arguments passed to
                the Mesh constructor.

        Returns:
            A Mesh instance.

        Example:
            >>> from stl import mesh
            >>> m = mesh.Mesh.from_ply_file('tests/ply_ascii/Cube.ply')
            >>> len(m.data) == 12
            True
        """
        from .ply import read_ply

        if fh:
            data, name = read_ply(fh, cls.dtype)
        else:
            with open(filename, 'rb') as fh:
                data, name = read_ply(fh, cls.dtype)

        return cls(
            data,
            calculate_normals,
            name=name,
            **kwargs,
        )


if TYPE_CHECKING:

    def StlMesh(  # noqa: N802
        filename: str,
        calculate_normals: bool = True,
        fh: 'IO[bytes] | None' = None,
        mode: Mode = Mode.AUTOMATIC,
        speedups: bool = True,
        **kwargs: Any,
    ) -> BaseStl: ...

else:
    StlMesh = BaseStl.from_file
