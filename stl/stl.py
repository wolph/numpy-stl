import datetime
import enum
import io
import os
import struct
import zipfile
from xml.etree import ElementTree

import numpy

from . import __about__ as metadata, base
from .utils import b

try:
    from . import _speedups
except ImportError:  # pragma: no cover
    _speedups = None


class Mode(enum.IntEnum):
    #: Automatically detect whether the output is a TTY, if so, write ASCII
    #: otherwise write BINARY
    AUTOMATIC = 0
    #: Force writing ASCII
    ASCII = 1
    #: Force writing BINARY
    BINARY = 2


# For backwards compatibility, leave the original references
AUTOMATIC = Mode.AUTOMATIC
ASCII = Mode.ASCII
BINARY = Mode.BINARY

#: Amount of bytes to read while using buffered reading
BUFFER_SIZE = 4096
#: The amount of bytes in the header field
HEADER_SIZE = 80
#: The amount of bytes in the count field
COUNT_SIZE = 4
#: The maximum amount of triangles we can read from binary files
MAX_COUNT = 1e8
#: The header format, can be safely monkeypatched. Limited to 80 characters
HEADER_FORMAT = '{package_name} ({version}) {now} {name}'


class BaseStl(base.BaseMesh):
    @classmethod
    def load(cls, fh, mode=AUTOMATIC, speedups=True):
        """Load Mesh from STL file

        Automatically detects binary versus ascii STL files.

        :param file fh: The file handle to open
        :param int mode: Automatically detect the filetype or force binary
        """
        header = fh.read(HEADER_SIZE)
        if not header:
            return

        if isinstance(header, str):  # pragma: no branch
            header = b(header)

        if mode is AUTOMATIC:
            if header.lstrip().lower().startswith(b'solid'):
                try:
                    name, data = cls._load_ascii(fh, header, speedups=speedups)
                except RuntimeError as exception:
                    print('exception', exception)
                    (recoverable, e) = exception.args
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
    def _load_binary(cls, fh, header, check_size=False):
        # Read the triangle count
        count_data = fh.read(COUNT_SIZE)
        if len(count_data) != COUNT_SIZE:
            count = 0
        else:
            (count,) = struct.unpack('<i', b(count_data))
        # raise RuntimeError()
        assert count < MAX_COUNT, (
            'File too large, got %d triangles which '
            'exceeds the maximum of %d'
        ) % (count, MAX_COUNT)

        if check_size:
            try:
                # Check the size of the file
                fh.seek(0, os.SEEK_END)
                raw_size = fh.tell() - HEADER_SIZE - COUNT_SIZE
                expected_count = int(raw_size / cls.dtype.itemsize)
                assert expected_count == count, (
                    'Expected %d vectors but ' 'header indicates %d'
                ) % (expected_count, count)
                fh.seek(HEADER_SIZE + COUNT_SIZE)
            except IOError:  # pragma: no cover
                pass

        name = header.strip()

        # Read the rest of the binary data
        try:
            return name, numpy.fromfile(fh, dtype=cls.dtype, count=count)
        except io.UnsupportedOperation:
            data = numpy.frombuffer(fh.read(), dtype=cls.dtype, count=count)
            # Copy to make the buffer writable
            return name, data.copy()

    @classmethod
    def _ascii_reader(cls, fh, header):
        if b'\n' in header:
            recoverable = [True]
        else:
            recoverable = [False]
            header += b(fh.read(BUFFER_SIZE))

        lines = b(header).split(b'\n')

        def get(prefix=''):
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
                elif line.startswith(b('endsolid')) or line.startswith(
                    b('end solid')
                ):
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
                        '%r should start with %r' % (line, prefix),
                    )

                if len(values) == 3:
                    return [float(v) for v in values]
                else:  # pragma: no cover
                    raise RuntimeError(
                        recoverable[0], 'Incorrect value %r' % line
                    )
            else:
                return b(raw_line)

        line = get()
        if not lines:
            raise RuntimeError(
                recoverable[0], 'No lines found, impossible to read'
            )

        # Yield the name
        yield line[5:].strip()

        while True:
            # Read from the header lines first, until that point we can recover
            # and go to the binary option. After that we cannot due to
            # unseekable files such as sys.stdin
            #
            # Numpy doesn't support any non-file types so wrapping with a
            # buffer and/or StringIO does not work.
            try:
                normals = get('facet normal')
                assert get().lower() == b('outer loop')
                v0 = get('vertex')
                v1 = get('vertex')
                v2 = get('vertex')
                assert get().lower() == b('endloop')
                assert get().lower() == b('endfacet')
                attrs = 0
                yield (normals, (v0, v1, v2), attrs)
            except AssertionError as e:  # pragma: no cover
                raise RuntimeError(recoverable[0], e)
            except StopIteration:
                return

    @classmethod
    def _load_ascii(cls, fh, header, speedups=True):
        # Speedups does not support non file-based streams
        try:
            fh.fileno()
        except io.UnsupportedOperation:
            speedups = False
        # The speedups module is covered by travis but it can't be tested in
        # all environments, this makes coverage checks easier
        if _speedups and speedups:  # pragma: no cover
            return _speedups.ascii_read(fh, header)
        else:
            iterator = cls._ascii_reader(fh, header)
            name = next(iterator)
            return name, numpy.fromiter(iterator, dtype=cls.dtype)

    def save(self, filename, fh=None, mode=AUTOMATIC, update_normals=True):
        """Save the STL to a (binary) file

        If mode is :py:data:`AUTOMATIC` an :py:data:`ASCII` file will be
        written if the output is a TTY and a :py:data:`BINARY` file otherwise.

        :param str filename: The file to load
        :param file fh: The file handle to open
        :param int mode: The mode to write, default is :py:data:`AUTOMATIC`.
        :param bool update_normals: Whether to update the normals
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
                except IOError:
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
            raise ValueError('Mode %r is invalid' % mode)

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
        except IOError:  # pragma: no cover
            pass

    def _write_ascii(self, fh, name):
        try:
            fh.fileno()
            speedups = self.speedups
        except io.UnsupportedOperation:
            speedups = False

        if _speedups and speedups:  # pragma: no cover
            _speedups.ascii_write(fh, b(name), self.data)
        else:

            def p(s, file):
                file.write(b('%s\n' % s))

            p('solid %s' % name, file=fh)

            for row in self.data:
                vectors = row['vectors']
                p(
                    'facet normal %r %r %r' % tuple(row['normals'].tolist()),
                    file=fh,
                )
                p('  outer loop', file=fh)
                p('    vertex %r %r %r' % tuple(vectors[0].tolist()), file=fh)
                p('    vertex %r %r %r' % tuple(vectors[1].tolist()), file=fh)
                p('    vertex %r %r %r' % tuple(vectors[2].tolist()), file=fh)
                p('  endloop', file=fh)
                p('endfacet', file=fh)

            p('endsolid %s' % name, file=fh)

    def get_header(self, name):
        # Format the header
        header = HEADER_FORMAT.format(
            package_name=metadata.__package_name__,
            version=metadata.__version__,
            now=datetime.datetime.now(),
            name=name,
        )

        # Make it exactly 80 characters
        return header[:80].ljust(80, ' ')

    def _write_binary(self, fh, name):
        header = self.get_header(name)
        packed = struct.pack('<i', self.data.size)

        if isinstance(fh, io.TextIOWrapper):  # pragma: no cover
            packed = str(packed)
        else:
            header = b(header)
            packed = b(packed)

        fh.write(header)
        fh.write(packed)

        if isinstance(fh, io.BufferedWriter):
            # Write to a true file.
            self.data.tofile(fh)
        else:
            # Write to a pseudo buffer.
            fh.write(self.data.data)

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
        filename,
        calculate_normals=True,
        fh=None,
        mode=Mode.AUTOMATIC,
        speedups=True,
        **kwargs,
    ):
        """Load a mesh from a STL file

        :param str filename: The file to load
        :param bool calculate_normals: Whether to update the normals
        :param file fh: The file handle to open
        :param dict kwargs: The same as for :py:class:`stl.mesh.Mesh`

        """
        if fh:
            name, data = cls.load(fh, mode=mode, speedups=speedups)
        else:
            with open(filename, 'rb') as fh:
                name, data = cls.load(fh, mode=mode, speedups=speedups)

        return cls(
            data, calculate_normals, name=name, speedups=speedups, **kwargs
        )

    @classmethod
    def from_multi_file(
        cls,
        filename,
        calculate_normals=True,
        fh=None,
        mode=Mode.AUTOMATIC,
        speedups=True,
        **kwargs,
    ):
        """Load multiple meshes from a STL file

        Note: mode is hardcoded to ascii since binary stl files do not support
        the multi format

        :param str filename: The file to load
        :param bool calculate_normals: Whether to update the normals
        :param file fh: The file handle to open
        :param dict kwargs: The same as for :py:class:`stl.mesh.Mesh`
        """
        if fh:
            close = False
        else:
            fh = open(filename, 'rb')
            close = True

        try:
            raw_data = cls.load(fh, mode=mode, speedups=speedups)
            while raw_data:
                name, data = raw_data
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
        filenames,
        calculate_normals=True,
        mode=Mode.AUTOMATIC,
        speedups=True,
        **kwargs,
    ):
        """Load multiple meshes from STL files into a single mesh

        Note: mode is hardcoded to ascii since binary stl files do not support
        the multi format

        :param list(str) filenames: The files to load
        :param bool calculate_normals: Whether to update the normals
        :param file fh: The file handle to open
        :param dict kwargs: The same as for :py:class:`stl.mesh.Mesh`
        """
        meshes = []
        for filename in filenames:
            meshes.append(
                cls.from_file(
                    filename,
                    calculate_normals=calculate_normals,
                    mode=mode,
                    speedups=speedups,
                    **kwargs,
                )
            )

        data = numpy.concatenate([mesh.data for mesh in meshes])
        return cls(data, calculate_normals=calculate_normals, **kwargs)

    @classmethod
    def from_3mf_file(cls, filename, calculate_normals=True, **kwargs):
        with zipfile.ZipFile(filename) as zip:
            with zip.open('_rels/.rels') as rels_fh:
                model = None
                root = ElementTree.parse(rels_fh).getroot()
                for child in root:  # pragma: no branch
                    type_ = child.attrib.get('Type', '')
                    if type_.endswith('3dmodel'):  # pragma: no branch
                        model = child.attrib.get('Target', '')
                        break

            assert model, 'No 3D model found in %s' % filename
            with zip.open(model.lstrip('/')) as fh:
                root = ElementTree.parse(fh).getroot()

                elements = root.findall('./{*}resources/{*}object/{*}mesh')
                for mesh_element in elements:  # pragma: no branch
                    triangles = []
                    vertices = []

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

                    mesh = cls(numpy.zeros(len(triangles), dtype=cls.dtype))
                    mesh.vectors[:] = numpy.array(triangles)
                    yield mesh


StlMesh = BaseStl.from_file
