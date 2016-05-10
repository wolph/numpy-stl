from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import io
import os
import numpy
import struct
import datetime

from . import base
from . import __about__ as metadata
from .utils import b
from .utils import s

#: Automatically detect whether the output is a TTY, if so, write ASCII
#: otherwise write BINARY
AUTOMATIC = 0
#: Force writing ASCII
ASCII = 1
#: Force writing BINARY
BINARY = 2

#: Amount of bytes to read while using buffered reading
BUFFER_SIZE = 4096
#: The amount of bytes in the header field
HEADER_SIZE = 80
#: The amount of bytes in the count field
COUNT_SIZE = 4
#: The maximum amount of triangles we can read from binary files
MAX_COUNT = 1e6


class BaseStl(base.BaseMesh):
    @classmethod
    def load(cls, fh, mode=AUTOMATIC):
        '''Load Mesh from STL file

        Automatically detects binary versus ascii STL files.

        :param file fh: The file handle to open
        :param int mode: Automatically detect the filetype or force binary
        '''
        header = fh.read(HEADER_SIZE).lower()
        if not header.strip():
            return

        if isinstance(header, str):  # pragma: no branch
            header = b(header)

        name = ''

        if mode in (AUTOMATIC, ASCII) and header.startswith(b('solid')):
            try:
                data = cls._load_ascii(fh, header)

                # Get the name from the first line
                name = header.split(b('\n'), 1)[0][5:].strip()
            except RuntimeError as exception:
                (recoverable, e) = exception.args
                if recoverable:  # Recoverable?
                    data = cls._load_binary(fh, header, check_size=False)
                else:
                    # Apparently we've read beyond the header. Let's try
                    # seeking :)
                    # Note that this fails when reading from stdin, we can't
                    # recover from that.
                    fh.seek(HEADER_SIZE)

                    # Since we know this is a seekable file now and we're not
                    # 100% certain it's binary, check the size while reading
                    data = cls._load_binary(fh, header, check_size=True)
        else:
            data = cls._load_binary(fh, header)

        return name, data

    @classmethod
    def _load_binary(cls, fh, header, check_size=False):
        # Read the triangle count
        count, = struct.unpack(s('@i'), b(fh.read(COUNT_SIZE)))
        # raise RuntimeError()
        assert count < MAX_COUNT, ('File too large, got %d triangles which '
                                   'exceeds the maximum of %d') % (
                                       count, MAX_COUNT)

        if check_size:
            try:
                # Check the size of the file
                fh.seek(0, os.SEEK_END)
                raw_size = fh.tell() - HEADER_SIZE - COUNT_SIZE
                expected_count = int(raw_size / cls.dtype.itemsize)
                assert expected_count == count, ('Expected %d vectors but '
                                                 'header indicates %d') % (
                                                     expected_count, count)
                fh.seek(HEADER_SIZE + COUNT_SIZE)
            except IOError:  # pragma: no cover
                pass

        # Read the rest of the binary data
        return numpy.fromfile(fh, dtype=cls.dtype, count=count)

    @classmethod
    def _ascii_reader(cls, fh, header):
        lines = b(header).split(b('\n'))
        recoverable = [True]

        def get(prefix=''):
            prefix = b(prefix)

            if lines:
                line = lines.pop(0)
            else:
                raise RuntimeError(recoverable[0], 'Unable to find more lines')
            if not lines:
                recoverable[0] = False

                # Read more lines and make sure we prepend any old data
                lines[:] = b(fh.read(BUFFER_SIZE)).split(b('\n'))
                line += lines.pop(0)

            line = line.lower().strip()
            if line == b(''):
                return get(prefix)

            if prefix:
                if line.startswith(prefix):
                    values = line.replace(prefix, b(''), 1).strip().split()
                elif line.startswith(b('endsolid')):
                    raise StopIteration()
                else:
                    raise RuntimeError(recoverable[0],
                                       '{0!r} should start with {1!r}'.format(line,
                                                                    prefix))

                if len(values) == 3:
                    return [float(v) for v in values]
                else:  # pragma: no cover
                    raise RuntimeError(recoverable[0],
                                       'Incorrect value {0!r}'.format(line))
            else:
                return b(line)

        line = get()
        if not line.startswith(b('solid ')) and \
                line.startswith(b('solid')):
            cls.warning('ASCII STL files should start with solid <space>. '
                        'The application that produced this STL file may be '
                        'faulty, please report this error. The erroneous '
                        'line: %r', line)

        if not lines:
            raise RuntimeError(recoverable[0],
                               'No lines found, impossible to read')

        while True:
            # Read from the header lines first, until that point we can recover
            # and go to the binary option. After that we cannot due to
            # unseekable files such as sys.stdin
            #
            # Numpy doesn't support any non-file types so wrapping with a
            # buffer and/or StringIO does not work.
            try:
                normals = get('facet normal')
                assert get() == b('outer loop')
                v0 = get('vertex')
                v1 = get('vertex')
                v2 = get('vertex')
                assert get() == b('endloop')
                assert get() == b('endfacet')
                attrs = 0
                yield (normals, (v0, v1, v2), attrs)
            except AssertionError as e:
                raise RuntimeError(recoverable[0], e)
            except StopIteration:
                raise

    @classmethod
    def _load_ascii(cls, fh, header):
        return numpy.fromiter(cls._ascii_reader(fh, header), dtype=cls.dtype)

    def save(self, filename, fh=None, mode=AUTOMATIC, update_normals=True):
        '''Save the STL to a (binary) file

        If mode is :py:data:`AUTOMATIC` an :py:data:`ASCII` file will be
        written if the output is a TTY and a :py:data:`BINARY` file otherwise.

        :param str filename: The file to load
        :param file fh: The file handle to open
        :param int mode: The mode to write, default is :py:data:`AUTOMATIC`.
        :param bool update_normals: Whether to update the normals
        '''
        assert filename, 'Filename is required for the STL headers'
        if update_normals:
            self.update_normals()

        if mode is AUTOMATIC:
            if fh and os.isatty(fh.fileno()):  # pragma: no cover
                write = self._write_ascii
            else:
                write = self._write_binary
        elif mode is BINARY:
            write = self._write_binary
        elif mode is ASCII:
            write = self._write_ascii
        else:
            raise ValueError('Mode {0!r} is invalid'.format(mode))

        name = os.path.split(filename)[-1]
        try:
            if fh:
                write(fh, name)
            else:
                with open(filename, 'wb') as fh:
                    write(fh, filename)
        except IOError:  # pragma: no cover
            pass

    def _write_ascii(self, fh, name):
        def p(s, file):
            file.write(b('{0!s}\n'.format(s)))

        p('solid {0!s}'.format(name), file=fh)

        for row in self.data:
            vectors = row['vectors']
            p('facet normal {0:f} {1:f} {2:f}'.format(*tuple(row['normals'])), file=fh)
            p('  outer loop', file=fh)
            p('    vertex {0:f} {1:f} {2:f}'.format(*tuple(vectors[0])), file=fh)
            p('    vertex {0:f} {1:f} {2:f}'.format(*tuple(vectors[1])), file=fh)
            p('    vertex {0:f} {1:f} {2:f}'.format(*tuple(vectors[2])), file=fh)
            p('  endloop', file=fh)
            p('endfacet', file=fh)

        p('endsolid {0!s}'.format(name), file=fh)

    def _write_binary(self, fh, name):
        # Create the header
        header = '{0!s} ({1!s}) {2!s} {3!s}'.format(
            metadata.__package_name__,
            metadata.__version__,
            datetime.datetime.now(),
            name
        )

        # Make it exactly 80 characters
        header = header[:80].ljust(80, ' ')
        packed = struct.pack('@i', self.data.size)

        if isinstance(fh, io.TextIOWrapper):  # pragma: no cover
            packed = str(packed)
        else:
            header = b(header)
            packed = b(packed)

        fh.write(header)
        fh.write(packed)
        self.data.tofile(fh)

    @classmethod
    def from_file(cls, filename, calculate_normals=True, fh=None,
                  mode=AUTOMATIC, **kwargs):
        '''Load a mesh from a STL file

        :param str filename: The file to load
        :param bool calculate_normals: Whether to update the normals
        :param file fh: The file handle to open
        :param dict \**kwargs: The same as for :py:class:`stl.mesh.Mesh`

        '''
        if fh:
            name, data = cls.load(fh, mode=mode)
        else:
            with open(filename, 'rb') as fh:
                name, data = cls.load(fh, mode=mode)

        return cls(data, calculate_normals, name=name, **kwargs)

    @classmethod
    def from_multi_file(cls, filename, calculate_normals=True, fh=None,
                        mode=ASCII, **kwargs):
        '''Load multiple meshes from a STL file

        :param str filename: The file to load
        :param bool calculate_normals: Whether to update the normals
        :param file fh: The file handle to open
        :param dict \**kwargs: The same as for :py:class:`stl.mesh.Mesh`
        '''
        if fh:
            close = False
        else:
            fh = open(filename, 'rb')
            close = True

        try:
            raw_data = cls.load(fh, mode=mode)
            while raw_data:
                name, data = raw_data
                yield cls(data, calculate_normals, name=name, **kwargs)
                raw_data = cls.load(fh, mode=mode)

        finally:
            if close:
                fh.close()


StlMesh = BaseStl.from_file

