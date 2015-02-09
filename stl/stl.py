import os
import numpy
import struct
import datetime

from . import base
from . import metadata

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
        if mode in (AUTOMATIC, ASCII) and header.startswith('solid'):
            try:
                data = cls._load_ascii(fh, header)
            except RuntimeError, (recoverable, e):
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

        return data

    @classmethod
    def _load_binary(cls, fh, header, check_size=False):
        # Read the triangle count
        count, = struct.unpack('@i', fh.read(COUNT_SIZE))
        assert count < MAX_COUNT, ('File too large, got %d triangles which '
                                   'exceeds the maximum of %d') % (
                                       count, MAX_COUNT)

        if check_size:
            try:
                # Check the size of the file
                fh.seek(0, os.SEEK_END)
                raw_size = fh.tell() - HEADER_SIZE - COUNT_SIZE
                expected_count = raw_size / cls.dtype.itemsize
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
        lines = header.split('\n')
        recoverable = [True]

        def get(prefix=''):
            if lines:
                line = lines.pop(0)
            else:
                raise RuntimeError(recoverable[0], 'Unable to find more lines')
            if not lines:
                recoverable[0] = False

                # Read more lines and make sure we prepend any old data
                lines[:] = fh.read(BUFFER_SIZE).split('\n')
                line += lines.pop(0)

            line = line.lower().strip()
            if prefix:
                if line.startswith(prefix):
                    values = line.replace(prefix, '', 1).strip().split()
                elif line.startswith('endsolid'):
                    raise StopIteration
                else:
                    raise RuntimeError(recoverable[0],
                                       '%r should start with %r' % (line,
                                                                    prefix))

                if len(values) == 3:
                    return [float(v) for v in values]
                else:  # pragma: no cover
                    raise RuntimeError(recoverable[0],
                                       'Incorrect value %r' % line)
            else:
                return line

        line = get()
        if not line.startswith('solid ') and line.startswith('solid'):
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
                assert get() == 'outer loop'
                v0 = get('vertex')
                v1 = get('vertex')
                v2 = get('vertex')
                assert get() == 'endloop'
                assert get() == 'endfacet'
                attrs = 0
                yield (normals, (v0, v1, v2), attrs)
            except AssertionError, e:
                raise RuntimeError(recoverable[0], e)

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
            raise ValueError('Mode %r is invalid' % mode)

        name = os.path.split(filename)[-1]
        try:
            if fh:
                write(fh, name)
            else:
                with open(name, 'wb') as fh:
                    write(fh, filename)
        except IOError:  # pragma: no cover
            pass

    def _write_ascii(self, fh, name):
        print >>fh, 'solid %s' % name

        for row in self.data:
            vectors = row['vectors']
            print >>fh, 'facet normal %f %f %f' % tuple(row['normals'])
            print >>fh, '  outer loop'
            print >>fh, '    vertex %f %f %f' % tuple(vectors[0])
            print >>fh, '    vertex %f %f %f' % tuple(vectors[1])
            print >>fh, '    vertex %f %f %f' % tuple(vectors[2])
            print >>fh, '  endloop'
            print >>fh, 'endfacet'

        print >>fh, 'endsolid %s' % name

    def _write_binary(self, fh, name):
        fh.write(('%s (%s) %s %s' % (
            metadata.__package_name__,
            metadata.__version__,
            datetime.datetime.now(),
            name,
        ))[:80].ljust(80, ' '))
        fh.write(struct.pack('@i', self.data.size))
        self.data.tofile(fh)

    @classmethod
    def from_file(cls, filename, calculate_normals=True, fh=None, **kwargs):
        '''Load a mesh from a STL file

        :param str filename: The file to load
        :param bool calculate_normals: Whether to update the normals
        :param file fh: The file handle to open
        :param dict **kwargs: The same as for :py:class:`stl.mesh.Mesh`

        '''
        if fh:
            data = cls.load(fh)
        else:
            with open(filename, 'rb') as fh:
                data = cls.load(fh)

        return cls(data, calculate_normals, **kwargs)


StlMesh = BaseStl.from_file

