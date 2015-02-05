import os
import numpy
import struct
from python_utils import logger

from . import mesh
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


class StlMesh(mesh.Mesh):
    def __init__(self, filename, calculate_normals=True, fh=None, **kwargs):
        '''Load a mesh from a STL file

        :param str filename: The file to load
        :param bool calculate_normals: Whether to update the normals
        :param file fh: The file handle to open
        :param dict **kwargs: The same as for :py:class:`stl.mesh.Mesh`

        '''
        logger.Logged.__init__(self)
        self.filename = filename
        if fh:
            data = self.load(fh)
        else:
            with open(filename, 'rb') as fh:
                data = self.load(fh)

        mesh.Mesh.__init__(self, data, calculate_normals, **kwargs)

    def load(self, fh, mode=AUTOMATIC):
        '''Load Mesh from STL file

        Automatically detects binary versus ascii STL files.

        :param file fh: The file handle to open
        :param int mode: Automatically detect the filetype or force binary
        '''
        header = fh.read(HEADER_SIZE).lower()
        if mode in (AUTOMATIC, ASCII) and header.startswith('solid'):
            try:
                data = self._load_ascii(fh, header)
            except RuntimeError, (recoverable, e):
                if recoverable:  # Recoverable?
                    data = self._load_binary(fh, header, check_size=False)
                else:
                    # Apparently we've read beyond the header. Let's try
                    # seeking :)
                    # Note that this fails when reading from stdin, we can't
                    # recover from that.
                    fh.seek(HEADER_SIZE)

                    # Since we know this is a seekable file now and we're not
                    # 100% certain it's binary, check the size while reading
                    data = self._load_binary(fh, header, check_size=True)
        else:
            data = self._load_binary(fh, header)

        return data

    def _load_binary(self, fh, header, check_size=False):
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
                expected_count = raw_size / self.dtype.itemsize
                assert expected_count == count, ('Expected %d vectors but '
                                                 'header indicates %d') % (
                                                     expected_count, count)
                fh.seek(HEADER_SIZE + COUNT_SIZE)
            except IOError:  # pragma: no cover
                pass

        # Read the rest of the binary data
        return numpy.fromfile(fh, dtype=self.dtype, count=count)

    def _ascii_reader(self, fh, header):
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
            self.warning('ASCII STL files should start with solid <space>. '
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

    def _load_ascii(self, fh, header):
        return numpy.fromiter(self._ascii_reader(fh, header), dtype=self.dtype)
