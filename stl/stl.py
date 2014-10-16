import os
import numpy
import struct
import datetime

from . import mesh
from . import metadata

#: Automatically detect whether the output is a TTY, if so, write ASCII
#: otherwise write BINARY
AUTOMATIC = 0
#: Force writing ASCII
ASCII = 1
#: Force writing BINARY
BINARY = 2


class StlMesh(mesh.Mesh):
    '''Load a mesh from a STL file

    :param str filename: The file to load
    :param bool calculate_normals: Whether to calculate the normals
    :param file fh: The file handle to open
    '''
    def __init__(self, filename, calculate_normals=True, fh=None):
        self.filename = filename
        if fh:
            data = self.load(fh)
        else:
            with open(filename, 'rb') as fh:
                data = self.load(fh)

        mesh.Mesh.__init__(self, data, calculate_normals)

    def load(self, fh):
        '''Load Mesh from STL file

        Automatically detects binary versus ascii STL files.

        :param file fh: The file handle to open
        '''
        begin = fh.read(5).lower()
        fh.seek(0)
        if begin.startswith('solid'):
            data = self._load_ascii(fh)
        else:
            data = self._load_binary(fh)

        return data

    def _load_binary(self, fh):
        # Skip the header
        fh.seek(80)
        # Read the size
        size, = struct.unpack('@i', fh.read(4))
        # Read the rest of the binary data
        return numpy.fromfile(fh, dtype=self.dtype, count=size)

    def _load_ascii(self, fh):
        assert fh.next().startswith('solid')

        def get(fh, prefix=''):
            line = line = fh.next().lower().strip()
            if prefix:
                values = line.replace(prefix, '', 1).strip().split()
                if len(values) == 3:
                    return [float(v) for v in values]
                elif values[0] == 'endsolid':
                    raise StopIteration
                else:  # pragma: no cover
                    raise ValueError('Incorrect value %r' % line)
            else:
                return line

        def read_facet(fh):
            while fh:
                data = []
                data.append(get(fh, 'facet normal'))

                assert get(fh) == 'outer loop'
                data.append(get(fh, 'vertex'))
                data.append(get(fh, 'vertex'))
                data.append(get(fh, 'vertex'))
                assert get(fh) == 'endloop'
                assert get(fh) == 'endfacet'

                # Attributes are currently not supported
                data.append(0)
                yield tuple(data)

        return numpy.fromiter(read_facet(fh), dtype=self.dtype)

    def save(self, filename, fh=None, mode=AUTOMATIC, calculate_normals=True):
        '''Save the STL to a (binary) file

        If mode is :py:data:`AUTOMATIC` an :py:data:`ASCII` file will be
        written if the output is a TTY and a :py:data:`BINARY` file otherwise.

        :param str filename: The file to load
        :param file fh: The file handle to open
        :param int mode: The mode to write, default is :py:data:`AUTOMATIC`.
        :param bool calculate_normals: Whether to calculate the normals
        '''
        if calculate_normals:
            self.calculate_normals()

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
        if fh:
            write(fh, name)
        else:
            with open(name, 'wb') as fh:
                write(fh, filename)

    def _write_ascii(self, fh, name):
        print >>fh, 'solid %s' % name

        for row in self.data:
            print >>fh, 'facet normal %f %f %f' % tuple(row['normals'])
            print >>fh, '  outer loop'
            print >>fh, '    vertex %f %f %f' % tuple(row['v0'])
            print >>fh, '    vertex %f %f %f' % tuple(row['v1'])
            print >>fh, '    vertex %f %f %f' % tuple(row['v2'])
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
