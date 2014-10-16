import numpy
import itertools
import collections

from python_utils import logger

VECTORS = 3


class Mesh(logger.Logged, collections.Mapping):
    '''
    Mesh object with easy access to the vectors through v0, v1 and v2. An

    :param numpy.array data: The data for this mesh
    :param bool calculate_normals: Whehter to calculate the normals

    >>> data = numpy.zeros(10, dtype=Mesh.dtype)
    >>> mesh = Mesh(data)
    >>> mesh.v0 += 1
    >>> mesh[0]
    array([ 1.,  1.,  1.], dtype=float32)
    >>> mesh[1]
    array([ 0.,  0.,  0.], dtype=float32)
    >>> mesh[0] += 1
    >>> mesh.v0[0]
    array([ 2.,  2.,  2.], dtype=float32)
    >>> len(mesh) == len(list(mesh))
    True
    '''
    dtype = numpy.dtype([
        ('normals', numpy.float32, (3, )),
        ('v0', numpy.float32, (3, )),
        ('v1', numpy.float32, (3, )),
        ('v2', numpy.float32, (3, )),
        ('attr', 'u2', (1, )),
    ])

    def __init__(self, data, calculate_normals=True):
        super(Mesh, self).__init__()
        self.data = data
        if calculate_normals:
            self.calculate_normals()

    def __getitem__(self, k):
        return self.data['v%d' % (k % VECTORS)][k / VECTORS]

    def __setitem__(self, k, v):
        self.data['v%d' % (k % VECTORS)][k / VECTORS] = v

    def __len__(self):
        return self.data.size * VECTORS

    def _get(k):
        def __get(self):
            return self.data[k]
        return __get

    def _set(k):
        def __set(self, v):
            self.data[k] = v
        return __set

    normals = property(_get('normals'), _set('normals'), doc='Normals')
    v0 = property(_get('v0'), _set('v0'), doc='Vector 0')
    v1 = property(_get('v1'), _set('v1'), doc='Vector 1')
    v2 = property(_get('v2'), _set('v2'), doc='Vector 2')
    attr = property(_get('attr'), _set('attr'), doc='Attributes')

    def calculate_normals(self):
        '''(re)calculate the normals for all points'''
        self.normals = numpy.cross(self.v1 - self.v0, self.v2 - self.v0)

    def __iter__(self):
        for v0, v1, v2 in itertools.izip(self.v0, self.v1, self.v2):
            yield v0
            yield v1
            yield v2

