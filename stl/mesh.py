import numpy
import collections

from python_utils import logger

VECTORS = 3
X = 0
Y = 1
Z = 2


class Mesh(logger.Logged, collections.Mapping):
    '''
    Mesh object with easy access to the vectors through v0, v1 and v2. An

    :param numpy.array data: The data for this mesh
    :param bool calculate_normals: Whehter to calculate the normals

    >>> data = numpy.zeros(10, dtype=Mesh.dtype)
    >>> mesh = Mesh(data)
    >>> # Increment vector 0 item 0
    >>> mesh.v0[0] += 1
    >>> mesh.v1[0] += 2

    # Check item 0 (contains v0, v1 and v2)
    >>> mesh[0]
    array([ 1.,  1.,  1.,  2.,  2.,  2.,  0.,  0.,  0.], dtype=float32)
    >>> mesh.vectors[0]
    array([[ 1.,  1.,  1.],
           [ 2.,  2.,  2.],
           [ 0.,  0.,  0.]], dtype=float32)
    >>> mesh.v0[0]
    array([ 1.,  1.,  1.], dtype=float32)
    >>> mesh.points[0]
    array([ 1.,  1.,  1.,  2.,  2.,  2.,  0.,  0.,  0.], dtype=float32)
    >>> mesh.data[0]
    ([0.0, 0.0, 0.0], [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.0, 0.0, 0.0]], [0])
    >>> mesh.x[0]
    array([ 1.,  2.,  0.], dtype=float32)

    >>> mesh[0] = 3
    >>> mesh[0]
    array([ 3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.], dtype=float32)

    >>> len(mesh) == len(list(mesh))
    True
    >>> (mesh.min_ < mesh.max_).all()
    True
    >>> mesh.update_normals()
    >>> mesh.units.sum()
    nan
    >>> mesh.v0 = mesh.v1 = mesh.v2 = 0
    >>> mesh.points.sum()
    0.0
    '''
    dtype = numpy.dtype([
        ('normals', numpy.float32, (3, )),
        ('vectors', numpy.float32, (3, 3)),
        ('attr', 'u2', (1, )),
    ])

    def __init__(self, data, calculate_normals=True):
        super(Mesh, self).__init__()
        self.data = data

        points = self.points = data['vectors']
        self.points.shape = data.size, 9
        self.x = points[:, X::3]
        self.y = points[:, Y::3]
        self.z = points[:, Z::3]
        # self.v0 = data['vectors'][:, 0]
        # self.v1 = data['vectors'][:, 1]
        # self.v2 = data['vectors'][:, 2]

        if calculate_normals:
            self.update_normals()

    def update_normals(self):
        '''Update the normals for all points'''
        self.normals = numpy.cross(self.v1 - self.v0, self.v2 - self.v0)

    def update_min(self):
        self._min = numpy.min((
            self.v0.min(axis=0),
            self.v1.min(axis=0),
            self.v2.min(axis=0),
        ), axis=0)

    def update_max(self):
        self._max = numpy.max((
            self.v0.max(axis=0),
            self.v1.max(axis=0),
            self.v2.max(axis=0),
        ), axis=0)

    def update_areas(self):
        areas = numpy.sqrt((self.normals ** 2).sum(axis=1))
        self.areas = areas.reshape((areas.size, 1))

    def update_units(self):
        self.units = self.normals / numpy.hstack((self.areas, self.areas,
                                                  self.areas))

    def _get_or_update(key):
        def _get(self):
            if not hasattr(self, '_%s' % key):
                getattr(self, 'update_%s' % key)()
            return getattr(self, '_%s' % key)

        return _get

    def _set(key):
        def _set(self, value):
            setattr(self, '_%s' % key, value)

        return _set

    min_ = property(_get_or_update('min'), _set('min'),
                    doc='Mesh minimum value')
    max_ = property(_get_or_update('max'), _set('max'),
                    doc='Mesh maximum value')
    areas = property(_get_or_update('areas'), _set('areas'),
                     doc='Mesh areas')
    units = property(_get_or_update('units'), _set('units'),
                     doc='Mesh unit vectors')

    def __getitem__(self, k):
        return self.points[k]

    def __setitem__(self, k, v):
        self.points[k] = v

    def __len__(self):
        return self.points.shape[0]

    def _get(k):
        def __get(self):
            return self.data[k]
        return __get

    def _set(k):
        def __set(self, v):
            self.data[k] = v
        return __set

    def _get_vector(n):
        def __get(self):
            return self.vectors[:, n]
        return __get

    def _set_vector(n):
        def __set(self, v):
            self.vectors[:, n] = v
        return __set

    def __iter__(self):
        for point in self.points:
            yield point

    normals = property(_get('normals'), _set('normals'), doc='Normals')
    vectors = property(_get('vectors'), _set('vectors'), doc='Vectors')
    attr = property(_get('attr'), _set('attr'), doc='Attributes')
    v0 = property(_get_vector(0), _set_vector(0), doc='Normals')
    v1 = property(_get_vector(1), _set_vector(1), doc='Normals')
    v2 = property(_get_vector(2), _set_vector(2), doc='Normals')

