import enum
import itertools
import logging
import math
from collections import abc
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    Literal as L,  # noqa: N817
    SupportsIndex,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import numpy.typing as npt
from python_utils import logger

if TYPE_CHECKING:  # pragma: no cover
    from types import EllipsisType
    from typing import Protocol, TypeAlias

    # this won't be changing anytime soon, so safe to import here
    from numpy._typing import _ArrayLikeFloat_co, _ArrayLikeInt_co

    # pyrefly: ignore[invalid-inheritance]
    class _Logged(logger.LoggerProtocol, Protocol):  # pragma: no cover
        logger: logging.Logger

    _Dedupe: 'TypeAlias' = 'RemoveDuplicates | int'
    _ToAxis: 'TypeAlias' = npt.NDArray[np.integer] | abc.Sequence[int]
    _ToPoint: 'TypeAlias' = (
        float | abc.Sequence[float] | npt.NDArray[np.floating | np.integer]
    )
    _ToTranslation: 'TypeAlias' = (
        abc.Sequence[_ToPoint] | npt.NDArray[np.floating | np.integer]
    )

    # same as used by `np.ndarray.__getitem__`
    _ToIndex: 'TypeAlias' = (
        SupportsIndex | slice | EllipsisType | _ArrayLikeInt_co | None
    )
    _ToIndices: 'TypeAlias' = _ToIndex | tuple[_ToIndex, ...]

    # specific to 2-d arrays
    _ToSlice2_0: 'TypeAlias' = tuple[SupportsIndex, SupportsIndex]
    _ToSlice2_1: 'TypeAlias' = (
        int
        | np.integer
        | tuple[slice | EllipsisType, int]
        | tuple[int, slice | EllipsisType]
    )
    _ToSlice2_2: 'TypeAlias' = (
        slice
        | tuple[()]
        | tuple[slice, slice]
        | list[int]
        | npt.NDArray[np.integer]
        | EllipsisType
    )

_bool_1d: 'TypeAlias' = np.ndarray[tuple[int], np.dtype[np.bool_]]
_intp_1d: 'TypeAlias' = np.ndarray[tuple[int], np.dtype[np.intp]]
_u16_1d: 'TypeAlias' = np.ndarray[tuple[int], np.dtype[np.uint16]]
_u16_2d: 'TypeAlias' = np.ndarray[tuple[int, int], np.dtype[np.uint16]]
_f32_1d: 'TypeAlias' = np.ndarray[tuple[int], np.dtype[np.float32]]
_f32_2d: 'TypeAlias' = np.ndarray[tuple[int, int], np.dtype[np.float32]]
_f32_3d: 'TypeAlias' = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
_f64_2d: 'TypeAlias' = np.ndarray[tuple[int, int], np.dtype[np.float64]]

# {"normals": _float32_1d, "vectors": _float32_2d, "attr": _uint16_1d}
_data_1d: 'TypeAlias' = np.ndarray[tuple[int], np.dtype[np.void]]

#: When removing empty areas, remove areas that are smaller than this
AREA_SIZE_THRESHOLD: Final[L[0]] = 0
#: Vectors in a point
VECTORS: Final[L[3]] = 3
#: Dimensions used in a vector
DIMENSIONS: Final[L[3]] = 3


class Dimension(enum.IntEnum):
    #: X index (for example, `mesh.v0[0][X]`)
    X = 0
    #: Y index (for example, `mesh.v0[0][Y]`)
    Y = 1
    #: Z index (for example, `mesh.v0[0][Z]`)
    Z = 2


# For backwards compatibility, leave the original references
X: Final[L[Dimension.X]] = Dimension.X
Y: Final[L[Dimension.Y]] = Dimension.Y
Z: Final[L[Dimension.Z]] = Dimension.Z


class RemoveDuplicates(enum.Enum):
    """
    Choose whether to remove no duplicates, leave only a single of the
    duplicates or remove all duplicates (leaving holes).
    """

    NONE = 0
    SINGLE = 1
    ALL = 2

    @classmethod
    def map(cls, /, value: '_Dedupe') -> 'RemoveDuplicates':
        if value is True:
            return cls.SINGLE
        elif value and value in cls:
            return cls(value)
        else:
            return cls.NONE


_LoggedT = TypeVar('_LoggedT', bound='_Logged')


def logged(class_: type[_LoggedT]) -> type[_LoggedT]:
    # For some reason the Logged baseclass is not properly initiated on Linux
    # systems while this works on OS X. Please let me know if you can tell me
    # what silly mistake I made here

    logger_name = cast(
        'str',
        logger.Logged._Logged__get_name(__name__, class_.__name__),  # type: ignore[attr-defined]
    )

    class_.logger = logging.getLogger(logger_name)

    for key in dir(logger.Logged):
        if not key.startswith('__'):
            setattr(class_, key, getattr(class_, key))

    return class_


@logged
class BaseMesh(logger.Logged, abc.Mapping['_ToIndices', np.ndarray]):
    """
    Mesh object with easy access to the vectors through v0, v1 and v2.
    The normals, areas, min, max and units are calculated automatically.

    :param numpy.array data: The data for this mesh
    :param bool calculate_normals: Whether to calculate the normals
    :param bool remove_empty_areas: Whether to remove triangles with 0 area
            (due to rounding errors for example)

    :ivar str name: Name of the solid, only exists in ASCII files
    :ivar numpy.array data: Data as :func:`BaseMesh.dtype`
    :ivar numpy.array points: All points (Nx9)
    :ivar numpy.array normals: Normals for this mesh, calculated automatically
        by default (Nx3)
    :ivar numpy.array vectors: Vectors in the mesh (Nx3x3)
    :ivar numpy.array attr: Attributes per vector (used by binary STL)
    :ivar numpy.array x: Points on the X axis by vertex (Nx3)
    :ivar numpy.array y: Points on the Y axis by vertex (Nx3)
    :ivar numpy.array z: Points on the Z axis by vertex (Nx3)
    :ivar numpy.array v0: Points in vector 0 (Nx3)
    :ivar numpy.array v1: Points in vector 1 (Nx3)
    :ivar numpy.array v2: Points in vector 2 (Nx3)

    >>> data = np.zeros(10, dtype=BaseMesh.dtype)
    >>> mesh = BaseMesh(data, remove_empty_areas=False)
    >>> # Increment vector 0 item 0
    >>> mesh.v0[0] += 1
    >>> mesh.v1[0] += 2

    >>> # Check item 0 (contains v0, v1 and v2)
    >>> assert np.array_equal(
    ...     mesh[0], np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0])
    ... )
    >>> assert np.array_equal(
    ...     mesh.vectors[0],
    ...     np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.0, 0.0, 0.0]]),
    ... )
    >>> assert np.array_equal(mesh.v0[0], np.array([1.0, 1.0, 1.0]))
    >>> assert np.array_equal(
    ...     mesh.points[0],
    ...     np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0]),
    ... )
    >>> assert np.array_equal(
    ...     mesh.data[0],
    ...     np.array(
    ...         (
    ...             [0.0, 0.0, 0.0],
    ...             [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.0, 0.0, 0.0]],
    ...             [0],
    ...         ),
    ...         dtype=BaseMesh.dtype,
    ...     ),
    ... )
    >>> assert np.array_equal(mesh.x[0], np.array([1.0, 2.0, 0.0]))

    >>> mesh[0] = 3
    >>> assert np.array_equal(
    ...     mesh[0], np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    ... )

    >>> len(mesh) == len(list(mesh))
    True
    >>> bool((mesh.min_ < mesh.max_).all())
    True
    >>> mesh.update_normals()
    >>> float(mesh.units.sum())
    0.0
    >>> mesh.v0[:] = mesh.v1[:] = mesh.v2[:] = 0
    >>> float(mesh.points.sum())
    0.0

    >>> mesh.v0 = mesh.v1 = mesh.v2 = 0
    >>> mesh.x = mesh.y = mesh.z = 0

    >>> mesh.attr = 1
    >>> bool((mesh.attr == 1).all())
    True

    >>> mesh.normals = 2
    >>> bool((mesh.normals == 2).all())
    True

    >>> mesh.vectors = 3
    >>> bool((mesh.vectors == 3).all())
    True

    >>> mesh.points = 4
    >>> bool((mesh.points == 4).all())
    True
    """

    #: - normals: :func:`numpy.float32`, `(3, )`
    #: - vectors: :func:`numpy.float32`, `(3, 3)`
    #: - attr: :func:`numpy.uint16`, `(1, )`
    dtype: ClassVar[np.dtype[np.void]] = np.dtype(
        [
            ('normals', np.float32, (3,)),
            ('vectors', np.float32, (3, 3)),
            ('attr', np.uint16, (1,)),
        ]
    ).newbyteorder('<')  # Even on big endian arches, use little e.

    speedups: Final[bool]
    name: Final['bytes | str']
    data: Final[_data_1d]

    _min: _f32_1d
    _max: _f32_1d
    _areas: _f32_2d
    _centroids: _f32_2d
    _units: _f32_2d

    def __init__(
        self,
        data: _data_1d,
        calculate_normals: bool = True,
        remove_empty_areas: bool = False,
        remove_duplicate_polygons: '_Dedupe' = RemoveDuplicates.NONE,
        name: 'bytes | str' = '',
        speedups: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.speedups = speedups

        if remove_empty_areas:
            data = self.remove_empty_areas(data)

        if (
            RemoveDuplicates.map(remove_duplicate_polygons)
            is not RemoveDuplicates.NONE
        ):
            data = self.remove_duplicate_polygons(
                data, remove_duplicate_polygons
            )

        self.name = name
        self.data = data

        if calculate_normals:
            self.update_normals()

    @property
    def attr(self) -> _u16_2d:
        # https://github.com/numpy/numpy/pull/30261
        return self.data['attr']  # type: ignore[return-value]

    @attr.setter
    def attr(self, value: '_ArrayLikeInt_co', /) -> None:
        self.data['attr'] = value

    @property
    def normals(self) -> _f32_2d:
        # https://github.com/numpy/numpy/pull/30261
        return self.data['normals']  # type: ignore[return-value]

    @normals.setter
    def normals(self, value: '_ArrayLikeFloat_co', /) -> None:
        self.data['normals'] = value

    @property
    def vectors(self) -> _f32_3d:
        # https://github.com/numpy/numpy/pull/30261
        return self.data['vectors']  # type: ignore[return-value]

    @vectors.setter
    def vectors(self, value: '_ArrayLikeFloat_co', /) -> None:
        self.data['vectors'] = value

    @property
    def points(self) -> _f32_2d:
        return self.vectors.reshape(self.data.size, 9)

    @points.setter
    def points(self, value: '_ArrayLikeFloat_co', /) -> None:
        self.points[:] = value

    @property
    def v0(self) -> _f32_2d:
        return self.vectors[:, 0]

    @v0.setter
    def v0(self, value: '_ArrayLikeFloat_co', /) -> None:
        self.vectors[:, 0] = value

    @property
    def v1(self) -> _f32_2d:
        return self.vectors[:, 1]

    @v1.setter
    def v1(self, value: '_ArrayLikeFloat_co', /) -> None:
        self.vectors[:, 1] = value

    @property
    def v2(self) -> _f32_2d:
        return self.vectors[:, 2]

    @v2.setter
    def v2(self, value: '_ArrayLikeFloat_co', /) -> None:
        self.vectors[:, 2] = value

    @property
    def x(self) -> _f32_2d:
        return self.points[:, Dimension.X :: 3]

    @x.setter
    def x(self, value: '_ArrayLikeFloat_co', /) -> None:
        self.points[:, Dimension.X :: 3] = value

    @property
    def y(self) -> _f32_2d:
        return self.points[:, Dimension.Y :: 3]

    @y.setter
    def y(self, value: '_ArrayLikeFloat_co', /) -> None:
        self.points[:, Dimension.Y :: 3] = value

    @property
    def z(self) -> _f32_2d:
        return self.points[:, Dimension.Z :: 3]

    @z.setter
    def z(self, value: '_ArrayLikeFloat_co', /) -> None:
        self.points[:, Dimension.Z :: 3] = value

    @staticmethod
    def remove_duplicate_polygons(
        data: _data_1d,
        value: '_Dedupe' = RemoveDuplicates.SINGLE,
    ) -> _data_1d:
        value = RemoveDuplicates.map(value)
        polygons: _f32_2d = data['vectors'].sum(axis=1)
        # Get a sorted list of indices
        idx: _intp_1d = np.lexsort(polygons.T)
        # Get the indices of all different indices
        diff: _bool_1d = np.any(
            polygons[idx[1:]] != polygons[idx[:-1]],
            axis=1,
        )

        if value is RemoveDuplicates.SINGLE:
            # Only return the unique data, the True is so we always get at
            # least the originals
            return data[np.sort(idx[np.concatenate(([True], diff))])]
        elif value is RemoveDuplicates.ALL:
            # We need to return both items of the shifted diff
            diff_a: _bool_1d = np.concatenate(([True], diff))
            diff_b: _bool_1d = np.concatenate((diff, [True]))
            diff = np.concatenate((diff, [False]))

            # Combine both unique lists
            filtered_data: _data_1d = data[np.sort(idx[diff_a & diff_b])]
            if len(filtered_data) <= len(data) / 2:
                return data[np.sort(idx[diff_a])]
            else:
                return data[np.sort(idx[diff])]
        else:
            return data

    @staticmethod
    def remove_empty_areas(data: _data_1d) -> _data_1d:
        # https://github.com/numpy/numpy/pull/30261
        vectors: _f32_3d = data['vectors']  # type: ignore[assignment]
        v0 = vectors[:, 0]
        v1 = vectors[:, 1]
        v2 = vectors[:, 2]
        normals = np.cross(v1 - v0, v2 - v0)
        squared_areas = (normals**2).sum(axis=1)
        return data[squared_areas > AREA_SIZE_THRESHOLD**2]

    def update_normals(
        self,
        update_areas: bool = True,
        update_centroids: bool = True,
    ) -> None:
        """Update the normals, areas, and centroids for all points"""
        normals: _f32_2d = np.cross(self.v1 - self.v0, self.v2 - self.v0)

        if update_areas:
            self.update_areas(normals)

        if update_centroids:
            self.update_centroids()

        self.normals[:] = normals

    def get_unit_normals(self) -> _f32_2d:
        normals = self.normals.copy()
        normal: _f32_1d = np.linalg.norm(normals, axis=1)
        non_zero = normal > 0
        if non_zero.any():
            normals[non_zero] /= normal[non_zero][:, None]
        return normals

    def update_min(self) -> None:
        self._min = self.vectors.min(axis=(0, 1))

    def update_max(self) -> None:
        self._max = self.vectors.max(axis=(0, 1))

    def update_areas(self, normals: '_f32_2d | None' = None) -> None:
        if normals is None:
            normals = np.cross(self.v1 - self.v0, self.v2 - self.v0)

        areas = 0.5 * np.sqrt((normals**2).sum(axis=1))
        self._areas = areas.reshape((areas.size, 1))

    def update_centroids(self) -> None:
        self._centroids = np.mean([self.v0, self.v1, self.v2], axis=0)

    def check(self, exact: bool = False) -> bool:
        """Check the mesh is valid or not

        :param bool exact: Perform exact checks.
        """
        return self.is_closed(exact=exact)

    def is_closed(self, exact: bool = False) -> bool:  # pragma: no cover
        """Check the mesh is closed or not

        :param bool exact: Perform a exact check on edges.
        """

        if exact:
            reversed_triangles: _bool_1d = (
                np.cross(self.v1 - self.v0, self.v2 - self.v0) * self.normals
            ).sum(axis=1) < 0
            directed_edges = {
                tuple(edge.ravel() if not rev else edge[::-1, :].ravel())
                for rev, edge in zip(
                    itertools.cycle(reversed_triangles),
                    itertools.chain(
                        self.vectors[:, (0, 1), :],
                        self.vectors[:, (1, 2), :],
                        self.vectors[:, (2, 0), :],
                    ),
                )
            }
            if len(directed_edges) == 3 * self.data.size:
                undirected_edges = {
                    frozenset((edge[:3], edge[3:])) for edge in directed_edges
                }
                if len(directed_edges) == 2 * len(undirected_edges):
                    return True

        else:
            self.logger.warning(
                """
            Use of not exact is_closed check. This check can lead to misleading
            results. You could try to use `exact=True`.
            See:
             - false positive: https://github.com/wolph/numpy-stl/issues/198
             - false negative: https://github.com/wolph/numpy-stl/pull/213
            """.strip()
            )
            normals = np.asarray(self.normals, dtype=np.float64)
            allowed_max_errors = (
                np.abs(normals).sum(axis=0) * np.finfo(np.float32).eps
            )
            if (np.abs(normals.sum(axis=0)) <= allowed_max_errors).all():
                return True

        self.logger.warning(
            """
        Your mesh is not closed, the mass methods will not function
        correctly on this mesh.  For more info:
        https://github.com/WoLpH/numpy-stl/issues/69
        """.strip()
        )
        return False

    def get_mass_properties(self) -> tuple[np.float32, _f32_1d, _f64_2d]:
        """
        Evaluate and return a tuple with the following elements:
          - the volume
          - the position of the center of gravity (COG)
          - the inertia matrix expressed at the COG

        Documentation can be found here:
        http://www.geometrictools.com/Documentation/PolyhedralMassProperties.pdf
        """
        self.check(True)

        def subexpression(x: _f32_2d) -> tuple[_f32_1d, ...]:
            w0, w1, w2 = x[:, 0], x[:, 1], x[:, 2]
            temp0 = w0 + w1
            f1 = temp0 + w2
            temp1 = w0 * w0
            temp2 = temp1 + w1 * temp0
            f2 = temp2 + w2 * f1
            f3 = w0 * temp1 + w1 * temp2 + w2 * f2
            g0 = f2 + w0 * (f1 + w0)
            g1 = f2 + w1 * (f1 + w1)
            g2 = f2 + w2 * (f1 + w2)
            return f1, f2, f3, g0, g1, g2

        x0, x1, x2 = self.x[:, 0], self.x[:, 1], self.x[:, 2]
        y0, y1, y2 = self.y[:, 0], self.y[:, 1], self.y[:, 2]
        z0, z1, z2 = self.z[:, 0], self.z[:, 1], self.z[:, 2]
        a1, b1, c1 = x1 - x0, y1 - y0, z1 - z0
        a2, b2, c2 = x2 - x0, y2 - y0, z2 - z0
        d0, d1, d2 = b1 * c2 - b2 * c1, a2 * c1 - a1 * c2, a1 * b2 - a2 * b1

        f1x, f2x, f3x, g0x, g1x, g2x = subexpression(self.x)
        f1y, f2y, f3y, g0y, g1y, g2y = subexpression(self.y)  # noqa: RUF059
        f1z, f2z, f3z, g0z, g1z, g2z = subexpression(self.z)  # noqa: RUF059

        intg = np.zeros(10)
        intg[0] = sum(d0 * f1x)
        intg[1:4] = sum(d0 * f2x), sum(d1 * f2y), sum(d2 * f2z)
        intg[4:7] = sum(d0 * f3x), sum(d1 * f3y), sum(d2 * f3z)
        intg[7] = sum(d0 * (y0 * g0x + y1 * g1x + y2 * g2x))
        intg[8] = sum(d1 * (z0 * g0y + z1 * g1y + z2 * g2y))
        intg[9] = sum(d2 * (x0 * g0z + x1 * g1z + x2 * g2z))
        intg /= np.array([6, 24, 24, 24, 60, 60, 60, 120, 120, 120])
        volume = intg[0]
        cog = intg[1:4] / volume
        cogsq = cog**2
        inertia = np.zeros((3, 3))
        inertia[0, 0] = intg[5] + intg[6] - volume * (cogsq[1] + cogsq[2])
        inertia[1, 1] = intg[4] + intg[6] - volume * (cogsq[2] + cogsq[0])
        inertia[2, 2] = intg[4] + intg[5] - volume * (cogsq[0] + cogsq[1])
        inertia[0, 1] = inertia[1, 0] = -(intg[7] - volume * cog[0] * cog[1])
        inertia[1, 2] = inertia[2, 1] = -(intg[8] - volume * cog[1] * cog[2])
        inertia[0, 2] = inertia[2, 0] = -(intg[9] - volume * cog[2] * cog[0])
        return volume, cog, inertia

    def is_convex(self) -> bool:
        """Return True if the mesh is convex, False otherwise."""
        # For each face, project every vertex onto the normal vector and make
        # sure it isn't longer than the projection of the face itself.
        # The dot product is a scaled projection: (a dot b) = |a||b| cos(angle)
        for i, normal_vector in enumerate(self.normals):
            face_projection = np.dot(self.v0[i], normal_vector)
            normal_vector_2d = np.expand_dims(normal_vector, axis=-1)
            all_vertex_projection = np.matmul(self.vectors, normal_vector_2d)
            if not np.all(all_vertex_projection <= face_projection):
                return False

        return True

    def update_units(self) -> None:
        units = self.normals.copy()
        non_zero_areas = self.areas > 0
        areas = self.areas

        if non_zero_areas.shape[0] != areas.shape[0]:  # pragma: no cover
            self.logger.warning(
                'Zero sized areas found, '
                'units calculation will be partially incorrect'
            )

        if non_zero_areas.any():
            non_zero_areas.shape = non_zero_areas.shape[0]
            areas = np.hstack((2 * areas[non_zero_areas],) * DIMENSIONS)
            units[non_zero_areas] /= areas

        self.units = units

    @staticmethod
    def rotation_matrix(axis: '_ToAxis', theta: float) -> _f64_2d:
        """
        Generate a rotation matrix to Rotate the matrix over the given axis by
        the given theta (angle)

        Uses the `Euler-Rodrigues
        <https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula>`_
        formula for fast rotations.

        :param numpy.array axis: Axis to rotate over (x, y, z)
        :param float theta: Rotation angle in radians, use `math.radians` to
                     convert degrees to radians if needed.
        """
        axis_ = np.asarray(axis)
        # No need to rotate if there is no actual rotation
        if not axis_.any():
            return np.identity(3)

        axis_ = axis_ / np.linalg.norm(axis_)

        theta_ = 0.5 * np.asarray(theta)
        a = math.cos(theta_)
        b, c, d = -axis_ * math.sin(theta_)
        angles = a, b, c, d
        powers = [x * y for x in angles for y in angles]
        aa, ab, ac, ad = powers[0:4]
        ba, bb, bc, bd = powers[4:8]  # noqa: RUF059
        ca, cb, cc, cd = powers[8:12]  # noqa: RUF059
        da, db, dc, dd = powers[12:16]  # noqa: RUF059

        return np.array(
            [
                [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
            ]
        )

    def rotate(
        self,
        axis: '_ToAxis',
        theta: float = 0,
        point: '_ToPoint | None' = None,
    ) -> None:
        """
        Rotate the matrix over the given axis by the given theta (angle)

        Uses the :py:func:`rotation_matrix` in the background.

        .. note:: Note that the `point` was accidentaly inverted with the
           old version of the code. To get the old and incorrect behaviour
           simply pass `-point` instead of `point` or `-numpy.array(point)` if
           you're passing along an array.

        :param numpy.array axis: Axis to rotate over (x, y, z)
        :param float theta: Rotation angle in radians, use `math.radians` to
                            convert degrees to radians if needed.
        :param numpy.array point: Rotation point so manual translation is not
                                  required
        """
        # No need to rotate if there is no actual rotation
        if not theta:
            return

        self.rotate_using_matrix(self.rotation_matrix(axis, theta), point)

    def rotate_using_matrix(
        self,
        rotation_matrix: '_f32_2d | _f64_2d',
        point: '_ToPoint | None' = None,
    ) -> None:
        """
        Rotate using a given rotation matrix and optional rotation point

        Note that this rotation produces clockwise rotations for positive
        angles which is arguably incorrect but will remain for legacy reasons.
        For more details, read here:
        https://github.com/WoLpH/numpy-stl/issues/166
        """

        identity = np.identity(rotation_matrix.shape[0])
        # No need to rotate if there is no actual rotation
        if not rotation_matrix.any() or (identity == rotation_matrix).all():
            return

        if isinstance(point, (np.ndarray, list, tuple)) and len(point) == 3:
            point = np.asarray(point)
        elif point is None:
            point = np.array([0, 0, 0])
        elif isinstance(point, (int, float)):
            point = np.asarray([point] * 3)
        else:
            raise TypeError('Incorrect type for point', point)

        def _rotate(matrix: _f32_2d) -> _f64_2d:
            if point.any():
                # Translate while rotating
                return (matrix - point).dot(rotation_matrix) + point
            else:
                # Simply apply the rotation
                return matrix.dot(rotation_matrix)

        # Rotate the normals
        self.normals[:] = _rotate(self.normals[:])

        # Rotate the vectors
        for i in range(3):
            self.vectors[:, i] = _rotate(self.vectors[:, i])

    def translate(self, translation: '_ToTranslation') -> None:
        """
        Translate the mesh in the three directions

        :param numpy.array translation: Translation vector (x, y, z)
        """
        assert len(translation) == 3, 'Translation vector must be of length 3'
        self.x += translation[0]
        self.y += translation[1]
        self.z += translation[2]

    def transform(self, matrix: '_f32_2d | _f64_2d') -> None:
        """
        Transform the mesh with a rotation and a translation stored in a
        single 4x4 matrix

        :param numpy.array matrix: Transform matrix with shape (4, 4), where
                                   matrix[0:3, 0:3] represents the rotation
                                   part of the transformation
                                   matrix[0:3, 3] represents the translation
                                   part of the transformation
        """
        is_a_4x4_matrix = matrix.shape == (4, 4)
        assert is_a_4x4_matrix, 'Transformation matrix must be of shape (4, 4)'
        rotation = matrix[0:3, 0:3]
        unit_det_rotation = np.allclose(np.linalg.det(rotation), 1.0)
        assert unit_det_rotation, 'Rotation matrix has not a unit determinant'
        for i in range(3):
            self.vectors[:, i] = np.dot(rotation, self.vectors[:, i].T).T
        self.x += matrix[0, 3]
        self.y += matrix[1, 3]
        self.z += matrix[2, 3]

    @property
    def min_(self) -> _f32_1d:
        """Mesh minimum value"""
        try:
            return self._min
        except AttributeError:
            self.update_min()
        return self._min

    @min_.setter
    def min_(self, min_: _f32_1d, /) -> None:
        self._min = min_  # pragma: no cover

    @property
    def max_(self) -> _f32_1d:
        """Mesh maximum value"""
        try:
            return self._max
        except AttributeError:
            self.update_max()
        return self._max

    @max_.setter
    def max_(self, max_: _f32_1d, /) -> None:
        self._max = max_  # pragma: no cover

    @property
    def areas(self) -> _f32_2d:
        """Mesh areas"""
        try:
            return self._areas
        except AttributeError:
            self.update_areas()
        return self._areas

    @areas.setter
    def areas(self, areas: _f32_2d, /) -> None:
        self._areas = areas  # pragma: no cover

    @property
    def centroids(self) -> _f32_2d:
        """Mesh centroids"""
        try:
            return self._centroids
        except AttributeError:
            self.update_centroids()
        return self._centroids

    @centroids.setter
    def centroids(self, centroids: _f32_2d, /) -> None:
        self._centroids = centroids  # pragma: no cover

    @property
    def units(self) -> _f32_2d:
        """Mesh unit vectors"""
        try:
            return self._units
        except AttributeError:
            self.update_units()
        return self._units

    @units.setter
    def units(self, units: _f32_2d, /) -> None:
        self._units = units

    @overload
    def __getitem__(self, k: '_ToSlice2_0', /) -> np.float32: ...
    @overload
    def __getitem__(self, k: '_ToSlice2_1', /) -> _f32_1d: ...
    @overload
    def __getitem__(self, k: '_ToSlice2_2', /) -> _f32_2d: ...
    @overload
    def __getitem__(self, k: '_ToIndices', /) -> Any: ...
    def __getitem__(self, k: '_ToIndices', /) -> Any:
        return self.points[k]

    def __setitem__(self, k: '_ToIndices', v: '_ArrayLikeFloat_co', /) -> None:
        self.points[k] = v

    def __len__(self) -> int:
        return self.points.shape[0]

    # mappings iterate over keys, this iterates over values
    def __iter__(self) -> abc.Iterator[_f32_1d]:  # pyright: ignore[reportIncompatibleMethodOverride]
        yield from self.points

    def __repr__(self) -> str:
        return f'<Mesh: {self.name!r} {self.data.size} vertices>'

    def get_mass_properties_with_density(
        self,
        density: float,
    ) -> tuple[np.float32, np.float32, _f32_1d, _f64_2d]:
        # add density for mesh,density unit kg/m3 when mesh is unit is m
        self.check(True)

        def subexpression(x: _f32_2d) -> tuple[_f32_1d, ...]:
            w0, w1, w2 = x[:, 0], x[:, 1], x[:, 2]
            temp0 = w0 + w1
            f1 = temp0 + w2
            temp1 = w0 * w0
            temp2 = temp1 + w1 * temp0
            f2 = temp2 + w2 * f1
            f3 = w0 * temp1 + w1 * temp2 + w2 * f2
            g0 = f2 + w0 * (f1 + w0)
            g1 = f2 + w1 * (f1 + w1)
            g2 = f2 + w2 * (f1 + w2)
            return f1, f2, f3, g0, g1, g2

        x0, x1, x2 = self.x[:, 0], self.x[:, 1], self.x[:, 2]
        y0, y1, y2 = self.y[:, 0], self.y[:, 1], self.y[:, 2]
        z0, z1, z2 = self.z[:, 0], self.z[:, 1], self.z[:, 2]
        a1, b1, c1 = x1 - x0, y1 - y0, z1 - z0
        a2, b2, c2 = x2 - x0, y2 - y0, z2 - z0
        d0, d1, d2 = b1 * c2 - b2 * c1, a2 * c1 - a1 * c2, a1 * b2 - a2 * b1

        f1x, f2x, f3x, g0x, g1x, g2x = subexpression(self.x)
        f1y, f2y, f3y, g0y, g1y, g2y = subexpression(self.y)  # noqa: RUF059
        f1z, f2z, f3z, g0z, g1z, g2z = subexpression(self.z)  # noqa: RUF059

        intg = np.zeros(10)
        intg[0] = sum(d0 * f1x)
        intg[1:4] = sum(d0 * f2x), sum(d1 * f2y), sum(d2 * f2z)
        intg[4:7] = sum(d0 * f3x), sum(d1 * f3y), sum(d2 * f3z)
        intg[7] = sum(d0 * (y0 * g0x + y1 * g1x + y2 * g2x))
        intg[8] = sum(d1 * (z0 * g0y + z1 * g1y + z2 * g2y))
        intg[9] = sum(d2 * (x0 * g0z + x1 * g1z + x2 * g2z))
        intg /= np.array([6, 24, 24, 24, 60, 60, 60, 120, 120, 120])
        volume = intg[0]
        cog = intg[1:4] / volume
        cogsq = cog**2
        vmass = volume * density
        inertia = np.zeros((3, 3))

        inertia[0, 0] = (intg[5] + intg[6]) * density - vmass * (
            cogsq[1] + cogsq[2]
        )
        inertia[1, 1] = (intg[4] + intg[6]) * density - vmass * (
            cogsq[2] + cogsq[0]
        )
        inertia[2, 2] = (intg[4] + intg[5]) * density - vmass * (
            cogsq[0] + cogsq[1]
        )
        inertia[0, 1] = inertia[1, 0] = -(
            intg[7] * density - vmass * cog[0] * cog[1]
        )
        inertia[1, 2] = inertia[2, 1] = -(
            intg[8] * density - vmass * cog[1] * cog[2]
        )
        inertia[0, 2] = inertia[2, 0] = -(
            intg[9] * density - vmass * cog[2] * cog[0]
        )

        return volume, vmass, cog, inertia
