import contextlib
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
    TypeAlias,
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

    _Dedupe: TypeAlias = 'RemoveDuplicates | int'
    _ToAxis: TypeAlias = npt.NDArray[np.integer] | abc.Sequence[int]
    _ToPoint: TypeAlias = (
        float | abc.Sequence[float] | npt.NDArray[np.floating | np.integer]
    )
    _ToTranslation: TypeAlias = (
        abc.Sequence[_ToPoint] | npt.NDArray[np.floating | np.integer]
    )

    # same as used by `np.ndarray.__getitem__`
    _ToIndex: TypeAlias = (
        SupportsIndex | slice | EllipsisType | _ArrayLikeInt_co | None
    )
    _ToIndices: TypeAlias = _ToIndex | tuple[_ToIndex, ...]

    # specific to 2-d arrays
    _ToSlice2_0: TypeAlias = tuple[SupportsIndex, SupportsIndex]
    _ToSlice2_1: TypeAlias = (
        int
        | np.integer
        | tuple[slice | EllipsisType, int]
        | tuple[int, slice | EllipsisType]
    )
    _ToSlice2_2: TypeAlias = (
        slice
        | tuple[()]
        | tuple[slice, slice]
        | list[int]
        | npt.NDArray[np.integer]
        | EllipsisType
    )

_bool_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.bool_]]
_intp_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.intp]]
_u16_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.uint16]]
_u16_2d: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.uint16]]
_f32_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float32]]
_f32_2d: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float32]]
_f32_3d: TypeAlias = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
_f64_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]
_f64_2d: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]

# {"normals": _float32_1d, "vectors": _float32_2d, "attr": _uint16_1d}
_data_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.void]]

#: When removing empty areas, remove areas that are smaller than this
AREA_SIZE_THRESHOLD: int = 0
#: Vectors in a point
VECTORS: int = 3
#: Dimensions used in a vector
DIMENSIONS: int = 3


class Dimension(enum.IntEnum):
    """Named indices for X/Y/Z axes."""

    #: X index (for example, `mesh.v0[0][X]`)
    X = 0
    #: Y index (for example, `mesh.v0[0][Y]`)
    Y = 1
    #: Z index (for example, `mesh.v0[0][Z]`)
    Z = 2


# For backwards compatibility, leave the original references
X: L[Dimension.X] = Dimension.X
Y: L[Dimension.Y] = Dimension.Y
Z: L[Dimension.Z] = Dimension.Z


class RemoveDuplicates(enum.Enum):
    """Strategy for handling duplicate triangles.

    Use with :meth:`BaseMesh.remove_duplicate_polygons`.
    """

    NONE = 0
    SINGLE = 1
    ALL = 2

    @classmethod
    def map(cls, /, value: '_Dedupe') -> 'RemoveDuplicates':
        """Convert an int or RemoveDuplicates to RemoveDuplicates.

        Args:
            value: Integer or RemoveDuplicates enum value.

        Returns:
            The corresponding RemoveDuplicates member.
        """
        if value is True:
            return cls.SINGLE
        elif value and value in cls:
            return cls(value)
        else:
            return cls.NONE


_LoggedT = TypeVar('_LoggedT', bound='_Logged')


def logged(class_: type[_LoggedT]) -> type[_LoggedT]:
    """Initialize the logger for a class.

    Workaround for the Logged base class not properly
    initializing on Linux.

    Args:
        class_: The class to add logging to.

    Returns:
        The class with logger initialized.
    """
    logger_name: str = '.'.join(
        part.strip() for part in (__name__, class_.__name__) if part.strip()
    )

    class_.logger = logging.getLogger(logger_name)

    for key in dir(logger.Logged):
        if not key.startswith('__'):
            setattr(class_, key, getattr(class_, key))

    return class_


@logged
class BaseMesh(logger.Logged, abc.Mapping['_ToIndices', np.ndarray]):
    """Mesh object with easy access to vectors through v0, v1, v2.

    Normals, areas, min, max, and units are calculated
    automatically.

    Args:
        data: Structured NumPy array with dtype
            :attr:`BaseMesh.dtype`.
        calculate_normals: Whether to recalculate normals
            after loading. Defaults to True.
        remove_empty_areas: Whether to remove triangles
            with zero area. Defaults to False.
        remove_duplicate_polygons: Strategy for handling
            duplicate triangles. Defaults to
            :attr:`RemoveDuplicates.NONE`.
        name: Name of the solid (from ASCII STL header).
        speedups: Use Cython speedups when available.
            Defaults to True.

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
    name: 'bytes | str'
    data: _data_1d

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
        """Per-triangle attribute field (uint16), shape (N, 1)."""
        # https://github.com/numpy/numpy/pull/30261
        return cast('_u16_2d', self.data['attr'])

    @attr.setter
    def attr(self, value: '_ArrayLikeInt_co', /) -> None:
        self.data['attr'] = value

    @property
    def normals(self) -> _f32_2d:
        """Per-triangle normal vectors, shape (N, 3).

        Lazily computed on first access. Call
        :meth:`update_normals` after modifying vertices
        to refresh.

        Example:
            >>> import numpy as np
            >>> from stl.base import BaseMesh
            >>> data = np.zeros(1, dtype=BaseMesh.dtype)
            >>> data['vectors'][0] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
            >>> m = BaseMesh(data, remove_empty_areas=False)
            >>> m.normals[0].tolist()
            [0.0, 0.0, 1.0]
        """
        # https://github.com/numpy/numpy/pull/30261
        return cast('_f32_2d', self.data['normals'])

    @normals.setter
    def normals(self, value: '_ArrayLikeFloat_co', /) -> None:
        self.data['normals'] = value

    @property
    def vectors(self) -> _f32_3d:
        """Triangle vertices as (N, 3, 3) array."""
        # https://github.com/numpy/numpy/pull/30261
        return cast('_f32_3d', self.data['vectors'])

    @vectors.setter
    def vectors(self, value: '_ArrayLikeFloat_co', /) -> None:
        self.data['vectors'] = value

    @property
    def points(self) -> _f32_2d:
        """All vertices flattened as (N, 9) array."""
        return self.vectors.reshape(self.data.size, 9)

    @points.setter
    def points(self, value: '_ArrayLikeFloat_co', /) -> None:
        self.points[:] = value

    @property
    def v0(self) -> _f32_2d:
        """First vertex of each triangle, shape (N, 3)."""
        return self.vectors[:, 0]

    @v0.setter
    def v0(self, value: '_ArrayLikeFloat_co', /) -> None:
        self.vectors[:, 0] = value

    @property
    def v1(self) -> _f32_2d:
        """Second vertex of each triangle, shape (N, 3)."""
        return self.vectors[:, 1]

    @v1.setter
    def v1(self, value: '_ArrayLikeFloat_co', /) -> None:
        self.vectors[:, 1] = value

    @property
    def v2(self) -> _f32_2d:
        """Third vertex of each triangle, shape (N, 3)."""
        return self.vectors[:, 2]

    @v2.setter
    def v2(self, value: '_ArrayLikeFloat_co', /) -> None:
        self.vectors[:, 2] = value

    @property
    def x(self) -> _f32_2d:
        """X coordinates by vertex, shape (N, 3)."""
        return self.points[:, Dimension.X :: 3]

    @x.setter
    def x(self, value: '_ArrayLikeFloat_co', /) -> None:
        self.points[:, Dimension.X :: 3] = value

    @property
    def y(self) -> _f32_2d:
        """Y coordinates by vertex, shape (N, 3)."""
        return self.points[:, Dimension.Y :: 3]

    @y.setter
    def y(self, value: '_ArrayLikeFloat_co', /) -> None:
        self.points[:, Dimension.Y :: 3] = value

    @property
    def z(self) -> _f32_2d:
        """Z coordinates by vertex, shape (N, 3)."""
        return self.points[:, Dimension.Z :: 3]

    @z.setter
    def z(self, value: '_ArrayLikeFloat_co', /) -> None:
        self.points[:, Dimension.Z :: 3] = value

    @staticmethod
    def remove_duplicate_polygons(
        data: _data_1d,
        value: '_Dedupe' = RemoveDuplicates.SINGLE,
    ) -> _data_1d:
        """Remove duplicate triangles from mesh data.

        Args:
            data: Structured mesh array.
            value: Deduplication strategy. Use
                :attr:`RemoveDuplicates.SINGLE` to keep one
                copy, :attr:`RemoveDuplicates.ALL` to remove
                all copies, or :attr:`RemoveDuplicates.NONE`
                to keep everything.

        Returns:
            Filtered mesh data array.

        Note:
            ``ALL`` has a safety fallback: when removing every
            duplicated polygon would drop half the mesh or more, a
            single copy of each duplicate is kept instead (the
            ``SINGLE`` behavior).
        """
        value = RemoveDuplicates.map(value)
        # Compare the full triangles; a lossy key such as the per-axis
        # vertex sum would treat distinct triangles as duplicates.
        polygons: _f32_2d = data['vectors'].reshape(len(data), 9)
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
            # A polygon is fully unique when it differs from both its
            # predecessor (diff_a) and its successor (diff_b) in the
            # sorted order.
            diff_a: _bool_1d = np.concatenate(([True], diff))
            diff_b: _bool_1d = np.concatenate((diff, [True]))

            filtered_data: _data_1d = data[np.sort(idx[diff_a & diff_b])]
            if len(filtered_data) <= len(data) / 2:
                return data[np.sort(idx[diff_a])]
            else:
                return filtered_data
        else:
            return data

    @staticmethod
    def remove_empty_areas(data: _data_1d) -> _data_1d:
        """Remove triangles with zero surface area.

        Filters out degenerate triangles where all three
        vertices are collinear or coincident.

        Args:
            data: Structured mesh array.

        Returns:
            Filtered mesh data array with zero-area
            triangles removed.
        """
        # https://github.com/numpy/numpy/pull/30261
        vectors: _f32_3d = cast('_f32_3d', data['vectors'])
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
        """Recalculate normals from current vertex positions.

        Also refreshes areas and centroids by default.

        Args:
            update_areas: Whether to also refresh cached
                areas. Defaults to True.
            update_centroids: Whether to also refresh cached
                centroids. Defaults to True.
        """
        normals: _f32_2d = np.cross(self.v1 - self.v0, self.v2 - self.v0)  # pyrefly: ignore

        if update_areas:
            self.update_areas(normals)

        if update_centroids:
            self.update_centroids()

        self.normals[:] = normals

    def get_unit_normals(self) -> _f32_2d:
        """Return a copy of normals normalized to unit length.

        Unlike the :attr:`units` property, this method
        always recomputes from the current :attr:`normals`
        array.

        Returns:
            Array of shape (N, 3) with unit-length normals.
            Zero-length normals remain as zeros.
        """
        normals = self.normals.copy()
        normal: _f32_1d = np.linalg.norm(normals, axis=1)
        non_zero = normal > 0
        if non_zero.any():
            normals[non_zero] /= normal[non_zero][:, None]
        return normals

    def update_min(self) -> None:
        """Refresh the cached bounding box minimum."""
        self._min = self.vectors.min(axis=(0, 1))

    def update_max(self) -> None:
        """Refresh the cached bounding box maximum."""
        self._max = self.vectors.max(axis=(0, 1))

    def _invalidate_bounds(self) -> None:
        """Drop cached bounding box values after vertex mutations.

        The :attr:`min_` / :attr:`max_` properties lazily recompute on
        the next access.
        """
        for attribute in ('_min', '_max'):
            if hasattr(self, attribute):
                delattr(self, attribute)

    def update_areas(self, normals: '_f32_2d | None' = None) -> None:
        """Refresh the cached per-triangle areas.

        Args:
            normals: Pre-computed cross products. If None,
                recomputes from current vertices.
        """
        if normals is None:
            normals = np.cross(self.v1 - self.v0, self.v2 - self.v0)  # pyrefly: ignore

        areas = 0.5 * np.sqrt((normals**2).sum(axis=1))  # pyrefly: ignore
        # https://github.com/numpy/numpy/pull/30261
        self._areas = cast('_f32_2d', areas.reshape((areas.size, 1)))

    def update_centroids(self) -> None:
        """Refresh the cached per-triangle centroids."""
        self._centroids = np.mean([self.v0, self.v1, self.v2], axis=0)

    def check(self, exact: bool = False) -> bool:
        """Check whether the mesh is valid (closed).

        Args:
            exact: If True, perform an exact edge-matching
                check. If False, use a faster normal-sum
                heuristic.

        Returns:
            True if the mesh is closed, False otherwise.

        Warning:
            The non-exact check (``exact=False``) can
            produce false positives and false negatives.
            For reliable results, use ``exact=True``. See
            `#198 <https://github.com/WoLpH/numpy-stl/issues/198>`_
            and `#213 <https://github.com/WoLpH/numpy-stl/issues/213>`_.
        """
        return self.is_closed(exact=exact)

    def is_closed(self, exact: bool = False) -> bool:
        """Check whether the mesh is watertight.

        A closed mesh has every edge shared by exactly two
        triangles with consistent winding.

        Args:
            exact: If True, checks directed edges for
                matching pairs. If False, uses a faster
                normal-sum heuristic.

        Returns:
            True if the mesh is closed, False otherwise.

        Warning:
            The non-exact check (``exact=False``) can give
            false positives and false negatives for certain
            mesh geometries. Use ``exact=True`` for reliable
            results. See
            `#198 <https://github.com/WoLpH/numpy-stl/issues/198>`_.
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

    def _get_mass_properties_components(self) -> tuple[_intp_1d, _f64_2d]:
        """Find edge-connected components and a local reference for each."""
        if not self.vectors.size:
            return np.zeros(0, dtype=np.intp), np.zeros((1, 3))
        vertices, indices = np.unique(
            self.vectors.reshape(-1, 3), axis=0, return_inverse=True
        )
        triangles = indices.reshape(-1, 3)
        edges = np.sort(triangles[:, ((0, 1), (1, 2), (2, 0))], axis=2)
        unique_edges, edge_indices = np.unique(
            edges.reshape(-1, 2), axis=0, return_inverse=True
        )
        edge_indices = edge_indices.reshape(-1, 3)
        parents = np.arange(len(triangles), dtype=np.intp)
        while True:
            previous = parents.copy()
            edge_roots = np.full(
                len(unique_edges), len(triangles), dtype=np.intp
            )
            np.minimum.at(
                edge_roots, edge_indices.ravel(), np.repeat(parents, 3)
            )
            parents = edge_roots[edge_indices].min(axis=1)
            parents = parents[parents]
            if np.array_equal(parents, previous):
                break
        roots, labels = np.unique(parents, return_inverse=True)
        references = np.full(len(roots), len(vertices), dtype=np.intp)
        np.minimum.at(references, labels, triangles.min(axis=1))
        return labels, vertices[references].astype(np.float64)

    def _get_mass_properties_exact(
        self, density: float, vectors: _f32_3d | None = None
    ) -> tuple[np.float64, np.float64, _f64_1d, _f64_2d] | None:
        """Accumulate signed tetrahedron moments as exact dyadic integers."""
        if vectors is None:
            vectors = self.vectors
        scale = max(float(x).as_integer_ratio()[1] for x in vectors.flat)
        volume = 0
        first = [0] * 3
        second = [[0] * 3 for _ in range(3)]
        for triangle in vectors:
            a, b, c = [
                [
                    numerator * (scale // denominator)
                    for numerator, denominator in (
                        float(x).as_integer_ratio() for x in vertex
                    )
                ]
                for vertex in triangle
            ]
            determinant = (
                a[0] * (b[1] * c[2] - b[2] * c[1])
                - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0])
            )
            volume += determinant
            sums = [a[i] + b[i] + c[i] for i in range(3)]
            for i in range(3):
                first[i] += determinant * sums[i]
                for j in range(3):
                    second[i][j] += determinant * (
                        sums[i] * sums[j]
                        + a[i] * a[j]
                        + b[i] * b[j]
                        + c[i] * c[j]
                    )
        if not volume:
            return None
        cog = np.array([f / (4 * volume * scale) for f in first])
        # Integral xx^T minus (integral x)(integral x)^T / volume.
        # Form the central moments before converting to float64.
        central = [
            [
                4 * volume * second[i][j] - 5 * first[i] * first[j]
                for j in range(3)
            ]
            for i in range(3)
        ]
        # Sum central diagonal moments before conversion too: signed solids
        # can cancel in a small principal inertia even at stable volume.
        density_numerator, density_denominator = (
            float(density).as_integer_ratio()
            if math.isfinite(density)
            else (1, 1)
        )
        inertia = np.empty((3, 3), dtype=np.float64)
        denominator = 480 * volume * scale**5 * density_denominator
        for i in range(3):
            for j in range(3):
                numerator = (
                    sum(central[k][k] for k in range(3) if k != i)
                    if i == j
                    else -central[i][j]
                )
                if math.isfinite(density):
                    try:
                        inertia[i, j] = (
                            numerator * density_numerator / denominator
                        )
                    except OverflowError:
                        # The exact quotient does not fit in a float64. The
                        # floating path overflows to inf instead of raising.
                        return None
                else:
                    # Only the exact sign is needed for infinite/NaN density.
                    # Converting the unit-density value could overflow first.
                    sign = (numerator > 0) - (numerator < 0)
                    sign *= 1 if volume > 0 else -1
                    inertia[i, j] = np.float64(sign) * density
        result_volume = np.float64(volume / (6 * scale**3))
        return result_volume, result_volume * density, cog, inertia

    def _get_mass_properties_ordinary(
        self, labels: _intp_1d, origins: _f64_2d, bounds: _f64_2d
    ) -> tuple[_bool_1d | None, bool]:
        """Select floating components and flag possible volume cancellation."""
        points = self.vectors.astype(np.float64)
        edge_a = points[:, 1] - points[:, 0]
        edge_b = points[:, 2] - points[:, 0]
        longest = np.maximum(
            np.linalg.norm(edge_a, axis=1),
            np.maximum(
                np.linalg.norm(edge_b, axis=1),
                np.linalg.norm(edge_b - edge_a, axis=1),
            ),
        )
        altitudes = np.divide(
            np.linalg.norm(np.cross(edge_a, edge_b), axis=1),
            longest,
            out=np.zeros_like(longest),
            where=longest != 0,
        )
        # Both short edges and nearly collinear long facets can destroy
        # float64 central moments. Stream exact moments in that range.
        flagged = (
            (altitudes > 0)
            & (
                bounds.max(axis=1)[labels]
                > altitudes / np.sqrt(np.finfo(np.float64).eps)
            )
            & np.isfinite(points).all()
        )
        # Broad triangles can still enclose a nearly coplanar solid. Include
        # cancellation inside the cross products, not just their final signs.
        cross_scale = np.abs(
            edge_a[:, (1, 2, 0)] * edge_b[:, (2, 0, 1)]
        ) + np.abs(edge_a[:, (2, 0, 1)] * edge_b[:, (1, 2, 0)])
        signed_terms: _f64_1d = (
            (points[:, 0] - origins[labels]) * np.cross(edge_a, edge_b)
        ).sum(axis=1)
        signed_volumes = np.bincount(
            labels, weights=signed_terms, minlength=len(origins)
        ).astype(np.float64, copy=False)
        solid_scale = np.bincount(
            labels,
            weights=(np.abs(points[:, 0] - origins[labels]) * cross_scale).sum(
                axis=1
            ),
            minlength=len(origins),
        )
        conditioned = (solid_scale > 0) & (
            np.abs(signed_volumes)
            <= solid_scale
            * np.bincount(labels, minlength=len(origins))
            * np.sqrt(np.finfo(np.float64).eps)
        )
        flagged |= conditioned[labels] & np.isfinite(points).all()
        if not flagged.any():
            return None, False
        ordinary = ~np.isin(labels, np.unique(labels[flagged]))
        # A conditioned volume estimate can have the wrong sign. Treat its
        # winding as unknown when deciding whether mixed moments can cancel.
        possible_cancellation = bool(
            conditioned.any()
            or (np.any(signed_volumes > 0) and np.any(signed_volumes < 0))
        )
        if possible_cancellation:
            # Reserve whole-mesh exact work for cancellation which can amplify
            # a conservative summation error bound beyond sqrt(eps).
            # Flagged facets will be exact and contribute no floating bound.
            cancellation_bound = (
                np.sum(np.abs(signed_terms), where=ordinary)
                * np.count_nonzero(ordinary)
                * np.sqrt(np.finfo(np.float64).eps)
            )
            if abs(signed_volumes.sum()) <= cancellation_bound:
                return np.zeros(
                    len(labels), dtype=np.bool_
                ), possible_cancellation
        return ordinary, possible_cancellation

    def _get_mass_properties_bounds(
        self, labels: _intp_1d, origins: _f64_2d
    ) -> _f64_2d:
        """Compute coordinate spans for each connected component."""
        lower = np.full_like(origins, np.inf)
        upper = np.full_like(origins, -np.inf)
        np.minimum.at(lower, labels, self.vectors.min(axis=1))
        np.maximum.at(upper, labels, self.vectors.max(axis=1))
        return upper - lower

    def _get_mass_properties(
        self, density: float
    ) -> tuple[np.float64, np.float64, _f64_1d, _f64_2d]:
        """Integrate local Eberly moments and combine at the mass centroid."""
        self.check(True)

        labels, origins = self._get_mass_properties_components()
        bounds = self._get_mass_properties_bounds(labels, origins)
        ordinary, possible_cancellation = self._get_mass_properties_ordinary(
            labels, origins, bounds
        )
        exact_components: (
            tuple[np.float64, np.float64, _f64_1d, _f64_2d] | None
        ) = None
        if ordinary is not None and np.count_nonzero(ordinary) * 2 <= len(
            ordinary
        ):
            # When exact facets dominate, stream the original array and release
            # component state rather than copying the majority subset.
            del ordinary
            del labels, origins, bounds
            exact = self._get_mass_properties_exact(density)
            if exact is not None:
                return exact
            labels, origins = self._get_mass_properties_components()
            bounds = self._get_mass_properties_bounds(labels, origins)
            ordinary = None
        elif ordinary is not None:
            # An unrepresentable centered subset also needs whole-mesh exact
            # work: cancellation within the subset can exceed float64 even
            # when the complete mesh's central moments fit.
            with contextlib.suppress(OverflowError):
                exact_components = self._get_mass_properties_exact(
                    1.0, cast('_f32_3d', self.vectors[~ordinary])
                )
            if exact_components is None:
                # A zero-volume signed subset can retain nonzero raw moments.
                # Integrate the whole mesh rather than dropping those moments.
                del ordinary, labels, origins, bounds
                exact = self._get_mass_properties_exact(density)
                if exact is not None:
                    return exact
                labels, origins = self._get_mass_properties_components()
                bounds = self._get_mass_properties_bounds(labels, origins)
                ordinary = None

        # Bin only ordinary contributions without copying the majority array.
        vectors = self.vectors
        references = origins[labels]
        x = vectors[:, :, 0].astype(np.float64)
        y = vectors[:, :, 1].astype(np.float64)
        z = vectors[:, :, 2].astype(np.float64)
        spans = bounds[labels]

        def subexpression(x: _f64_2d) -> tuple[_f64_1d, ...]:
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

        x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
        y0, y1, y2 = y[:, 0], y[:, 1], y[:, 2]
        z0, z1, z2 = z[:, 0], z[:, 1], z[:, 2]
        a1, b1, c1 = x1 - x0, y1 - y0, z1 - z0
        a2, b2, c2 = x2 - x0, y2 - y0, z2 - z0
        d0, d1, d2 = b1 * c2 - b2 * c1, a2 * c1 - a1 * c2, a1 * b2 - a2 * b1
        x -= references[:, 0, None]
        y -= references[:, 1, None]
        z -= references[:, 2, None]
        # Estimate each component's mass centroid before second moments.
        initial_terms = [
            np.choose(
                spans.argmax(axis=1),
                [d0 * x.sum(axis=1), d1 * y.sum(axis=1), d2 * z.sum(axis=1)],
            )
            / 6,
            d0 * subexpression(x)[1] / 24,
            d1 * subexpression(y)[1] / 24,
            d2 * subexpression(z)[1] / 24,
        ]
        initial = np.stack(
            [
                np.bincount(
                    labels,
                    weights=term if ordinary is None else term * ordinary,
                    minlength=len(origins),
                )
                for term in initial_terms
            ],
            axis=1,
        ).astype(np.float64, copy=False)
        volume_roundoff = (
            np.bincount(
                labels,
                weights=np.abs(initial_terms[0]),
                minlength=len(origins),
            )
            * np.bincount(labels, minlength=len(origins))
            * np.finfo(np.float64).eps
        )
        shifts = np.divide(
            initial[:, 1:4],
            initial[:, :1],
            out=np.zeros_like(origins),
            where=np.abs(initial[:, :1]) > volume_roundoff[:, None],
        )
        # Keep the cancellation scale from the original integration frame.
        # Recentering can otherwise hide a lost coordinate of the centroid.
        centroid_scale = np.array(
            [
                np.sum(np.abs(term), where=ordinary)
                if ordinary is not None
                else np.sum(np.abs(term))
                for term in initial_terms[1:]
            ]
        ) + np.abs(initial[:, :1] * origins).sum(axis=0)
        origins += shifts
        x -= shifts[labels, 0, None]
        y -= shifts[labels, 1, None]
        z -= shifts[labels, 2, None]
        del initial_terms, initial, volume_roundoff, shifts
        f1x, f2x, f3x, g0x, g1x, g2x = subexpression(x)
        f1y, f2y, f3y, g0y, g1y, g2y = subexpression(y)
        f1z, f2z, f3z, g0z, g1z, g2z = subexpression(z)
        terms = [
            np.choose(spans.argmax(axis=1), [d0 * f1x, d1 * f1y, d2 * f1z]),
            d0 * f2x,
            d1 * f2y,
            d2 * f2z,
            d0 * f3x,
            d1 * f3y,
            d2 * f3z,
            np.where(
                spans[:, 0] >= spans[:, 1],
                d0 * (y0 * g0x + y1 * g1x + y2 * g2x),
                d1 * (x0 * g0y + x1 * g1y + x2 * g2y),
            ),
            np.where(
                spans[:, 1] >= spans[:, 2],
                d1 * (z0 * g0y + z1 * g1y + z2 * g2y),
                d2 * (y0 * g0z + y1 * g1z + y2 * g2z),
            ),
            np.where(
                spans[:, 2] >= spans[:, 0],
                d2 * (x0 * g0z + x1 * g1z + x2 * g2z),
                d0 * (z0 * g0x + z1 * g1x + z2 * g2x),
            ),
        ]
        intg = np.stack(
            [
                np.bincount(
                    labels,
                    weights=term if ordinary is None else term * ordinary,
                    minlength=len(origins),
                )
                for term in terms
            ],
            axis=1,
        ).astype(np.float64, copy=False)
        del ordinary
        intg /= np.array([6, 24, 24, 24, 60, 60, 60, 120, 120, 120])
        if exact_components is not None:
            exact_volume, _, exact_cog, _ = exact_components
            # Translate the exact subset as a point mass here. Add its central
            # inertia directly below, preserving small principal moments.
            intg = np.vstack(
                (
                    intg,
                    [exact_volume, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                )
            )
            origins = np.vstack((origins, exact_cog))
        volumes = intg[:, 0]
        volume = volumes.sum()
        first = intg[:, 1:4]
        first_global = first + volumes[:, None] * origins
        first_scale = (
            centroid_scale
            + np.abs(first).sum(axis=0)
            + np.abs(volumes[:, None] * origins).sum(axis=0)
        )
        cog = first_global.sum(axis=0) / volume
        # Translate raw moments, including any zero-volume signed component.
        # Dividing each first moment by its volume would lose such components.
        second = np.zeros((len(volumes), 3, 3))
        second[:, 0, 0], second[:, 1, 1], second[:, 2, 2] = intg[:, 4:7].T
        second[:, 0, 1] = second[:, 1, 0] = intg[:, 7]
        second[:, 1, 2] = second[:, 2, 1] = intg[:, 8]
        second[:, 0, 2] = second[:, 2, 0] = intg[:, 9]
        offsets = origins - cog
        second += (
            first[:, :, None] * offsets[:, None, :]
            + offsets[:, :, None] * first[:, None, :]
            + volumes[:, None, None]
            * offsets[:, :, None]
            * offsets[:, None, :]
        )
        inertia_scale = np.abs(second).sum(axis=0)
        inertia_scale[(0, 1, 2), (0, 1, 2)] = np.abs(
            second[:, (1, 2, 0), (1, 2, 0)] + second[:, (2, 0, 1), (2, 0, 1)]
        ).sum(axis=0)
        second = second.sum(axis=0)
        inertia = -second
        inertia[0, 0] = second[1, 1] + second[2, 2]
        inertia[1, 1] = second[0, 0] + second[2, 2]
        inertia[2, 2] = second[0, 0] + second[1, 1]
        if exact_components is not None:
            inertia += exact_components[3]
            inertia_scale += np.abs(exact_components[3])
        unstable = np.any(
            (first_scale > 0)
            & (
                np.abs(first_global.sum(axis=0))
                <= first_scale
                * len(self.vectors)
                * np.sqrt(np.finfo(np.float64).eps)
            )
        ) or (
            exact_components is not None
            and (
                (
                    possible_cancellation
                    and abs(volume)
                    <= np.abs(volumes).sum()
                    * len(self.vectors)
                    * np.sqrt(np.finfo(np.float64).eps)
                )
                # Central and parallel-axis products can cancel even when
                # every component has positive volume.
                or np.any(
                    (inertia_scale > 0)
                    & (
                        np.abs(inertia)
                        <= inertia_scale
                        * len(self.vectors)
                        * np.sqrt(np.finfo(np.float64).eps)
                    )
                )
            )
        )
        inertia *= density
        result = volume, volume * density, cog, inertia
        if unstable:
            # Release facet-sized arrays before streaming a signed fallback.
            del vectors, labels, origins, bounds, references, spans
            del x, y, z, x0, x1, x2, y0, y1, y2, z0, z1, z2
            del a1, b1, c1, a2, b2, c2, d0, d1, d2
            del f1x, f2x, f3x, g0x, g1x, g2x
            del f1y, f2y, f3y, g0y, g1y, g2y
            del f1z, f2z, f3z, g0z, g1z, g2z
            del terms, intg, volumes, first, second, offsets
            del inertia_scale, first_global, first_scale, exact_components
            del centroid_scale
            exact = self._get_mass_properties_exact(density)
            return exact or result
        return result

    def get_mass_properties(self) -> tuple[np.float64, _f64_1d, _f64_2d]:
        """Compute volume, center of gravity, and inertia.

        Uses the polyhedral mass properties algorithm from
        Eberly (Geometric Tools).

        Returns:
            A tuple of (volume, center_of_gravity, inertia):

            - **volume** -- Mesh volume as float64.
            - **center_of_gravity** -- COG as (3,) array.
            - **inertia** -- Inertia tensor as (3, 3) array
              expressed at the COG.

        Example:
            >>> from stl import mesh
            >>> m = mesh.Mesh.from_file('tests/stl_binary/HalfDonut.stl')
            >>> vol, cog, inertia = m.get_mass_properties()
            >>> float(vol) > 0
            True

        Warning:
            These values are only meaningful for closed
            (watertight) meshes. This method checks via
            ``check(exact=True)`` and logs a warning for
            open meshes, but still computes and returns
            the (then unreliable) values. Use
            :meth:`is_closed` to verify beforehand.
        """
        volume, _, cog, inertia = self._get_mass_properties(1.0)
        return volume, cog, inertia

    def is_convex(self) -> bool:
        """Return True if the mesh is convex.

        Tests whether every vertex lies on or behind every
        face plane.

        Returns:
            True if convex, False otherwise.
        """
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
        """Refresh the cached unit normal vectors."""
        units = self.normals.copy()
        non_zero_areas = self.areas > 0
        areas = self.areas

        if non_zero_areas.shape[0] != areas.shape[0]:  # pragma: no cover
            self.logger.warning(
                'Zero sized areas found, '
                'units calculation will be partially incorrect'
            )

        if non_zero_areas.any():
            non_zero_areas = non_zero_areas.reshape(non_zero_areas.shape[0])
            areas = np.hstack((2 * areas[non_zero_areas],) * DIMENSIONS)
            units[non_zero_areas] /= areas

        self.units = units

    @staticmethod
    def rotation_matrix(axis: '_ToAxis', theta: float) -> _f64_2d:
        """Generate a 3x3 rotation matrix.

        Uses the Euler-Rodrigues formula for fast rotation
        matrix construction.

        Args:
            axis: Axis to rotate around as [x, y, z].
            theta: Rotation angle in radians. Use
                ``math.radians()`` to convert from degrees.

        Returns:
            A (3, 3) rotation matrix. Returns the identity
            matrix if the axis is zero.
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
        """Rotate the mesh around an axis.

        Args:
            axis: Axis to rotate around as [x, y, z].
            theta: Rotation angle in radians. Use
                ``math.radians()`` to convert from degrees.
            point: Optional point to rotate around. If
                None, rotates around the origin.

        Example:
            >>> import math
            >>> import numpy as np
            >>> from stl.base import BaseMesh
            >>> data = np.zeros(1, dtype=BaseMesh.dtype)
            >>> data['vectors'][0] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            >>> m = BaseMesh(data, remove_empty_areas=False)
            >>> m.rotate([0, 0, 1], math.radians(90))

        Warning:
            In older versions, the ``point`` parameter was
            accidentally inverted. If you relied on the old
            behavior, pass ``-point`` instead.
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
        """Rotate using a pre-computed rotation matrix.

        Args:
            rotation_matrix: A (3, 3) rotation matrix.
            point: Optional point to rotate around. If
                None, rotates around the origin.

        Warning:
            This method produces clockwise rotations for
            positive angles, which is arguably incorrect
            but retained for backwards compatibility. See
            `#166 <https://github.com/WoLpH/numpy-stl/issues/166>`_.
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

        self._invalidate_bounds()

    def translate(self, translation: '_ToTranslation') -> None:
        """Translate (move) the mesh.

        Args:
            translation: Translation vector [x, y, z].

        Raises:
            AssertionError: If translation is not length 3.

        Example:
            >>> import numpy as np
            >>> from stl.base import BaseMesh
            >>> data = np.zeros(1, dtype=BaseMesh.dtype)
            >>> data['vectors'][0] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
            >>> m = BaseMesh(data, remove_empty_areas=False)
            >>> m.translate([10, 20, 30])
            >>> float(m.v0[0][0])
            10.0
        """
        assert len(translation) == 3, 'Translation vector must be of length 3'
        self.x += translation[0]
        self.y += translation[1]
        self.z += translation[2]
        self._invalidate_bounds()

    def transform(self, matrix: '_f32_2d | _f64_2d') -> None:
        """Apply a 4x4 transformation matrix.

        The upper-left 3x3 submatrix is the rotation.
        The rightmost column (0:3, 3) is the translation.

        Args:
            matrix: A (4, 4) transformation matrix. The
                rotation part must have unit determinant.

        Raises:
            AssertionError: If matrix shape is not (4, 4)
                or rotation determinant is not 1.0.
        """
        is_a_4x4_matrix = matrix.shape == (4, 4)
        assert is_a_4x4_matrix, 'Transformation matrix must be of shape (4, 4)'
        rotation = matrix[0:3, 0:3]
        unit_det_rotation = np.allclose(np.linalg.det(rotation), 1.0)
        assert unit_det_rotation, 'Rotation matrix has not a unit determinant'
        for i in range(3):
            self.vectors[:, i] = np.dot(rotation, self.vectors[:, i].T).T
        # Rotate the stored normals along with the geometry; for a
        # rotation matrix the inverse transpose equals the matrix itself.
        self.normals[:] = np.dot(rotation, self.normals.T).T
        self.x += matrix[0, 3]
        self.y += matrix[1, 3]
        self.z += matrix[2, 3]
        self._invalidate_bounds()

    @property
    def min_(self) -> _f32_1d:
        """Bounding box minimum corner, shape (3,).

        Lazily computed and cached. Call :meth:`update_min`
        after modifying vertices to refresh.

        Warning:
            This value is cached on first access. If you
            modify vertices, call :meth:`update_min` to
            refresh.
        """
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
        """Bounding box maximum corner, shape (3,).

        Lazily computed and cached. Call :meth:`update_max`
        after modifying vertices to refresh.

        Warning:
            This value is cached on first access. If you
            modify vertices, call :meth:`update_max` to
            refresh.
        """
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
        """Per-triangle surface areas, shape (N, 1).

        Lazily computed and cached. Call
        :meth:`update_areas` after modifying vertices
        to refresh.

        Example:
            >>> import numpy as np
            >>> from stl.base import BaseMesh
            >>> data = np.zeros(2, dtype=BaseMesh.dtype)
            >>> data['vectors'][0] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
            >>> m = BaseMesh(data, remove_empty_areas=False)
            >>> float(m.areas[0][0])
            0.5

        Warning:
            This value is cached on first access. If you
            modify vertices, call :meth:`update_areas` to
            get correct results.
        """
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
        """Per-triangle centroids, shape (N, 3).

        Lazily computed and cached. Call
        :meth:`update_centroids` after modifying vertices.

        Example:
            >>> import numpy as np
            >>> from stl.base import BaseMesh
            >>> data = np.zeros(1, dtype=BaseMesh.dtype)
            >>> data['vectors'][0] = [[0, 0, 0], [3, 0, 0], [0, 3, 0]]
            >>> m = BaseMesh(data, remove_empty_areas=False)
            >>> m.centroids[0].tolist()
            [1.0, 1.0, 0.0]
        """
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
        """Per-triangle unit normal vectors, shape (N, 3).

        Lazily computed and cached. Call
        :meth:`update_units` after modifying vertices.

        Example:
            >>> import numpy as np
            >>> from stl.base import BaseMesh
            >>> data = np.zeros(1, dtype=BaseMesh.dtype)
            >>> data['vectors'][0] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
            >>> m = BaseMesh(data, remove_empty_areas=False)
            >>> m.units[0].tolist()
            [0.0, 0.0, 1.0]
        """
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

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        return NotImplemented

    def __hash__(self) -> int:
        # Identity hash, consistent with the identity-based __eq__ above.
        return object.__hash__(self)

    def __repr__(self) -> str:
        return f'<Mesh: {self.name!r} {self.data.size} vertices>'

    def get_mass_properties_with_density(
        self,
        density: float,
    ) -> tuple[np.float64, np.float64, _f64_1d, _f64_2d]:
        """Compute mass properties with a given density.

        Like :meth:`get_mass_properties` but scales volume
        to mass using the provided density.

        Args:
            density: Material density in consistent units
                (e.g., kg/m^3 when mesh units are meters).

        Returns:
            A tuple of (volume, mass, cog, inertia):

            - **volume** -- Mesh volume.
            - **mass** -- Volume * density.
            - **cog** -- Center of gravity as (3,) array.
            - **inertia** -- Inertia tensor as (3, 3)
              array.

        Warning:
            These values are only meaningful for closed
            (watertight) meshes. This method checks via
            ``check(exact=True)`` and logs a warning for
            open meshes, but still computes and returns
            the (then unreliable) values.
        """
        return self._get_mass_properties(density)
