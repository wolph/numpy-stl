# Changelog

All notable changes to numpy-stl are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `rotate()` and `rotate_using_matrix()` with a `point` other than the
  origin moved the normals around that point as well as the vertices, so
  a mesh rotated around a pivot carried normals that were no longer
  perpendicular to its faces until you called `update_normals()`. Files
  written with `update_normals=False` kept those wrong normals. Normals
  are now rotated in place and keep their magnitude, and a pivot taken
  from the mesh itself, such as `mesh.normals[0]`, is copied before the
  rotation changes it. The vertices come out exactly as before.
  Thanks to @antonsoo (#248)

## [4.1.0] - 2026-10-16

### Fixed
- `Mesh.from_3mf_file()` built each mesh from zeros and filled in the
  triangles afterwards, so the normals of a loaded 3MF mesh were all zero
  and `calculate_normals`, `name`, `remove_empty_areas` and the other
  constructor options were ignored. An empty 3MF mesh also failed to load.
  The loader now constructs each mesh from the parsed triangles with the
  options you pass. Thanks to @schlagskill for the report (#247)
- `get_mass_properties()` and `get_mass_properties_with_density()` could
  still lose accuracy on a closed mesh whose connected component holds a
  long thin appendage, on a nearly coplanar solid, or on parts whose
  signed volumes cancel. Each component is now integrated around an
  estimate of its own mass centroid, and poorly conditioned components
  fall back to exact integer arithmetic. That exact path is a plain
  Python loop, so a mesh that needs it, including one whose centre of
  gravity has a coordinate close to zero, takes roughly 1.6 times as
  long as before. Thanks to @antonsoo (#246)
- `get_mass_properties()` and `get_mass_properties_with_density()` returned
  a wrong centre of gravity and inertia tensor, often with negative diagonal
  entries, for closed meshes away from the origin or built from distant
  parts. Both now integrate each connected component in float64 around a
  local reference point and combine the parts with the parallel-axis
  theorem. Thanks to @antonsoo (#245)
- The `save()` and `save_ply()` docstring examples wrote to a fixed path
  under `/tmp`, which does not exist on Windows and left files behind after
  every test run. The doctests now run in a temporary directory with a copy
  of the example data, so the examples read and write plain relative paths.
  Thanks to @schlagskill (#244)

### Changed
- Pull requests now receive automatic Ruff fixes from autofix.ci, and the
  README badges report CI and coverage for master.

## [4.0.1] - 2026-09-07

### Fixed
- Resolved all mypy and basedpyright findings; malformed
  `# type: ignore[ty:...]` suppressions replaced by `typing.cast` or
  restructured code.
- Replaced `argparse.FileType` (deprecated since Python 3.14) in the CLI with
  path arguments opened after parsing; stdin/stdout `-` semantics unchanged.
  This also fixes a latent bug where the output file was truncated at
  argument-parsing time, before the input was validated.
- The optional speedups were silently disabled with `speedups>=2.1.0`,
  which moved `ascii_read`/`ascii_write` to the `speedups.stl`
  submodule. The import now targets that submodule and the `fast`
  extra requires `speedups>=2.1.0`.

### Changed
- CI now enforces all four type checkers (pyrefly, mypy, basedpyright, ty).

### Removed
- Legacy `build.cmd` MSVC build script (the speedups extension it built was
  removed in 4.0.0).

## [4.0.0] - 2026-06-17

### Fixed
- Binary STL triangle count is now read and written as the unsigned 32-bit
  integer the spec requires; counts with the high bit set no longer slip past
  the size check and silently read the whole file
- The `stl`, `stl2ascii` and `stl2bin` commands work with piped stdin/stdout
  again: the std stream defaults now use the underlying binary buffers, and
  non-seekable streams are buffered so format auto-detection can rewind
- ASCII STL output uses `{:.9g}` formatting: float32 values round-trip
  exactly instead of values below 5e-7 being flattened to `0.000000`
- `save()` no longer swallows `OSError`s; write failures (disk full, broken
  pipe) propagate instead of silently producing truncated output
- `transform()` now rotates the stored normals along with the vertices
- `translate()`, `transform()` and `rotate_using_matrix()` invalidate the
  cached `min_`/`max_` bounding box
- `remove_duplicate_polygons` compares full triangles instead of per-axis
  vertex sums; distinct triangles with equal sums are no longer dropped
- PLY reader: elements declared before `vertex` are skipped instead of being
  consumed as vertex data; scalar and extra list face properties around the
  vertex index list are consumed correctly; out-of-range and negative face
  indices raise `ValueError` instead of crashing or wrapping around silently
- PLY reader: vertex elements with list properties raise a descriptive
  `ValueError` instead of `KeyError: 'list'`
- 3MF loader raises descriptive `ValueError`s for missing vertex/triangle
  attributes and out-of-range vertex indices
- The pure-Python ASCII reader no longer hits `RecursionError` on files with
  long runs of blank lines
- `Mesh.from_file()` on an empty file raises a descriptive `ValueError`
  instead of `TypeError`
- `save()`/`load()` accept plain ints for `mode` and validate them as
  `Mode` values
- Speedups are disabled automatically on non-seekable streams for both
  reading and writing (the C extension requires seekable files)

### Changed
- `get_mass_properties()` docs no longer claim a `RuntimeError` is raised
  for open meshes; the actual behavior (warning + unreliable values) is
  documented
- Publishing to PyPI is gated on the full CI test suite
- numpy dependency declares the tested lower bound (`numpy>=1.24`)
- The ASCII speedups are no longer bundled as a compiled Cython extension.
  Install the optional `numpy-stl[fast]` extra (the external
  `speedups>=2.0.0` package) to enable them. Without it, `speedups=True`
  transparently falls back to the pure-Python reader/writer, so existing
  code keeps working — only the C-accelerated path now needs the extra.

### Removed
- Support for Python 3.9 and earlier. The minimum supported version is now
  Python 3.10 (`requires-python = ">=3.10"`); installing on older
  interpreters resolves to the previous release.
- The bundled `stl._speedups` Cython extension and its `_speedups.pyx`
  source. `import stl._speedups` no longer works; the accelerated
  `ascii_read`/`ascii_write` now come from the external `speedups` package
  (see the `numpy-stl[fast]` extra under Changed above).

## [3.2.0] - 2024-11-25

### Fixed
- ASCII STL save compatibility with NumPy 2.x

### Changed
- Test matrix: separate NumPy 1.x and 2.x CI jobs

## [3.1.0] - 2023-11-08

### Fixed
- Minor bug fixes and compatibility improvements

## [3.0.0] - 2022-12-14

### Added
- 3MF file format support (read-only)
- Support for `end solid` keyword variant in ASCII STL

## [2.17.0] - 2022-05-14

### Added
- Mass properties calculation with custom density

## [2.16.0] - 2021-03-28

### Fixed
- Various bug fixes

## [2.14.0] - 2021-01-31

### Added
- `is_convex()` method for convexity checking

## [2.11.0] - 2020-03-25

### Added
- `transform()` method for 4x4 matrix transformations

## [2.9.0] - 2018-12-17

### Added
- Rotation around arbitrary points (point parameter)

## [2.7.0] - 2018-06-25

### Added
- Multi-file loading (`from_files()`)

## [2.0.0] - 2016-08-20

### Added
- Cython speedups for ASCII I/O
- Improved test coverage

[Unreleased]: https://github.com/WoLpH/numpy-stl/compare/v4.1.0...HEAD
[4.1.0]: https://github.com/WoLpH/numpy-stl/compare/v4.0.1...v4.1.0
[4.0.1]: https://github.com/WoLpH/numpy-stl/compare/v4.0.0...v4.0.1
[4.0.0]: https://github.com/WoLpH/numpy-stl/compare/v3.2.0...v4.0.0
[3.2.0]: https://github.com/WoLpH/numpy-stl/compare/v3.1.0...v3.2.0
[3.1.0]: https://github.com/WoLpH/numpy-stl/compare/v3.0.0...v3.1.0
[3.0.0]: https://github.com/WoLpH/numpy-stl/compare/v2.17.0...v3.0.0
[2.17.0]: https://github.com/WoLpH/numpy-stl/compare/v2.16.0...v2.17.0
[2.16.0]: https://github.com/WoLpH/numpy-stl/compare/v2.14.0...v2.16.0
[2.14.0]: https://github.com/WoLpH/numpy-stl/compare/v2.11.0...v2.14.0
[2.11.0]: https://github.com/WoLpH/numpy-stl/compare/v2.9.0...v2.11.0
[2.9.0]: https://github.com/WoLpH/numpy-stl/compare/v2.7.0...v2.9.0
[2.7.0]: https://github.com/WoLpH/numpy-stl/compare/v2.0.0...v2.7.0
[2.0.0]: https://github.com/WoLpH/numpy-stl/compare/v1.3.8...v2.0.0
