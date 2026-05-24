# Changelog

All notable changes to numpy-stl are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/WoLpH/numpy-stl/compare/v3.2.0...HEAD
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
