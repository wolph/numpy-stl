import typing
from importlib.metadata import (
    PackageNotFoundError,
    version as _version,
)

try:
    __version__: typing.Final[str] = _version('numpy-stl')
except PackageNotFoundError:
    __version__: typing.Final[str] = '0.0.0'  # type: ignore[misc]

__package_name__: typing.Final[str] = 'numpy-stl'
__import_name__: typing.Final[str] = 'stl'
__author__: typing.Final[str] = 'Rick van Hattem'
__author_email__: typing.Final[str] = 'Wolph@Wol.ph'
__description__: typing.Final[str] = (
    'Library to make reading, writing and modifying'
    ' both binary and ascii STL files easy.'
)
__url__: typing.Final[str] = 'https://github.com/WoLpH/numpy-stl/'
