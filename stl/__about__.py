import typing
from importlib.metadata import (
    PackageNotFoundError,
    version as _version,
)

try:
    _found_version: str = _version('numpy-stl')
except PackageNotFoundError:
    _found_version = '0.0.0'

__version__: typing.Final[str] = _found_version

__package_name__: typing.Final[str] = 'numpy-stl'
__import_name__: typing.Final[str] = 'stl'
__author__: typing.Final[str] = 'Rick van Hattem'
__author_email__: typing.Final[str] = 'Wolph@Wol.ph'
__description__: typing.Final[str] = (
    'Library to make reading, writing and modifying'
    ' both binary and ascii STL files easy.'
)
__url__: typing.Final[str] = 'https://github.com/WoLpH/numpy-stl/'
