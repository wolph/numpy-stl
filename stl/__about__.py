from typing import Final

try:
    from importlib.metadata import version as _version

    __version__: Final[str] = _version('numpy-stl')
except Exception:
    __version__: Final[str] = '0.0.0'  # type: ignore[misc]

__package_name__: Final[str] = 'numpy-stl'
__import_name__: Final[str] = 'stl'
__author__: Final[str] = 'Rick van Hattem'
__author_email__: Final[str] = 'Wolph@Wol.ph'
__description__: Final[str] = ' '.join(
    """
Library to make reading, writing and modifying both binary and ascii STL files
easy.
""".split()
)
__url__: Final[str] = 'https://github.com/WoLpH/numpy-stl/'
