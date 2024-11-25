import os
import sys
import warnings

from setuptools import extension, setup
from setuptools.command.build_ext import build_ext

setup_kwargs = {}


def error(*lines):
    for _line in lines:
        pass


try:
    from stl import stl

    if not hasattr(stl, 'BaseStl'):
        error(
            'ERROR',
            'You have an incompatible stl package installed'
            'Please run "pip uninstall -y stl" first',
        )
        sys.exit(1)
except ImportError:
    pass


try:
    import numpy as np
    from Cython import Build

    setup_kwargs['ext_modules'] = Build.cythonize(
        [
            extension.Extension(
                'stl._speedups',
                ['stl/_speedups.pyx'],
                include_dirs=[np.get_include()],
            ),
        ]
    )
except ImportError:
    error(
        'WARNING',
        'Cython and Numpy is required for building extension.',
        'Falling back to pure Python implementation.',
    )

# To prevent importing about and thereby breaking the coverage info we use this
# exec hack
about = {}
with open('stl/__about__.py') as fh:
    exec(fh.read(), about)


if os.path.isfile('README.rst'):
    with open('README.rst') as fh:
        long_description = fh.read()
else:
    long_description = 'See http://pypi.python.org/pypi/{}/'.format(
        about['__package_name__']
    )

install_requires = [
    'numpy',
    'python-utils>=3.4.5',
]


tests_require = ['pytest']


class BuildExt(build_ext):
    def run(self):
        try:
            build_ext.run(self)
        except Exception as e:
            warnings.warn(
                f"""
            Unable to build speedups module, defaulting to pure Python. Note
            that the pure Python version is more than fast enough in most cases
            {e!r}
            """,
                stacklevel=2,
            )


if __name__ == '__main__':
    setup(
        python_requires='>3.9.0',
        name=about['__package_name__'],
        version=about['__version__'],
        author=about['__author__'],
        author_email=about['__author_email__'],
        description=about['__description__'],
        url=about['__url__'],
        license='BSD',
        packages=['stl'],
        package_data={about['__import_name__']: ['py.typed']},
        long_description=long_description,
        tests_require=tests_require,
        entry_points={
            'console_scripts': [
                'stl = {}.main:main'.format(about['__import_name__']),
                'stl2ascii = {}.main:to_ascii'.format(
                    about['__import_name__']
                ),
                'stl2bin = {}.main:to_binary'.format(about['__import_name__']),
            ],
        },
        classifiers=[
            'Development Status :: 6 - Mature',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: BSD License',
            'Operating System :: OS Independent',
            'Natural Language :: English',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        install_requires=install_requires,
        cmdclass=dict(
            build_ext=BuildExt,
        ),
        extras_require={
            'docs': [
                'mock',
                'sphinx',
                'python-utils',
            ],
            'tests': [
                'cov-core',
                'coverage',
                'docutils',
                'execnet',
                'numpy',
                'cython',
                'pep8',
                'py',
                'pyflakes',
                'pytest',
                'pytest-cache',
                'pytest-cov',
                'python-utils',
                'Sphinx',
                'flake8',
                'wheel',
            ],
        },
        **setup_kwargs,
    )
