import pathlib

import pytest

from stl import mesh

FILES_PATH = pathlib.Path(__file__).parent / 'stl_tests'


@pytest.mark.parametrize('line_ending', ['dos', 'unix'])
def test_line_endings(line_ending, speedups):
    filename = FILES_PATH / (f'{line_ending}.stl')
    mesh.Mesh.from_file(filename, speedups=speedups)
