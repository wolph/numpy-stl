import pathlib

import pytest


def pytest_generate_tests(metafunc):
    # Run all tests both with and without speedups
    metafunc.fixturenames.append('speedups')
    metafunc.parametrize('speedups', [False, True])


@pytest.fixture(scope='session')
def cwd() -> pathlib.Path:
    return pathlib.Path(__file__).parent


@pytest.fixture(scope='session')
def ascii_path(cwd) -> pathlib.Path:
    return cwd / 'stl_ascii'


@pytest.fixture(scope='session')
def binary_path(cwd) -> pathlib.Path:
    return cwd / 'stl_binary'


@pytest.fixture(scope='session')
def three_mf_path(cwd) -> pathlib.Path:
    return cwd / '3mf'


@pytest.fixture(scope='session', params=['ascii', 'binary'])
def binary_ascii_path(request, ascii_path, binary_path) -> pathlib.Path:
    return ascii_path if request.param == 'ascii' else binary_path


@pytest.fixture(scope='session')
def ascii_file(ascii_path) -> str:
    return str(ascii_path / 'HalfDonut.stl')


@pytest.fixture(scope='session')
def binary_file(binary_path) -> str:
    return str(binary_path / 'HalfDonut.stl')
