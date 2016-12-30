import py
import pytest


def pytest_generate_tests(metafunc):
    # Run all tests both with and without speedups
    metafunc.fixturenames.append('speedups')
    metafunc.parametrize('speedups', [False, True])


@pytest.fixture(scope='session')
def cwd():
    return py.path.local(__file__).dirpath()


@pytest.fixture(scope='session')
def ascii_path(cwd):
    return cwd.join('stl_ascii')


@pytest.fixture(scope='session')
def binary_path(cwd):
    return cwd.join('stl_binary')


@pytest.fixture(scope='session')
def ascii_file(ascii_path):
    return str(ascii_path.join('HalfDonut.stl'))


@pytest.fixture(scope='session')
def binary_file(binary_path):
    return str(binary_path.join('HalfDonut.stl'))
