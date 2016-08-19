
def pytest_generate_tests(metafunc):
    # Run all tests both with and without speedups
    metafunc.fixturenames.append('speedups')
    metafunc.parametrize('speedups', [False, True])


