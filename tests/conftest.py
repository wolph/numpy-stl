
def pytest_generate_tests(metafunc):
    # Run all tests both with and without speedups
    metafunc.fixturenames.append('enable_speedups')
    metafunc.parametrize('enable_speedups', [False, True])


