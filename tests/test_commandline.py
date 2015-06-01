import sys

from stl import main

ascii_file = 'tests/stl_ascii/HalfDonut.stl'
binary_file = 'tests/stl_binary/HalfDonut.stl'


def test_main(tmpdir):
    original_argv = sys.argv[:]
    try:
        sys.argv[:] = ['stl', ascii_file, str(tmpdir.join('binary.stl'))]
        main.main()
        sys.argv[:] = ['stl', '-r', ascii_file, str(tmpdir.join('binary.stl'))]
        main.main()
        sys.argv[:] = ['stl', '-a', binary_file, str(tmpdir.join('ascii.stl'))]
        main.main()
        sys.argv[:] = ['stl', '-b', ascii_file, str(tmpdir.join('binary.stl'))]
        main.main()
    finally:
        sys.argv[:] = original_argv


def test_args(tmpdir):
    parser = main._get_parser('')

    def _get_name(*args):
        return main._get_name(parser.parse_args(list(map(str, args))))

    assert _get_name('--name', 'foobar') == 'foobar'
    assert _get_name('-', tmpdir.join('binary.stl')).endswith('binary.stl')
    assert _get_name(ascii_file, '-').endswith('HalfDonut.stl')
    assert _get_name('-', '-')


def test_ascii(tmpdir):
    original_argv = sys.argv[:]
    try:
        print(str(tmpdir.join('ascii.stl')))
        sys.argv[:] = ['stl', binary_file, str(tmpdir.join('ascii.stl'))]
        try:
            main.to_ascii()
        except SystemExit:
            pass
    finally:
        sys.argv[:] = original_argv


def test_binary(tmpdir):
    original_argv = sys.argv[:]
    try:
        sys.argv[:] = ['stl', ascii_file, str(tmpdir.join('binary.stl'))]
        try:
            main.to_binary()
        except SystemExit:
            pass
    finally:
        sys.argv[:] = original_argv
