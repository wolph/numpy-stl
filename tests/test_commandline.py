import sys

from stl import main


def test_main(ascii_file, binary_file, tmpdir, speedups):
    original_argv = sys.argv[:]
    args_pre = ['stl']
    args_post = [str(tmpdir.join('output.stl'))]

    if not speedups:
        args_pre.append('-s')

    try:
        sys.argv[:] = args_pre + [ascii_file] + args_post
        main.main()
        sys.argv[:] = args_pre + ['-r', ascii_file] + args_post
        main.main()
        sys.argv[:] = args_pre + ['-a', binary_file] + args_post
        main.main()
        sys.argv[:] = args_pre + ['-b', ascii_file] + args_post
        main.main()
    finally:
        sys.argv[:] = original_argv


def test_args(ascii_file, tmpdir):
    parser = main._get_parser('')

    def _get_name(*args):
        return main._get_name(parser.parse_args(list(map(str, args))))

    assert _get_name('--name', 'foobar') == 'foobar'
    assert _get_name('-', tmpdir.join('binary.stl')).endswith('binary.stl')
    assert _get_name(ascii_file, '-').endswith('HalfDonut.stl')
    assert _get_name('-', '-')


def test_ascii(binary_file, tmpdir, speedups):
    original_argv = sys.argv[:]
    try:
        sys.argv[:] = [
            'stl',
            '-s' if not speedups else '',
            binary_file,
            str(tmpdir.join('ascii.stl')),
        ]
        try:
            main.to_ascii()
        except SystemExit:
            pass
    finally:
        sys.argv[:] = original_argv


def test_binary(ascii_file, tmpdir, speedups):
    original_argv = sys.argv[:]
    try:
        sys.argv[:] = [
            'stl',
            '-s' if not speedups else '',
            ascii_file,
            str(tmpdir.join('binary.stl')),
        ]
        try:
            main.to_binary()
        except SystemExit:
            pass
    finally:
        sys.argv[:] = original_argv
