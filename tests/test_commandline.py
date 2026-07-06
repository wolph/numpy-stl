import io
import subprocess
import sys

import pytest

from stl import main, mesh


def test_main(ascii_file, binary_file, tmpdir, speedups):
    original_argv = sys.argv[:]
    args_pre = ['stl']
    args_post = [str(tmpdir.join('output.stl'))]

    if not speedups:
        args_pre.append('-s')

    try:
        sys.argv[:] = [*args_pre, ascii_file, *args_post]
        main.main()
        sys.argv[:] = [*args_pre, '-r', ascii_file, *args_post]
        main.main()
        sys.argv[:] = [*args_pre, '-a', binary_file, *args_post]
        main.main()
        sys.argv[:] = [*args_pre, '-b', ascii_file, *args_post]
        main.main()
    finally:
        sys.argv[:] = original_argv


def test_args(ascii_file, tmpdir):
    parser = main._get_parser('')

    def _get_name(*args) -> str:
        return str(main._get_name(parser.parse_args(list(map(str, args)))))

    assert _get_name('--name', 'foobar') == 'foobar'
    assert _get_name('-', tmpdir.join('binary.stl')).endswith('binary.stl')
    assert _get_name(ascii_file, '-').endswith('HalfDonut.stl')
    assert _get_name('-', '-')


def test_ascii(binary_file, tmpdir, speedups):
    original_argv = sys.argv[:]
    output = tmpdir.join('ascii.stl')
    try:
        sys.argv[:] = [
            'stl',
            *(['-s'] if not speedups else []),
            binary_file,
            str(output),
        ]
        main.to_ascii()
    finally:
        sys.argv[:] = original_argv

    assert output.read_binary().startswith(b'solid')
    assert len(mesh.Mesh.from_file(str(output)).data) > 0


def test_binary(ascii_file, tmpdir, speedups):
    original_argv = sys.argv[:]
    output = tmpdir.join('binary.stl')
    try:
        sys.argv[:] = [
            'stl',
            *(['-s'] if not speedups else []),
            ascii_file,
            str(output),
        ]
        main.to_binary()
    finally:
        sys.argv[:] = original_argv

    assert not output.read_binary().startswith(b'solid')
    assert len(mesh.Mesh.from_file(str(output)).data) > 0


def _run_pipe(
    entry_point: str, stdin_file: str
) -> 'subprocess.CompletedProcess[bytes]':
    # Piped stdin/stdout (not TTYs): the CLI must read/write binary
    # data through the std streams' underlying buffers.
    with open(stdin_file, 'rb') as stdin:
        return subprocess.run(
            [
                sys.executable,
                '-c',
                f'from stl.main import {entry_point}; {entry_point}()',
            ],
            stdin=stdin,
            capture_output=True,
            check=False,
        )


def test_main_stdin_stdout_pipes(binary_file):
    result = _run_pipe('main', binary_file)
    assert result.returncode == 0, result.stderr.decode()

    loaded = mesh.Mesh.from_file('out.stl', fh=io.BytesIO(result.stdout))
    assert len(loaded.data) > 0


def test_to_ascii_stdin_stdout_pipes(binary_file):
    result = _run_pipe('to_ascii', binary_file)
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout.startswith(b'solid')


def test_open_helpers_std_streams():
    assert main._open_infile('-') is sys.stdin.buffer
    assert main._open_outfile('-') is sys.stdout.buffer


def test_main_std_streams_in_process(binary_file, monkeypatch):
    with open(binary_file, 'rb') as fh:
        data = fh.read()

    stdin_buffer = io.BytesIO(data)
    stdout_buffer = io.BytesIO()
    monkeypatch.setattr(sys, 'stdin', io.TextIOWrapper(stdin_buffer))
    monkeypatch.setattr(sys, 'stdout', io.TextIOWrapper(stdout_buffer))
    monkeypatch.setattr(sys, 'argv', ['stl', '-b'])

    main.main()

    loaded = mesh.Mesh.from_file(
        'out.stl', fh=io.BytesIO(stdout_buffer.getvalue())
    )
    assert len(loaded.data) > 0


def test_main_missing_infile(tmpdir, monkeypatch, capsys):
    missing = str(tmpdir.join('does-not-exist.stl'))
    out = str(tmpdir.join('out.stl'))
    monkeypatch.setattr(sys, 'argv', ['stl', missing, out])

    with pytest.raises(SystemExit):
        main.main()
    assert "can't open" in capsys.readouterr().err


def test_main_unopenable_outfile(binary_file, tmpdir, monkeypatch, capsys):
    bad_out = str(tmpdir.join('missing-dir', 'out.stl'))
    monkeypatch.setattr(sys, 'argv', ['stl', binary_file, bad_out])

    with pytest.raises(SystemExit):
        main.main()
    assert "can't open" in capsys.readouterr().err
