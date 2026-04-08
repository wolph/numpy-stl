from __future__ import annotations

import contextlib
import io
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
import traceback
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
README_PATH: Final = PROJECT_ROOT / 'README.md'
DOCS_ROOT: Final = PROJECT_ROOT / 'docs'
SUPPORTED_LANGUAGES: Final = frozenset({'bash', 'python'})
CLI_COMMANDS: Final = frozenset({'stl', 'stl2ascii', 'stl2bin'})
INSTALL_REWRITES: Final = {
    'pip install numpy-stl': '.',
    'pip install numpy-stl[fast]': '.[fast]',
}
SANDBOX_FIXTURES: Final = {
    'binary_model.stl': (
        PROJECT_ROOT / 'tests' / 'stl_binary' / 'HalfDonut.stl'
    ),
    'closed_model.stl': PROJECT_ROOT / 'tests' / 'stl_binary' / 'Cube.stl',
    'input.stl': PROJECT_ROOT / 'tests' / 'stl_ascii' / 'HalfDonut.stl',
    'model.3mf': PROJECT_ROOT / 'tests' / '3mf' / 'Moon.3mf',
    'model.stl': PROJECT_ROOT / 'tests' / 'stl_ascii' / 'HalfDonut.stl',
    'part1.stl': PROJECT_ROOT / 'tests' / 'stl_ascii' / 'Cube.stl',
    'part2.stl': PROJECT_ROOT / 'tests' / 'stl_binary' / 'HalfDonut.stl',
}
MULTI_SOLID_SAMPLE: Final = textwrap.dedent(
    """\
    solid first
      facet normal 0 0 1
        outer loop
          vertex 0 0 0
          vertex 1 0 0
          vertex 0 1 0
        endloop
      endfacet
    endsolid first
    solid second
      facet normal 0 0 1
        outer loop
          vertex 0 0 1
          vertex 1 0 1
          vertex 0 1 1
        endloop
      endfacet
    endsolid second
    """
)


@dataclass(frozen=True)
class CodeSample:
    source_path: Path
    language: str
    body: str
    block_index: int
    start_line: int

    @property
    def display_path(self) -> str:
        return self.source_path.relative_to(PROJECT_ROOT).as_posix()

    @property
    def preview(self) -> str:
        for line in self.body.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped

        return '<empty sample>'


@dataclass(frozen=True)
class CommandResult:
    command: str
    stdout: str
    stderr: str
    returncode: int


class SampleExecutionError(AssertionError):
    def __init__(
        self,
        sample: CodeSample,
        *,
        command: str,
        stdout: str = '',
        stderr: str = '',
        details: str = '',
    ) -> None:
        self.sample = sample
        self.command = command
        self.stdout = stdout
        self.stderr = stderr
        self.details = details
        super().__init__(format_execution_error(self))


class InstallSandboxCache:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self._venvs: dict[str, Path] = {}

    def run(self, requirement: str) -> CommandResult:
        venv_dir = self._venvs.get(requirement)
        if venv_dir is None:
            venv_name = 'fast' if requirement == '.[fast]' else 'base'
            venv_dir = self.base_dir / venv_name
            create_virtualenv(venv_dir)
            self._venvs[requirement] = venv_dir

        python = venv_python(venv_dir)
        env = os.environ.copy()
        env.setdefault('PIP_DISABLE_PIP_VERSION_CHECK', '1')
        uv = shutil.which('uv')
        if uv is not None:
            command = [
                uv,
                'pip',
                'install',
                '--python',
                str(python),
                requirement,
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                cwd=PROJECT_ROOT,
                env=env,
                text=True,
            )
        else:
            command = [
                str(python),
                '-m',
                'pip',
                'install',
                '--disable-pip-version-check',
                '--no-input',
                requirement,
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                cwd=PROJECT_ROOT,
                env=env,
                text=True,
            )

        return CommandResult(
            command=' '.join(command),
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )


def iter_doc_sources() -> tuple[Path, ...]:
    return (README_PATH, *sorted(DOCS_ROOT.rglob('*.rst')))


def extract_code_samples(path: Path) -> list[CodeSample]:
    text = path.read_text()
    if path.suffix == '.md':
        return extract_markdown_samples(path, text)
    if path.suffix == '.rst':
        return extract_rst_code_blocks(path, text)

    msg = f'Unsupported docs source: {path}'
    raise ValueError(msg)


def extract_markdown_samples(path: Path, text: str) -> list[CodeSample]:
    samples: list[CodeSample] = []
    lines = text.splitlines()
    block_index = 0
    index = 0

    while index < len(lines):
        line = lines[index]
        if not line.startswith('```'):
            index += 1
            continue

        language = line[3:].strip().split(maxsplit=1)[0].lower()
        block_index += 1
        index += 1
        start_line = index + 1
        body: list[str] = []

        while index < len(lines):
            current_line = lines[index]
            if current_line.startswith('```'):
                break

            body.append(current_line)
            index += 1

        samples.append(
            CodeSample(
                source_path=path,
                language=language,
                body='\n'.join(body),
                block_index=block_index,
                start_line=start_line,
            )
        )
        index += 1

    return samples


def extract_rst_code_blocks(path: Path, text: str) -> list[CodeSample]:
    samples: list[CodeSample] = []
    lines = text.splitlines()
    block_index = 0
    index = 0

    while index < len(lines):
        line = lines[index]
        if not line.startswith('.. code-block::'):
            index += 1
            continue

        language = line.split('::', 1)[1].strip().split(maxsplit=1)[0].lower()
        block_index += 1
        index += 1

        block_start = find_rst_block_start(lines, index)
        if block_start is None:
            break
        index = block_start

        block_indent = len(lines[index]) - len(lines[index].lstrip(' '))
        start_line = index + 1
        body: list[str] = []

        while index < len(lines):
            current_line = lines[index]
            if not current_line.strip():
                body.append('')
                index += 1
                continue

            current_indent = len(current_line) - len(current_line.lstrip(' '))
            if current_indent < block_indent:
                break

            body.append(current_line[block_indent:])
            index += 1

        while body and not body[-1]:
            body.pop()

        samples.append(
            CodeSample(
                source_path=path,
                language=language,
                body='\n'.join(body),
                block_index=block_index,
                start_line=start_line,
            )
        )

    return samples


def seed_docs_sandbox(target_dir: Path) -> None:
    for name, source in SANDBOX_FIXTURES.items():
        shutil.copyfile(source, target_dir / name)

    (target_dir / 'multi.stl').write_text(MULTI_SOLID_SAMPLE)


def build_python_namespace() -> dict[str, object]:
    namespace: dict[str, object] = {'__name__': '__main__'}
    prelude = textwrap.dedent(
        """\
        import os
        import warnings

        os.environ.setdefault('MPLBACKEND', 'Agg')
        warnings.filterwarnings(
            'ignore',
            message=(
                'FigureCanvasAgg is non-interactive, and thus '
                'cannot be shown'
            ),
        )

        try:
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as _docs_pyplot
        except ImportError:
            pass
        else:
            _docs_pyplot.show = lambda *args, **kwargs: None
        """
    )
    exec(prelude, namespace, namespace)
    return namespace


def execute_python_sample(
    sample: CodeSample,
    namespace: dict[str, object],
    cwd: Path,
) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    namespace['__file__'] = str(sample.source_path)
    filename = f'{sample.display_path}:{sample.start_line}'

    try:
        with working_directory(cwd):
            with contextlib.redirect_stdout(stdout):
                with contextlib.redirect_stderr(stderr):
                    compiled = compile(sample.body, filename, 'exec')
                    exec(compiled, namespace, namespace)
    except Exception:
        raise SampleExecutionError(
            sample,
            command='python',
            stdout=stdout.getvalue(),
            stderr=stderr.getvalue(),
            details=traceback.format_exc(),
        ) from None


def execute_bash_sample(
    sample: CodeSample,
    cwd: Path,
    install_cache: InstallSandboxCache,
) -> None:
    requirement = install_requirement(sample.body)
    if requirement is not None:
        result = install_cache.run(requirement)
        if result.returncode != 0:
            raise SampleExecutionError(
                sample,
                command=result.command,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        return

    env = os.environ.copy()
    env.setdefault('MPLBACKEND', 'Agg')
    completed = subprocess.run(
        ['bash', '-euxo', 'pipefail', '-c', sample.body],
        capture_output=True,
        check=False,
        cwd=cwd,
        env=env,
        text=True,
    )

    if completed.returncode != 0:
        raise SampleExecutionError(
            sample,
            command=sample.body,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    missing_outputs = [
        output
        for output in expected_cli_outputs(sample.body)
        if output != '-' and not (cwd / output).exists()
    ]
    if missing_outputs:
        outputs = ', '.join(missing_outputs)
        raise SampleExecutionError(
            sample,
            command=sample.body,
            stdout=completed.stdout,
            stderr=completed.stderr,
            details=f'Expected CLI outputs were not created: {outputs}',
        )


def expected_cli_outputs(script: str) -> list[str]:
    outputs: list[str] = []

    for line in script.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue

        argv = shlex.split(stripped)
        if len(argv) < 3 or argv[0] not in CLI_COMMANDS:
            continue

        outputs.append(argv[2])

    return outputs


def install_requirement(script: str) -> str | None:
    commands = [
        line.strip()
        for line in script.splitlines()
        if line.strip() and not line.strip().startswith('#')
    ]
    if len(commands) != 1:
        return None

    return INSTALL_REWRITES.get(commands[0])


def format_execution_error(error: SampleExecutionError) -> str:
    sample = error.sample
    parts = [
        (
            'Docs example failed: '
            f'{sample.display_path}:{sample.start_line} '
            f'(block {sample.block_index}, {sample.language})'
        ),
        f'Preview: {sample.preview}',
        'Code:',
        textwrap.indent(sample.body, '    '),
        'Command:',
        textwrap.indent(error.command, '    '),
    ]
    if error.stdout:
        parts.extend(
            [
                'Stdout:',
                textwrap.indent(error.stdout.rstrip(), '    '),
            ]
        )
    if error.stderr:
        parts.extend(
            [
                'Stderr:',
                textwrap.indent(error.stderr.rstrip(), '    '),
            ]
        )
    if error.details:
        parts.extend(
            [
                'Details:',
                textwrap.indent(error.details.rstrip(), '    '),
            ]
        )

    return '\n'.join(parts)


def runnable_samples(path: Path) -> list[CodeSample]:
    return [
        sample
        for sample in extract_code_samples(path)
        if sample.language in SUPPORTED_LANGUAGES
    ]


def venv_python(venv_dir: Path) -> Path:
    scripts_dir = 'Scripts' if os.name == 'nt' else 'bin'
    executable = 'python.exe' if os.name == 'nt' else 'python'
    return venv_dir / scripts_dir / executable


def create_virtualenv(venv_dir: Path) -> None:
    uv = shutil.which('uv')
    if uv is not None:
        completed = subprocess.run(
            [uv, 'venv', str(venv_dir), '--python', sys.executable],
            capture_output=True,
            check=False,
            cwd=PROJECT_ROOT,
            text=True,
        )
        if completed.returncode == 0:
            return

        msg = '\n'.join(
            [
                f'Failed to create virtualenv with uv: {venv_dir}',
                completed.stdout.rstrip(),
                completed.stderr.rstrip(),
            ]
        )
        raise RuntimeError(msg)

    venv.EnvBuilder(with_pip=True).create(venv_dir)


def find_rst_block_start(lines: list[str], start_index: int) -> int | None:
    index = start_index
    while index < len(lines):
        current_line = lines[index]
        if not current_line.strip():
            index += 1
            continue
        if current_line.lstrip().startswith(':'):
            index += 1
            continue
        return index

    return None


@contextlib.contextmanager
def working_directory(path: Path):
    original_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original_cwd)
