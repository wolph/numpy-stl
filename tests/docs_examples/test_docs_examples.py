import os

import pytest

from ._docs_examples import (
    PROJECT_ROOT,
    InstallSandboxCache,
    build_python_namespace,
    execute_bash_sample,
    execute_python_sample,
    iter_doc_sources,
    runnable_samples,
    seed_docs_sandbox,
)

DOC_SOURCES = tuple(
    path for path in iter_doc_sources() if runnable_samples(path)
)

pytestmark = pytest.mark.skipif(
    os.environ.get('NUMPY_STL_RUN_DOCS_EXAMPLES') != '1',
    reason='Run docs example tests via tox -e docs-examples',
)


@pytest.fixture(scope='session')
def install_cache(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp('docs-example-installs')
    return InstallSandboxCache(cache_dir)


@pytest.mark.parametrize(
    'source_path',
    DOC_SOURCES,
    ids=lambda path: path.relative_to(PROJECT_ROOT).as_posix(),
)
def test_docs_code_samples_execute(
    source_path,
    tmp_path,
    install_cache,
    speedups,
):
    sandbox = tmp_path / 'sandbox'
    sandbox.mkdir()
    seed_docs_sandbox(sandbox)
    namespace = build_python_namespace()

    for sample in runnable_samples(source_path):
        if sample.language == 'python':
            execute_python_sample(sample, namespace, sandbox)
        else:
            execute_bash_sample(sample, sandbox, install_cache)
