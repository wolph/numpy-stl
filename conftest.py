"""Pytest configuration for the doctests in the ``stl`` package.

The examples in the docstrings read files such as
``tests/stl_binary/HalfDonut.stl`` and write files such as ``example.stl``
with relative paths, the way a user would from their own project directory.
Run from the checkout they would leave the written files behind, so every
doctest runs in its own temporary directory that holds a copy of the example
data. Fixtures for the regular tests live in ``tests/conftest.py``.
"""

from __future__ import annotations

import pathlib
import shutil

import pytest

TESTS_DIR: pathlib.Path = pathlib.Path(__file__).parent / 'tests'


@pytest.fixture(autouse=True)
def doctest_directory(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run each doctest in a scratch directory with the example data."""
    if not isinstance(request.node, pytest.DoctestItem):
        return

    directory: pathlib.Path = request.getfixturevalue('tmp_path')
    # Only the data: no test modules, caches or leftovers of a local
    # ``pytest --basetemp=tests/tmp`` run.
    shutil.copytree(
        TESTS_DIR,
        directory / 'tests',
        ignore=shutil.ignore_patterns('*.py', '__pycache__', 'tmp'),
    )
    monkeypatch.chdir(directory)
