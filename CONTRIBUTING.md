# Contributing to numpy-stl

Contributions are welcome! This guide covers the modern development
workflow.

## Quick Start

1. Clone and install:

   ```bash
   git clone https://github.com/WoLpH/numpy-stl.git
   cd numpy-stl
   uv sync --all-extras
   ```

2. Install git hooks:

   ```bash
   lefthook install
   ```

3. Run tests:

   ```bash
   uv run pytest
   ```

4. Lint and format:

   ```bash
   ruff check stl tests
   ruff format stl tests
   ```

5. Type check:

   ```bash
   ty check
   ```

## Development Workflow

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (package manager)
- [lefthook](https://github.com/evilmartians/lefthook) (git hooks)

### Running the Full Test Suite

```bash
uv run pytest
```

This runs all tests with coverage reporting and doctest validation.

To run a subset:

```bash
uv run pytest tests/test_mesh.py -x
```

### Pre-commit Hooks

Lefthook runs these checks in parallel on every commit:

- `ruff check` -- linting
- `ruff format` -- code formatting (auto-fixes staged)
- `ty check` -- type checking

If a hook fails, fix the issue and re-commit.

### Code Style

- **Formatter**: ruff (79-character line length)
- **Quotes**: single quotes for all strings, including docstrings
- **Docstrings**: Google-style (`Args:`, `Returns:`, `Raises:`)
- **Type hints**: required on all functions and methods

### Building Documentation

```bash
uv sync --extra docs
cd docs
uv run sphinx-build -b html . _build/html
```

Open `docs/_build/html/index.html` to preview.

## Pull Request Guidelines

1. The PR should include tests for new functionality
2. All CI checks must pass (tests, lint, type checking)
3. The PR should work for Python 3.10, 3.11, 3.12, and 3.13

## Reporting Bugs

File issues at https://github.com/WoLpH/numpy-stl/issues.

Include:

- Your OS and Python version
- Steps to reproduce
- Expected vs actual behavior
- If applicable, the STL file that triggers the issue
