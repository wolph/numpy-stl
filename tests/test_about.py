import importlib
import importlib.metadata


def test_version_fallback_on_import_error(monkeypatch, speedups):
    """When importlib.metadata.version raises, __version__ falls back."""

    # Monkeypatch version() to raise PackageNotFoundError
    def _raise(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, 'version', _raise)

    # Force re-import of __about__ to trigger the except branch
    import stl.__about__

    importlib.reload(stl.__about__)
    assert stl.__about__.__version__ == '0.0.0'


def test_version_normal(speedups):
    """Under normal conditions, __version__ is a non-empty string."""
    # Reload to ensure fresh state (previous test may have left
    # the module with the fallback value)
    import stl.__about__

    importlib.reload(stl.__about__)
    version = stl.__about__.__version__

    assert isinstance(version, str)
    assert version != ''
    # The version should match the installed package version
    expected = importlib.metadata.version('numpy-stl')
    assert version == expected
