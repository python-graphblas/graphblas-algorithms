import pathlib

import pytest

import graphblas_algorithms as ga

try:
    import setuptools
except ImportError:  # pragma: no cover (import)
    setuptools = None

try:
    import tomli
except ImportError:  # pragma: no cover (import)
    tomli = None


def test_version():
    assert ga.__version__ > "2022.11.0"


@pytest.mark.skipif("not setuptools or not tomli or not ga.__file__")
def test_packages():
    """Ensure all packages are declared in pyproject.toml."""
    # Currently assume s`pyproject.toml` is at the same level as `graphblas_algorithms` folder.
    # This probably isn't always True, and we can probably do a better job of finding it.
    path = pathlib.Path(ga.__file__).parent
    pkgs = [f"graphblas_algorithms.{x}" for x in setuptools.find_packages(path)]
    pkgs.append("graphblas_algorithms")
    pkgs.sort()
    pyproject = path.parent / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("Did not find pyproject.toml")
    with pyproject.open("rb") as f:
        pkgs2 = sorted(tomli.load(f)["tool"]["setuptools"]["packages"])
    assert pkgs == pkgs2
