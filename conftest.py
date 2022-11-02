import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--check-structure",
        "--checkstructure",
        default=None,
        action="store_true",
        help="Check that `graphblas_algorithms.nxapi` matches networkx structure",
    )


def pytest_runtest_setup(item):
    if "checkstructure" in item.keywords and not item.config.getoption("--check-structure"):
        pytest.skip("need --check-structure option to run")
