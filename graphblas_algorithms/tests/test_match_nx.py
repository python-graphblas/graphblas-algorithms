"""Test that `graphblas.nxapi` structure matches that of networkx.

This currently checks the locations and names of all networkx-dispatchable functions
that are implemented by `graphblas_algorithms`.  It ignores names that begin with `_`.

The `test_dispatched_funcs_in_nxap` test below will say what to add and delete under `nxapi`.

We should consider removing any test here that becomes too much of a nuisance.
For now, though, let's try to match and stay up-to-date with NetworkX!

"""
import sys
from collections import namedtuple
from pathlib import Path

import pytest

try:
    import networkx as nx  # noqa: F401
except ImportError:
    pytest.skip(
        "Matching networkx namespace requires networkx to be installed", allow_module_level=True
    )
else:
    from networkx.classes import backends  # noqa: F401


def isdispatched(func):
    """Can this NetworkX function dispatch to other backends?"""
    return (
        callable(func) and hasattr(func, "dispatchname") and func.__module__.startswith("networkx")
    )


def dispatchname(func):
    """The dispatched name of the dispatchable NetworkX function"""
    # Haha, there should be a better way to get this
    if not isdispatched(func):
        raise ValueError(f"Function is not dispatched in NetworkX: {func.__name__}")
    return func.dispatchname


def fullname(func):
    return f"{func.__module__}.{func.__name__}"


NameInfo = namedtuple("NameInfo", ["dispatchname", "fullname", "curpath"])


@pytest.fixture(scope="module")
def nx_info():
    rv = {}  # {modulepath: {dispatchname: NameInfo}}
    for modname, module in sys.modules.items():
        cur = {}
        if not modname.startswith("networkx.") and modname != "networkx" or "tests" in modname:
            continue
        for key, val in vars(module).items():
            if not key.startswith("_") and isdispatched(val):
                dname = dispatchname(val)
                cur[dname] = NameInfo(dname, fullname(val), f"{modname}.{key}")
        if cur:
            rv[modname] = cur
    return rv


@pytest.fixture(scope="module")
def gb_info():
    rv = {}  # {modulepath: {dispatchname: NameInfo}}
    from graphblas_algorithms import nxapi  # noqa: F401
    from graphblas_algorithms.interface import Dispatcher

    ga_map = {
        fullname(val): key
        for key, val in vars(Dispatcher).items()
        if callable(val) and fullname(val).startswith("graphblas_algorithms.nxapi.")
    }
    for modname, module in sys.modules.items():
        cur = {}
        if not modname.startswith("graphblas_algorithms.nxapi") or "tests" in modname:
            continue
        for key, val in vars(module).items():
            try:
                fname = fullname(val)
            except Exception:
                continue
            if key.startswith("_") or fname not in ga_map:
                continue
            dname = ga_map[fname]
            cur[dname] = NameInfo(dname, fname, f"{modname}.{key}")
        if cur:
            rv[modname] = cur
    return rv


@pytest.fixture(scope="module")
def nx_names_to_info(nx_info):
    rv = {}  # {dispatchname: {NameInfo}}
    for names in nx_info.values():
        for name, info in names.items():
            if name not in rv:
                rv[name] = set()
            rv[name].add(info)
    return rv


@pytest.fixture(scope="module")
def gb_names_to_info(gb_info):
    rv = {}  # {dispatchname: {NameInfo}}
    for names in gb_info.values():
        for name, info in names.items():
            if name not in rv:
                rv[name] = set()
            rv[name].add(info)
    return rv


@pytest.mark.checkstructure
def test_nonempty(nx_info, gb_info, nx_names_to_info, gb_names_to_info):
    assert len(nx_info) > 15
    assert len(gb_info) > 15
    assert len(nx_names_to_info) > 30
    assert len(gb_names_to_info) > 30


def nx_to_gb_info(info):
    gb = "graphblas_algorithms.nxapi"
    return NameInfo(
        info[0],
        info[1].replace("networkx.algorithms", gb).replace("networkx", gb),
        info[2].replace("networkx.algorithms", gb).replace("networkx", gb),
    )


def module_exists(info):
    return info[2].rsplit(".", 1)[0] in sys.modules


@pytest.mark.checkstructure
def test_dispatched_funcs_in_nxapi(nx_names_to_info, gb_names_to_info):
    """Are graphblas_algorithms functions in the correct locations in nxapi?"""
    failing = False
    for name in nx_names_to_info.keys() & gb_names_to_info.keys():
        nx_paths = {
            gbinfo
            for info in nx_names_to_info[name]
            if module_exists(gbinfo := nx_to_gb_info(info))
        }
        gb_paths = gb_names_to_info[name]
        if nx_paths != gb_paths:  # pragma: no cover
            failing = True
            if missing := (nx_paths - gb_paths):
                from_ = ":".join(next(iter(missing))[1].rsplit(".", 1))
                print(f"Add `{name}` from `{from_}` here:")
                for _, _, path in sorted(missing):
                    print("   ", ":".join(path.rsplit(".", 1)))
            if extra := (gb_paths - nx_paths):
                from_ = ":".join(next(iter(extra))[1].rsplit(".", 1))
                print(f"Remove `{name}` from `{from_}` here:")
                for _, _, path in sorted(extra):
                    print("   ", ":".join(path.rsplit(".", 1)))
    if failing:  # pragma: no cover
        raise AssertionError


def test_print_dispatched_not_implemented(nx_names_to_info, gb_names_to_info):
    """It may be informative to see the results from this to identify functions to implement.

    $ pytest -s -k test_print_dispatched_not_implemented
    """
    not_implemented = nx_names_to_info.keys() - gb_names_to_info.keys()
    fullnames = {next(iter(nx_names_to_info[name])).fullname for name in not_implemented}
    print()
    print("=================================================================================")
    print("Functions dispatched in NetworkX that ARE NOT implemented in graphblas-algorithms")
    print("---------------------------------------------------------------------------------")
    for i, name in enumerate(sorted(fullnames)):
        print(i, name)
    print("=================================================================================")


def test_print_dispatched_implemented(nx_names_to_info, gb_names_to_info):
    """It may be informative to see the results from this to identify implemented functions.

    $ pytest -s -k test_print_dispatched_implemented
    """
    implemented = nx_names_to_info.keys() & gb_names_to_info.keys()
    fullnames = {next(iter(nx_names_to_info[name])).fullname for name in implemented}
    print()
    print("=============================================================================")
    print("Functions dispatched in NetworkX that ARE implemented in graphblas-algorithms")
    print("-----------------------------------------------------------------------------")
    for i, name in enumerate(sorted(fullnames)):
        print(i, name)
    print("=============================================================================")


def test_algorithms_in_readme(nx_names_to_info, gb_names_to_info):
    """Ensure all algorithms are mentioned in README.md."""
    implemented = nx_names_to_info.keys() & gb_names_to_info.keys()
    path = Path(__file__).parent.parent.parent / "README.md"
    if not path.exists():
        return
    with path.open("r") as f:
        text = f.read()
    missing = set()
    for name in sorted(implemented):
        if name not in text:
            missing.add(name)
    if missing:
        msg = f"Algorithms missing in README.md: {', '.join(sorted(missing))}"
        print(msg)
        raise AssertionError(msg)
