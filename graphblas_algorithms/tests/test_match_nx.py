""" Test that `graphblas.nxapi` structure matches that of networkx.

This currently checks the locations and names of all networkx-dispatchable functions
that are implemented by `graphblas_algorithms`.  It ignores names that begin with `_`.

The `test_dispatched_funcs_in_nxap` test below will say what to add and delete under `nxapi`.

We should consider removing any test here that becomes too much of a nuisance.
For now, though, let's try to match and stay up-to-date with NetworkX!

"""
import sys
from collections import namedtuple

import pytest

import graphblas_algorithms as ga

try:
    import networkx as nx
except ImportError:
    pytest.skip(
        "Matching networkx namespace requires networkx to be installed", allow_module_level=True
    )
else:
    from networkx.classes import backends


def isdispatched(func):
    """Can this NetworkX function dispatch to other backends?"""
    # Haha, there should be a better way to know this
    registered_algorithms = backends._registered_algorithms
    try:
        return (
            func.__globals__.get("_registered_algorithms") is registered_algorithms
            and func.__module__.startswith("networkx")
            and func.__module__ != "networkx.classes.backends"
            and set(func.__code__.co_freevars) == {"func", "name"}
        )
    except Exception:
        return False


def dispatchname(func):
    """The dispatched name of the dispatchable NetworkX function"""
    # Haha, there should be a better way to get this
    if not isdispatched(func):
        raise ValueError(f"Function is not dispatched in NetworkX: {func.__name__}")
    index = func.__code__.co_freevars.index("name")
    return func.__closure__[index].cell_contents


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
    from graphblas_algorithms import nxapi
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


@pytest.mark.checkstructure
def test_dispatched_funcs_in_nxapi(nx_names_to_info, gb_names_to_info):
    """Are graphblas_algorithms functions in the correct locations in nxapi?"""
    failing = False
    for name in nx_names_to_info.keys() & gb_names_to_info.keys():
        nx_paths = {nx_to_gb_info(info) for info in nx_names_to_info[name]}
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
        raise AssertionError()
