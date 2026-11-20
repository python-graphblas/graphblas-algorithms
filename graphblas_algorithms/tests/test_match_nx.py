"""Test that `graphblas.nxapi` structure matches that of networkx.

This currently checks the locations and names of all networkx-dispatchable functions
that are implemented by `graphblas_algorithms`.  It ignores names that begin with `_`.

The `test_dispatched_funcs_in_nxap` test below will say what to add and delete under `nxapi`.

We should consider removing any test here that becomes too much of a nuisance.
For now, though, let's try to match and stay up-to-date with NetworkX!

"""

import re
import sys
from collections import namedtuple
from pathlib import Path

import pytest

from graphblas_algorithms.tests._dispatched import dispatcher_functions

try:
    import networkx as nx
except ImportError:
    pytest.skip(
        "Matching networkx namespace requires networkx to be installed", allow_module_level=True
    )


def isdispatched(func):
    """Can this NetworkX function dispatch to other backends?"""
    return (
        callable(func)
        and hasattr(func, "preserve_edge_attrs")
        and func.__module__.startswith("networkx")
    )


def dispatchname(func):
    """The dispatched name of the dispatchable NetworkX function"""
    # Haha, there should be a better way to get this
    if not isdispatched(func):
        raise ValueError(f"Function is not dispatched in NetworkX: {func.__name__}")
    return func.name


def fullname(func):
    return f"{func.__module__}.{func.__name__}"


NameInfo = namedtuple("NameInfo", ["dispatchname", "fullname", "curpath"])


@pytest.fixture(scope="module")
def nx_info():
    rv = {}  # {modulepath: {dispatchname: NameInfo}}
    for modname, module in sys.modules.items():
        cur = {}
        if (not modname.startswith("networkx.") and modname != "networkx") or "tests" in modname:
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
    """Where each name we implement lives, read from `nxapi` as `scripts/maketree.py` reads it.

    Reading the generated dispatch table instead would make it its own witness.
    """
    from networkx.utils.backends import _registered_algorithms

    from graphblas_algorithms import nxapi  # noqa: F401  (populates sys.modules)

    rv = {}  # {modulepath: {dispatchname: NameInfo}}
    for modname, module in sys.modules.items():
        cur = {}
        if not modname.startswith("graphblas_algorithms.nxapi") or "tests" in modname:
            continue
        for key, val in vars(module).items():
            if key.startswith("_") or key not in _registered_algorithms or not callable(val):
                continue
            if getattr(val, "__module__", "").startswith("graphblas_algorithms.nxapi."):
                cur[key] = NameInfo(key, fullname(val), f"{modname}.{key}")
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


def get_fullname(info):
    fullname = info.fullname
    if not fullname.endswith(f".{info.dispatchname}"):
        fullname += f" ({info.dispatchname})"
    return fullname


def test_print_dispatched_not_implemented(nx_names_to_info, gb_names_to_info):
    """It may be informative to see the results from this to identify functions to implement.

    $ pytest -s -k test_print_dispatched_not_implemented
    """
    not_implemented = nx_names_to_info.keys() - gb_names_to_info.keys()
    fullnames = {get_fullname(next(iter(nx_names_to_info[name]))) for name in not_implemented}
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
    fullnames = {get_fullname(next(iter(nx_names_to_info[name]))) for name in implemented}
    print()
    print("=============================================================================")
    print("Functions dispatched in NetworkX that ARE implemented in graphblas-algorithms")
    print("-----------------------------------------------------------------------------")
    for i, name in enumerate(sorted(fullnames)):
        print(i, name)
    print("=============================================================================")


def test_algorithms_in_readme(nx_names_to_info, gb_names_to_info):
    """Ensure all algorithms are mentioned in README.md, each as a whole word.

    A substring match would count `reciprocity` as documented by `overall_reciprocity`.
    """
    implemented = nx_names_to_info.keys() & gb_names_to_info.keys()
    path = Path(__file__).parent.parent.parent / "README.md"
    if not path.exists():
        pytest.skip("README.md is not shipped with the package")
    text = path.read_text()
    assert implemented, "no implemented names to check"
    missing = {name for name in implemented if not re.search(rf"\b{re.escape(name)}\b", text)}
    assert not missing, f"Algorithms missing in README.md: {', '.join(sorted(missing))}"


def _maketree_or_skip():
    """Import `scripts/maketree.py` by path; `scripts/` is not a package, nor shipped."""
    import importlib.util

    path = Path(__file__).parent.parent.parent / "scripts" / "maketree.py"
    if not path.exists():
        pytest.skip("generators are not shipped with the package")
    spec = importlib.util.spec_from_file_location("maketree", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_autogenerated_code_is_fresh(tmp_path):
    """The committed generated blocks are what the generators would write today.

    The generators write into copies, so a stale checkout fails here instead of being repaired,
    and all three files are compared before anything is asserted.
    """
    maketree = _maketree_or_skip()
    repo = Path(__file__).parent.parent.parent

    from _nx_graphblas.core import main as plugin_main

    plugin_file = repo / "_nx_graphblas" / "__init__.py"
    plugin_copy = tmp_path / "plugin_init.py"
    plugin_copy.write_text(plugin_file.read_text())
    plugin_fresh = plugin_main(plugin_copy) == plugin_file.read_text()

    readme = repo / "README.md"
    interface = repo / "graphblas_algorithms" / "interface.py"
    readme_copy = tmp_path / "README.md"
    interface_copy = tmp_path / "interface.py"
    readme_copy.write_text(readme.read_text(encoding="utf-8"), encoding="utf-8")
    interface_copy.write_text(interface.read_text(encoding="utf-8"), encoding="utf-8")
    maketree.main(print_to_console=False, readme_path=readme_copy, interface_path=interface_copy)

    stale = []
    if interface_copy.read_text(encoding="utf-8") != interface.read_text(encoding="utf-8"):
        stale.append("graphblas_algorithms/interface.py (run `make readme`)")
    if not plugin_fresh:
        stale.append("_nx_graphblas/__init__.py (run `make plugin-info`)")
    if readme_copy.read_text(encoding="utf-8") != readme.read_text(encoding="utf-8"):
        stale.append("README.md (run `make readme`)")
    assert not stale, "Generated code is out of date: " + "; ".join(stale)


class _FakeModule:
    """Stands in for an `nxapi` module the scan will walk."""

    def __init__(self, name, **names):
        self.__name__ = name
        for key, val in names.items():
            setattr(self, key, val)


def test_a_public_name_networkx_does_not_dispatch_stops_the_generator(monkeypatch):
    """A name in `__all__` that NetworkX does not dispatch is reported, not dropped."""
    maketree = _maketree_or_skip()
    module = _FakeModule("graphblas_algorithms.nxapi.invented", __all__=["not_a_networkx_name"])
    monkeypatch.setitem(sys.modules, "graphblas_algorithms.nxapi.invented", module)
    with pytest.raises(ValueError, match="not_a_networkx_name") as excinfo:
        maketree.get_names()
    assert "graphblas_algorithms.nxapi.invented" in str(excinfo.value)


def test_a_module_without_all_is_read_by_the_functions_it_defines(monkeypatch):
    """Without `__all__`, a function the module defines counts and one it imports does not."""
    maketree = _maketree_or_skip()

    def invented_name():  # pragma: no cover (never called)
        pass

    invented_name.__module__ = "graphblas_algorithms.nxapi.invented"
    module = _FakeModule(
        "graphblas_algorithms.nxapi.invented", invented_name=invented_name, borrowed=nx.pagerank
    )
    monkeypatch.setitem(sys.modules, "graphblas_algorithms.nxapi.invented", module)
    with pytest.raises(ValueError, match="invented_name") as excinfo:
        maketree.get_names()
    assert "borrowed" not in str(excinfo.value)


def test_todays_public_surface_passes_the_generator():
    """Every public `nxapi` name is dispatched today, and the generator finds each one."""
    maketree = _maketree_or_skip()
    assert len(maketree.get_names()) == len(dispatcher_functions())


def test_plugin_info_matches_dispatcher():
    """The committed `_nx_graphblas` advertises what the `Dispatcher` offers.

    Unlike the freshness test, this also runs on an installed package, which has no `scripts/`.
    """
    from _nx_graphblas import get_info
    from _nx_graphblas.core import get_extra_parameters

    functions = dispatcher_functions()
    assert functions, "no dispatched functions to check"
    advertised = get_info()["functions"]
    assert set(advertised) == set(functions), (
        "Plugin info is out of sync (run `make plugin-info`). "
        f"Only in _nx_graphblas: {sorted(set(advertised) - set(functions))}; "
        f"only in Dispatcher: {sorted(set(functions) - set(advertised))}"
    )
    fresh, _undocumented, _unregistered = get_extra_parameters(functions)
    expected = {
        name: {"extra_parameters": fresh[name]} if name in fresh else {} for name in functions
    }
    assert advertised == expected, "Extra parameters are out of sync (run `make plugin-info`)"


def test_every_dispatched_name_resolves_to_a_wrapper():
    """Every dispatch table entry is an `nxapi` wrapper, never the kernel of the same name."""
    wrong = {}
    for name, func in sorted(dispatcher_functions().items()):
        module = getattr(func, "__module__", None)
        if module is None or not module.startswith("graphblas_algorithms.nxapi"):
            wrong[name] = module
    assert not wrong, f"dispatched names not resolving to a wrapper: {wrong}"


def test_nxapi_linalg_keeps_its_own_submodules():
    """`nxapi.linalg` keeps its own submodules, which share names with `linalg`'s.

    A star import from a submodule without `__all__` can put the kernel package's modules here.
    """
    from graphblas_algorithms import nxapi

    for name in ("graphmatrix", "laplacianmatrix", "modularitymatrix", "bethehessianmatrix"):
        module = getattr(nxapi.linalg, name)
        assert module.__name__ == f"graphblas_algorithms.nxapi.linalg.{name}", module.__name__
