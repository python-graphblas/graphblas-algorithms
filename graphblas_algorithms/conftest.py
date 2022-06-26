import inspect
import sys
import types

import networkx as nx
import pytest
from networkx.conftest import *  # noqa

import graphblas_algorithms as ga


class Orig:
    pass


@pytest.fixture(scope="session", autouse=True)
def orig():
    """Monkey-patch networkx with functions from graphblas-algorithms"""
    # This doesn't replace functions that have been renamed such as via `import xxx as _xxx`
    orig = Orig()
    replacements = {
        key: (getattr(nx, key), val)
        for key, val in vars(ga).items()
        if not key.startswith("_")
        and hasattr(nx, key)
        and not isinstance(val, types.ModuleType)
        and key not in {"Graph", "DiGraph"}
    }
    replacements["pagerank_scipy"] = (nx.pagerank_scipy, ga.pagerank)
    replacements["pagerank_numpy"] = (nx.pagerank_numpy, ga.pagerank)
    for key, (orig_val, new_val) in replacements.items():
        setattr(orig, key, orig_val)
        if key not in {"pagerank_numpy"}:
            # Ignore keyword-only parameters (works until NetworkX has keyword-only arguments)
            sig = inspect.signature(new_val)
            sig = sig.replace(
                parameters=[x for x in sig.parameters.values() if x.kind != x.KEYWORD_ONLY]
            )
            assert inspect.signature(orig_val) == sig, key
    for name, module in sys.modules.items():
        if not name.startswith("networkx.") and name != "networkx":
            continue
        for key, (orig_val, new_val) in replacements.items():
            if getattr(module, key, None) is orig_val:
                setattr(module, key, new_val)
    yield orig
