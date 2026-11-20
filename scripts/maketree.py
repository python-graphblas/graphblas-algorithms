#!/usr/bin/env python
"""Run this script to auto-generate API when adding or removing nxapi functions.

This updates API tree in README.md and dispatch functions in `graphblas_algorithms/interface.py`.

"""

import sys
from io import StringIO
from pathlib import Path

import rich
from graphblas.core.utils import _autogenerate_code
from rich.tree import Tree


def trim(name):
    for prefix in ["networkx.algorithms.", "networkx."]:
        if name.startswith(prefix):
            return name[len(prefix) :]
    raise ValueError(f"{name!r} does not begin with a recognized prefix")


def nx_fullname(name, func):
    """`func`'s NetworkX path, plus " (name)" when it is registered under another name.

    For example `bipartite_average_clustering` is `bipartite.cluster.average_clustering`.
    """
    fullname = f"{func.__module__}.{func.__name__}"
    if not fullname.endswith(f".{name}"):
        fullname += f" ({name})"
    return fullname


def get_names():
    """Names we implement: the `nxapi` functions that NetworkX's registry knows.

    Scanning `nxapi`, not the `Dispatcher`, keeps the generated table from being its own input.
    """
    from networkx.utils.backends import _registered_algorithms

    from graphblas_algorithms import nxapi  # noqa: F401  (populates sys.modules)

    implemented = set()
    for modname, module in sys.modules.items():
        if not modname.startswith("graphblas_algorithms.nxapi") or "tests" in modname:
            continue
        for key, val in vars(module).items():
            if key.startswith("_") or key not in _registered_algorithms or not callable(val):
                continue
            if getattr(val, "__module__", "").startswith("graphblas_algorithms.nxapi."):
                implemented.add(key)
    return sorted(trim(nx_fullname(name, _registered_algorithms[name])) for name in implemented)


# Dispatched functions that are only available from `nxapi`
SHORTPATH = {
    "overall_reciprocity",
    "reciprocity",
}


def main(print_to_console=True, update_readme=True, update_interface=True):
    fullnames = get_names()
    # Step 1: add to README.md
    tree = Tree("graphblas_algorithms.nxapi")
    subtrees = {}

    def addtree(path):
        if path in subtrees:
            rv = subtrees[path]
        elif "." not in path:
            rv = subtrees[path] = tree.add(path)
        else:
            subpath, last = path.rsplit(".", 1)
            subtree = addtree(subpath)
            rv = subtrees[path] = subtree.add(last)
        return rv

    for fullname in fullnames:
        addtree(fullname)
    if print_to_console:
        rich.print(tree)
    if update_readme:
        s = StringIO()
        rich.print(tree, file=s)
        s.seek(0)
        text = s.read()
        _autogenerate_code(
            Path(__file__).parent.parent / "README.md",
            f"\n```\n{text}```\n\n",
            begin='[//]: # "Begin auto-generated code"',
            end='[//]: # "End auto-generated code"',
            callblack=False,
        )
    # Step 2: add to interface.py
    lines = []
    prev_mod = None
    for fullname in fullnames:
        mod, subpath = fullname.split(".", 1)
        if mod != prev_mod:
            if prev_mod is not None:
                lines.append("")
            prev_mod = mod
            lines.append(f"    mod = nxapi.{mod}")
            lines.append("    # " + "=" * (len(mod) + 10))
        if " (" in subpath:
            subpath, name = subpath.rsplit(" (", 1)
            name = name.split(")")[0]
        else:
            name = subpath.rsplit(".", 1)[-1]
        if name in SHORTPATH:
            subpath = subpath.rsplit(".", 1)[-1]
            lines.append(f"    {name} = nxapi.{subpath}")
        else:
            lines.append(f"    {name} = mod.{subpath}")
    lines.append("")
    lines.append("    del mod")
    lines.append("")
    # The text's last line becomes the END marker's indent, so end at the class body's.
    text = "\n".join(lines) + " " * 4
    if update_interface:
        _autogenerate_code(
            Path(__file__).parent.parent / "graphblas_algorithms" / "interface.py",
            text,
            specializer="dispatch",
        )
    return tree


if __name__ == "__main__":
    main()
