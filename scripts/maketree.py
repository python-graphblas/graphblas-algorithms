#!/usr/bin/env python
"""Run this script to auto-generate API when adding or removing nxapi functions.

This updates API tree in README.md and dispatch functions in `graphblas_algorithms/interface.py`.

"""

import inspect
import sys
from io import StringIO
from pathlib import Path

import rich
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


def public_names(modname, module):
    """`module.__all__` if it has one, else the public functions `modname` itself defines."""
    if (names := getattr(module, "__all__", None)) is not None:
        return list(names)
    return [
        key
        for key, val in vars(module).items()
        if not key.startswith("_")
        and inspect.isfunction(val)
        and getattr(val, "__module__", "") == modname
    ]


def get_names():
    """Names we implement: the `nxapi` functions that NetworkX's registry knows.

    Scanning `nxapi`, not the `Dispatcher`, keeps the generated table from being its own input.
    A public `nxapi` name NetworkX does not dispatch raises, rather than being left out of the
    generated code while it stays importable. On the older supported NetworkX this also stops on a
    wrapper for a name only the newer one dispatches.
    """
    from networkx.utils.backends import _registered_algorithms

    from graphblas_algorithms import nxapi  # noqa: F401  (populates sys.modules)

    implemented = set()
    unregistered = {}
    for modname, module in sys.modules.items():
        if not modname.startswith("graphblas_algorithms.nxapi") or "tests" in modname:
            continue
        if any(part.startswith("_") for part in modname.split(".")[1:]):
            continue
        missing = [
            name for name in public_names(modname, module) if name not in _registered_algorithms
        ]
        if missing:
            unregistered[modname] = sorted(missing)
        for key, val in vars(module).items():
            if key.startswith("_") or key not in _registered_algorithms or not callable(val):
                continue
            if getattr(val, "__module__", "").startswith("graphblas_algorithms.nxapi."):
                implemented.add(key)
    if unregistered:
        import networkx

        found = "; ".join(
            f"{mod}: {', '.join(names)}" for mod, names in sorted(unregistered.items())
        )
        raise ValueError(
            f"networkx {networkx.__version__} does not dispatch these public nxapi names, "
            f"so the generated code would leave them out: {found}"
        )
    return sorted(trim(nx_fullname(name, _registered_algorithms[name])) for name in implemented)


README_BEGIN = '[//]: # "Begin auto-generated code"'
README_END = '[//]: # "End auto-generated code"'
INTERFACE_BEGIN = "# Begin auto-generated code: dispatch"
INTERFACE_END = "# End auto-generated code: dispatch"


def splice(path, text, begin, end):
    """Replace what lies between the `begin` and `end` markers of `path` with `text`.

    Each marker, with its newline, must appear exactly once, so a damaged or duplicated marker
    raises instead of leaving the file unchanged or splicing into the wrong block. UTF-8 on
    both sides, since the README tree is box-drawing characters.
    """
    old = path.read_text(encoding="utf-8")
    begin, end = f"{begin}\n", f"{end}\n"
    counts = {m.strip(): old.count(m) for m in (begin, end)}
    if any(n != 1 for n in counts.values()):
        found = ", ".join(f"{m!r} found {n} times" for m, n in counts.items())
        raise ValueError(f"{path}: expected each generator marker exactly once, but {found}")
    start = old.index(begin) + len(begin)
    stop = old.index(end, start)
    path.write_text(old[:start] + text + old[stop:], encoding="utf-8")


# Dispatched functions that are only available from `nxapi`
SHORTPATH = {
    "overall_reciprocity",
    "reciprocity",
}


def main(
    print_to_console=True,
    update_readme=True,
    update_interface=True,
    readme_path=None,
    interface_path=None,
):
    """Write the algorithm tree into README.md and the dispatch table into interface.py.

    The paths default to this checkout's files; a test passes copies.
    """
    repo = Path(__file__).parent.parent
    readme_path = repo / "README.md" if readme_path is None else Path(readme_path)
    interface_path = (
        repo / "graphblas_algorithms" / "interface.py"
        if interface_path is None
        else Path(interface_path)
    )
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
        splice(readme_path, f"\n```\n{text}```\n\n", README_BEGIN, README_END)
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
        splice(interface_path, text, INTERFACE_BEGIN, INTERFACE_END)
    return tree


if __name__ == "__main__":
    main()
