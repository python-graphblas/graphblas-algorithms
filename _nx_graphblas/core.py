"""Keep the backend info in ``_nx_graphblas/__init__.py`` in step with the dispatcher.

Run ``python _nx_graphblas/__init__.py``, or ``make plugin-info``, after adding or removing
an algorithm. The generated blocks are delimited by ``BEGIN:``/``END:`` comments.
"""

import inspect
from pathlib import Path

# Backend protocol hooks on the Dispatcher; not algorithms.
_BACKEND_METHODS = {"convert_from_nx", "convert_to_nx", "on_start_tests", "can_run", "should_run"}

# Docs for parameters beyond NetworkX's signature, keyed by ("function", "parameter") or by
# "parameter" alone; the specific key wins. The generator appends the live default.
_PARAM_DOCS = {
    "chunksize": (
        "chunksize : int or str, optional",
        "Split the computation into chunks; may specify size as string or number of rows.",
    ),
}


def get_functions():
    from graphblas_algorithms.interface import Dispatcher

    return {
        key: val
        for key, val in sorted(vars(Dispatcher).items())
        if callable(val) and not key.startswith("_") and key not in _BACKEND_METHODS
    }


def dq_repr(s):
    """Quote like black does: double quotes unless the string itself contains one."""
    rs = repr(s)
    if rs.startswith("'") and '"' not in rs:
        return f'"{rs[1:-1]}"'
    return rs


def _format_default(value):
    return f'"{value}"' if isinstance(value, str) else repr(value)


def get_extra_parameters(functions=None):
    """Parameters our signature accepts that the NetworkX signature does not.

    Returns the table, the parameters with no entry in `_PARAM_DOCS`, and the names NetworkX
    does not dispatch (left out of the table).
    """
    from networkx.utils.backends import _registered_algorithms

    if functions is None:
        functions = get_functions()
    rv = {}
    undocumented = []
    unregistered = []
    for name, func in functions.items():
        if name not in _registered_algorithms:
            unregistered.append(name)
            continue
        nx_params = set(inspect.signature(_registered_algorithms[name].orig_func).parameters)
        our_sig = inspect.signature(func)
        params = {}
        for pname in sorted(set(our_sig.parameters) - nx_params):
            param = our_sig.parameters[pname]
            if param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
                continue
            doc = _PARAM_DOCS.get((name, pname)) or _PARAM_DOCS.get(pname)
            if doc is None:
                header, prose = f"{pname} : optional", ""
                undocumented.append(f"{name}({pname})")
            else:
                header, prose = doc
            if param.default is inspect.Parameter.empty:
                params[header] = prose
            else:
                params[header] = f"{prose} Default {_format_default(param.default)}".lstrip()
        if params:
            rv[name] = params
    return rv, undocumented, unregistered


def update_text(text, lines_to_add, target, indent=" " * 4):
    begin = f"# BEGIN: {target}\n"
    end = f"# END: {target}\n"
    start = text.index(begin)
    stop = text.index(end)
    to_add = "\n".join(f"{indent}{line}" for line in lines_to_add)
    return f"{text[:start]}{begin}{to_add}\n{indent}{text[stop:]}"


def main(filepath=None):
    filepath = Path(__file__).parent / "__init__.py" if filepath is None else Path(filepath)
    orig_text = filepath.read_text()

    functions = get_functions()
    text = update_text(orig_text, [f'"{name}",' for name in sorted(functions)], "functions")

    extra_params, undocumented, unregistered = get_extra_parameters(functions)
    lines = []
    for name in sorted(extra_params):
        params = extra_params[name]
        lines.append(f'"{name}": {{')
        lines.extend(f"    {dq_repr(h)}: {dq_repr(params[h])}," for h in sorted(params))
        lines.append("},")
    text = update_text(text, lines, "extra_parameters")

    if undocumented:
        print("No description in _PARAM_DOCS for: " + ", ".join(sorted(undocumented)))
    if unregistered:
        print("Not dispatchable in this NetworkX, so not documented: " + ", ".join(unregistered))
    if text != orig_text:
        filepath.write_text(text)
        print(f"Updated {filepath}")
    else:
        print(f"No changes needed for {filepath}")
    return text
