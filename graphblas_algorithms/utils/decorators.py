from graphblas import Matrix
from networkx.utils.decorators import not_implemented_for as _not_implemented_for

from ._misc import get_all


def not_implemented_for(*graph_types):
    rv = _not_implemented_for(*graph_types)
    func = rv._func

    def inner(g):
        if not isinstance(g, Matrix):
            return func(g)
        # Let Matrix objects pass through and check later.
        # We could check now and convert to appropriate graph type.
        return g

    rv._func = inner
    return rv


__all__ = get_all(__name__)
