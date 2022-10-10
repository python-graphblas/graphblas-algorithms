from graphblas import Matrix

__all__ = ["not_implemented_for"]


def not_implemented_for(*graph_types):
    import networkx.utils.decorators

    rv = networkx.utils.decorators.not_implemented_for(*graph_types)
    func = rv._func

    def inner(g):
        if not isinstance(g, Matrix):
            return func(g)
        # Let Matrix objects pass through and check later.
        # We could check now and convert to appropriate graph type.
        return g

    rv._func = inner
    return rv
