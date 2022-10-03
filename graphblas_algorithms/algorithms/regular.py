from graphblas import monoid

from graphblas_algorithms.classes.digraph import to_graph
from graphblas_algorithms.classes.graph import to_undirected_graph
from graphblas_algorithms.utils import get_all, not_implemented_for


def is_regular_core(G):
    if not G.is_directed():
        degrees = G.get_property("degrees+")
        if degrees.nvals != degrees.size:
            return False
        d = degrees[0].value
        return (degrees == d).reduce(monoid.land).value
    else:
        row_degrees = G.get_property("row_degrees+")
        if row_degrees.nvals != row_degrees.size:
            return False
        column_degrees = G.get_property("column_degrees+")
        if column_degrees.nvals != column_degrees.size:
            return False
        d = row_degrees[0].value
        if not (row_degrees == d).reduce(monoid.land):
            return False
        d = column_degrees[0].value
        return (column_degrees == d).reduce(monoid.land).value


def is_regular(G):
    G = to_graph(G)
    return is_regular_core(G)


def is_k_regular_core(G, k):
    degrees = G.get_property("degrees+")
    if degrees.nvals != degrees.size:
        return False
    return (degrees == k).reduce(monoid.land).value


@not_implemented_for("directed")
def is_k_regular(G, k):
    G = to_undirected_graph(G)
    return is_k_regular_core(G, k)


__all__ = get_all(__name__)
