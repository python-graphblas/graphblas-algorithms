from graphblas_algorithms.classes.digraph import DiGraph, to_graph
from graphblas_algorithms.classes.graph import Graph
from graphblas_algorithms.utils import get_all


def is_triad_core(G):
    # TODO: have DiGraph inherit from Graph to better match NetworkX
    return (
        isinstance(G, (Graph, DiGraph))
        and G.is_directed()
        and G.order() == 3
        and not G.get_property("has_self_edges")
    )


def is_triad(G):
    G = to_graph(G)
    return is_triad_core(G)


__all__ = get_all(__name__)
