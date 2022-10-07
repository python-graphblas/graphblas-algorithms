from graphblas_algorithms.classes.digraph import DiGraph
from graphblas_algorithms.classes.graph import Graph

__all__ = ["is_triad"]


def is_triad(G):
    return (
        isinstance(G, (Graph, DiGraph))
        and G.is_directed()
        and G.order() == 3
        and not G.get_property("has_self_edges")
    )
