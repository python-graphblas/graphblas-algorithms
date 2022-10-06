from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph
from graphblas_algorithms.classes.graph import to_undirected_graph
from graphblas_algorithms.utils import not_implemented_for

__all__ = ["is_regular", "is_k_regular"]


def is_regular(G):
    G = to_graph(G)
    return algorithms.is_regular(G)


@not_implemented_for("directed")
def is_k_regular(G, k):
    G = to_undirected_graph(G)
    return algorithms.is_k_regular(G, k)
