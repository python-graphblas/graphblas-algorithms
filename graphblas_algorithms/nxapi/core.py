from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.graph import to_undirected_graph
from graphblas_algorithms.utils import not_implemented_for

__all__ = ["k_truss"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def k_truss(G, k):
    G = to_undirected_graph(G, dtype=bool)
    result = algorithms.k_truss(G, k)
    return result
