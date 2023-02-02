from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph

__all__ = ["floyd_warshall"]


def floyd_warshall(G, weight="weight"):
    G = to_graph(G, weight=weight)
    D = algorithms.floyd_warshall(G, is_weighted=weight is not None)
    return G.matrix_to_dicts(D)
