from graphblas_algorithms import linalg
from graphblas_algorithms.classes.digraph import to_graph

__all__ = ["adjacency_matrix"]


def adjacency_matrix(G, nodelist=None, dtype=None, weight="weight"):
    G = to_graph(G, weight=weight, dtype=dtype)
    return linalg.adjacency_matrix(G, nodelist, dtype, is_weighted=weight is not None)
