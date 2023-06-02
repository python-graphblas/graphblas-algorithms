from graphblas_algorithms import linalg
from graphblas_algorithms.classes.digraph import to_graph

__all__ = ["laplacian_matrix", "normalized_laplacian_matrix"]


def laplacian_matrix(G, nodelist=None, weight="weight"):
    G = to_graph(G, weight=weight)
    return linalg.laplacian_matrix(G, nodelist, is_weighted=weight is not None)


def normalized_laplacian_matrix(G, nodelist=None, weight="weight"):
    G = to_graph(G, weight=weight)
    return linalg.normalized_laplacian_matrix(G, nodelist, is_weighted=weight is not None)
