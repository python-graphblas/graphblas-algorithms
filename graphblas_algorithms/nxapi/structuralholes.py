from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph

__all__ = []


def mutual_weight(G, u, v, weight=None):
    G = to_graph(G, weight=weight)
    return algorithms.mutual_weight(G, u, v)
