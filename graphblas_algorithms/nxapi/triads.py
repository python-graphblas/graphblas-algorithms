from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph

__all__ = ["is_triad"]


def is_triad(G):
    G = to_graph(G)
    return algorithms.is_triad(G)
