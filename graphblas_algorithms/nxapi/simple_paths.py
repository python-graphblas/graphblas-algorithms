from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph

__all__ = ["is_simple_path"]


def is_simple_path(G, nodes):
    G = to_graph(G)
    return algorithms.is_simple_path(G, nodes)
