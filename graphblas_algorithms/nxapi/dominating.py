from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph

__all__ = ["is_dominating_set"]


def is_dominating_set(G, nbunch):
    G = to_graph(G)
    v = G.set_to_vector(nbunch, ignore_extra=True)
    return algorithms.is_dominating_set(G, v)
