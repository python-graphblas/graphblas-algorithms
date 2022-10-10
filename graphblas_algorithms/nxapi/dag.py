from networkx import NetworkXError

from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph

__all__ = ["descendants", "ancestors"]


def descendants(G, source):
    G = to_graph(G)
    try:
        result = algorithms.descendants(G, source)
        return G.vector_to_set(result)
    except KeyError as e:
        raise NetworkXError(*e.args) from e


def ancestors(G, source):
    G = to_graph(G)
    try:
        result = algorithms.ancestors(G, source)
        return G.vector_to_set(result)
    except KeyError as e:
        raise NetworkXError(*e.args) from e
