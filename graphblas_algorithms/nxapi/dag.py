from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph

from .exception import NetworkXError

__all__ = ["descendants", "ancestors"]


def descendants(G, source):
    G = to_graph(G)
    try:
        result = algorithms.descendants(G, source)
    except KeyError as e:
        raise NetworkXError(*e.args) from e
    else:
        return G.vector_to_nodeset(result)


def ancestors(G, source):
    G = to_graph(G)
    try:
        result = algorithms.ancestors(G, source)
    except KeyError as e:
        raise NetworkXError(*e.args) from e
    else:
        return G.vector_to_nodeset(result)
