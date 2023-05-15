from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph

from ..exception import NetworkXError

__all__ = [
    "complement",
    "reverse",
]


def complement(G):
    G = to_graph(G)
    return algorithms.complement(G)


def reverse(G, copy=True):
    G = to_graph(G)
    try:
        return algorithms.reverse(G, copy=copy)
    except algorithms.exceptions.GraphBlasAlgorithmException as e:
        raise NetworkXError(*e.args) from e
