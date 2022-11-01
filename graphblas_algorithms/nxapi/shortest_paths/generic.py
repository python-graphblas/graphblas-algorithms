from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph

from ..exception import NodeNotFound

__all__ = ["has_path"]


def has_path(G, source, target):
    G = to_graph(G)
    try:
        return algorithms.has_path(G, source, target)
    except KeyError as e:
        raise NodeNotFound(*e.args) from e
