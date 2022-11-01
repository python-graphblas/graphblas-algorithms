from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph

from .exception import NetworkXError

__all__ = ["s_metric"]


def s_metric(G, normalized=True):
    if normalized:
        raise NetworkXError("Normalization not implemented")
    G = to_graph(G)
    return algorithms.s_metric(G)
