from networkx import NetworkXError

from graphblas_algorithms.classes.digraph import to_graph
from graphblas_algorithms.utils import get_all


def s_metric_core(G, normalized=True):
    if normalized:
        raise NetworkXError("Normalization not implemented")
    if G.is_directed():
        degrees = G.get_property("total_degrees+")
    else:
        degrees = G.get_property("degrees+")
    # Alternatives
    # return (degrees @ binary.second(G._A & degrees)).reduce().get(0) / 2
    # return (binary.first(degrees & G._A) @ degrees).reduce().get(0) / 2
    return degrees.outer(degrees).new(mask=G._A.S).reduce_scalar().get(0) / 2


def s_metric(G, normalized=True):
    G = to_graph(G)
    return s_metric_core(G, normalized=normalized)


__all__ = get_all(__name__)
