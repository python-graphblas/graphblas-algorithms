from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph

__all__ = [
    "fast_could_be_isomorphic",
    "faster_could_be_isomorphic",
]


def fast_could_be_isomorphic(G1, G2):
    G1 = to_graph(G1)
    G2 = to_graph(G2)
    return algorithms.fast_could_be_isomorphic(G1, G2)


fast_graph_could_be_isomorphic = fast_could_be_isomorphic


def faster_could_be_isomorphic(G1, G2):
    G1 = to_graph(G1)
    G2 = to_graph(G2)
    return algorithms.faster_could_be_isomorphic(G1, G2)


faster_graph_could_be_isomorphic = faster_could_be_isomorphic
