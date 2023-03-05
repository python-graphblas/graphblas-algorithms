from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph

from ..exception import NetworkXError

__all__ = [
    "bfs_layers",
    "descendants_at_distance",
]


def bfs_layers(G, sources):
    G = to_graph(G)
    try:
        for layer in algorithms.bfs_layers(G, sources):
            yield G.vector_to_list(layer)
    except KeyError as e:
        raise NetworkXError(*e.args) from e


def descendants_at_distance(G, source, distance):
    G = to_graph(G)
    try:
        v = algorithms.descendants_at_distance(G, source, distance)
    except KeyError as e:
        raise NetworkXError(*e.args) from e
    return G.vector_to_nodeset(v)
