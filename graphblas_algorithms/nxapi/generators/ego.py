from graphblas_algorithms import generators
from graphblas_algorithms.classes.digraph import to_graph

__all__ = ["ego_graph"]


def ego_graph(G, n, radius=1, center=True, undirected=False, distance=None):
    G = to_graph(G, weight=distance)
    return generators.ego_graph(
        G, n, radius=radius, center=center, undirected=undirected, is_weighted=distance is not None
    )
