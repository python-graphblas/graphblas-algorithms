from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph

__all__ = ["floyd_warshall", "floyd_warshall_predecessor_and_distance"]


def floyd_warshall(G, weight="weight"):
    G = to_graph(G, weight=weight)
    D = algorithms.floyd_warshall(G, is_weighted=weight is not None)
    return G.matrix_to_nodenodemap(D)


def floyd_warshall_predecessor_and_distance(G, weight="weight"):
    G = to_graph(G, weight=weight)
    P, D = algorithms.floyd_warshall_predecessor_and_distance(G, is_weighted=weight is not None)
    return (
        G.matrix_to_nodenodemap(P, values_are_keys=True),
        G.matrix_to_nodenodemap(D, fill_value=float("inf")),
    )
