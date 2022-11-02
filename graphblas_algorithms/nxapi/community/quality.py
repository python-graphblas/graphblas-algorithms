from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph

__all__ = []


def intra_community_edges(G, partition):
    G = to_graph(G)
    partition = [G.set_to_vector(block, ignore_extra=True) for block in partition]
    return algorithms.intra_community_edges(G, partition)


def inter_community_edges(G, partition):
    G = to_graph(G)
    partition = [G.set_to_vector(block, ignore_extra=True) for block in partition]
    return algorithms.inter_community_edges(G, partition)
