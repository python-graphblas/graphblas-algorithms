from graphblas.semiring import plus_pair

__all__ = ["intra_community_edges", "inter_community_edges"]


def intra_community_edges(G, partition):
    A = G._A
    count = 0
    for block in partition:
        # is A or A.T faster?
        count += plus_pair(A @ block).new(mask=block.S).reduce().get(0)
    return count


def inter_community_edges(G, partition):
    A = G._A
    count = 0
    for block in partition:
        # is A or A.T faster?
        count += plus_pair(A @ block).new(mask=~block.S).reduce().get(0)
    return count
    # Alternative implementation if partition is complete set:
    # return G._A.nvals - intra_community_edges_core(G, partition)
