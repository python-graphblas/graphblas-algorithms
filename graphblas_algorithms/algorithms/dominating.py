from graphblas.semiring import lor_pair

__all__ = ["is_dominating_set"]


def is_dominating_set(G, nbunch):
    nbrs = lor_pair(nbunch @ G._A).new(mask=~nbunch.S)  # A or A.T?
    return nbrs.size - nbunch.nvals - nbrs.nvals == 0
