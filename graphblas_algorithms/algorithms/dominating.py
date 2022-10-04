from graphblas.semiring import any_pair

from graphblas_algorithms.classes.digraph import to_graph
from graphblas_algorithms.utils import get_all


def is_dominating_set_core(G, nbunch):
    nbrs = any_pair(G._A.T @ nbunch).new(mask=~nbunch.S)  # A or A.T?
    return nbrs.size - nbunch.nvals - nbrs.nvals == 0


def is_dominating_set(G, nbunch):
    G = to_graph(G)
    v = G.set_to_vector(nbunch, ignore_extra=True)
    return is_dominating_set_core(G, v)


__all__ = get_all(__name__)
