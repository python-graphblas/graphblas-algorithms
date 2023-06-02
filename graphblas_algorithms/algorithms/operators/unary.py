from graphblas import select

from ..exceptions import GraphBlasAlgorithmException

__all__ = ["complement", "reverse"]


def complement(G, *, name="complement"):
    A = G._A
    R = (~A.S).new(A.dtype, name=name)
    R << select.offdiag(R)
    return type(G)(R, key_to_id=G._key_to_id)


def reverse(G, copy=True):
    if not G.is_directed():
        raise GraphBlasAlgorithmException("Cannot reverse an undirected graph.")
    return G.reverse(copy=copy)
