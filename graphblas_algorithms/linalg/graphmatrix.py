from graphblas import unary

__all__ = ["adjacency_matrix"]


def adjacency_matrix(G, nodelist=None, dtype=None, is_weighted=False, *, name="adjacency_matrix"):
    if dtype is None:
        dtype = G._A.dtype
    if G.is_multigraph():
        is_weighted = True  # XXX
    if nodelist is None:
        if not is_weighted:
            return unary.one[dtype](G._A).new(name=name)
        return G._A.dup(dtype, name=name)
    ids = G.list_to_ids(nodelist)
    A = G._A[ids, ids].new(dtype, name=name)
    if not is_weighted:
        A << unary.one(A)
    return A
