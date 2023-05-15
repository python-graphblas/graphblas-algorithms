from graphblas import monoid, unary

from .laplacianmatrix import _laplacian_helper

__all__ = ["modularity_matrix", "directed_modularity_matrix"]


def modularity_matrix(G, nodelist=None, is_weighted=False, *, name="modularity_matrix"):
    k, A = _laplacian_helper(G, nodelist, is_weighted)
    m = k.reduce().get(0)
    X = k.outer(k).new(float, name=name)
    X /= m
    X << A - X
    return X


def directed_modularity_matrix(
    G, nodelist=None, is_weighted=False, *, name="directed_modularity_matrix"
):
    A = G._A
    if nodelist is not None:
        ids = G.list_to_ids(nodelist)
        A = A[ids, ids].new()
        if not is_weighted:
            A << unary.one(A)
        k_out = A.reduce_rowwise(monoid.plus).new()
        k_in = A.reduce_columnwise(monoid.plus).new()
    elif is_weighted:
        k_out, k_in = G.get_properties("plus_rowwise+ plus_columnwise+")
    else:
        A = unary.one(A).new()
        k_out, k_in = G.get_properties("row_degrees+ column_degrees+")
    m = k_out.reduce().get(0)
    X = k_out.outer(k_in).new(float, name=name)
    X /= m
    X << A - X
    return X
