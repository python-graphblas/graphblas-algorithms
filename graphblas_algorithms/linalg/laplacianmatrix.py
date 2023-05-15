from graphblas import monoid, unary

__all__ = [
    "laplacian_matrix",
    "normalized_laplacian_matrix",
]


def _laplacian_helper(G, nodelist=None, is_weighted=False):
    if G.is_multigraph():
        is_weighted = True  # XXX
    A = G._A
    if nodelist is not None:
        ids = G.list_to_ids(nodelist)
        A = A[ids, ids].new()
        if not is_weighted:
            A << unary.one(A)
        d = A.reduce_rowwise(monoid.plus).new()
    elif is_weighted:
        d = G.get_property("plus_rowwise+")
    else:
        d = G.get_property("degrees+")
        A = unary.one(A).new()
    return d, A


def laplacian_matrix(G, nodelist=None, is_weighted=False, *, name="laplacian_matrix"):
    d, A = _laplacian_helper(G, nodelist, is_weighted)
    D = d.diag(name="D")
    return (D - A).new(name=name)


def normalized_laplacian_matrix(
    G, nodelist=None, is_weighted=False, *, name="normalized_laplacian_matrix"
):
    d, A = _laplacian_helper(G, nodelist, is_weighted)
    d_invsqrt = unary.sqrt(d).new(name="d_invsqrt")
    d_invsqrt << unary.minv(d_invsqrt)

    # XXX: what if `d` is 0 and `d_invsqrt` is infinity? (not tested)
    # d_invsqrt(unary.isinf(d_invsqrt)) << 0

    # Calculate: A_weighted = D_invsqrt @ A @ D_invsqrt
    A_weighted = d_invsqrt.outer(d_invsqrt).new(mask=A.S, name=name)
    A_weighted *= A
    # Alt (no idea which implementation is better)
    # D_invsqrt = d_invsqrt.diag(name="D_invsqrt")
    # A_weighted = (D_invsqrt @ A).new(name=name)
    # A_weighted @= D_invsqrt

    d_invsqrt << unary.one(d_invsqrt)
    D = d_invsqrt.diag(name="D")
    A_weighted << D - A_weighted
    return A_weighted
