from graphblas import Vector, binary

__all__ = ["bethe_hessian_matrix"]


def bethe_hessian_matrix(G, r=None, nodelist=None, *, name="bethe_hessian_matrix"):
    A = G._A
    if nodelist is not None:
        ids = G.list_to_ids(nodelist)
        A = A[ids, ids].new()
        d = A.reduce_rowwise().new(name="d")
    else:
        d = G.get_property("plus_rowwise+")
    if r is None:
        degrees = G.get_property("degrees+")
        k = degrees.reduce().get(0)
        k2 = (degrees @ degrees).get(0)
        r = k2 / k - 1
    n = A.nrows
    # result = (r**2 - 1) * I - r * A + D
    ri = Vector.from_scalar(r**2 - 1.0, n, name="ri")
    ri += d
    rI = ri.diag(name=name)
    rI(binary.plus) << binary.times(-r, A)  # rI += -r * A
    return rI
