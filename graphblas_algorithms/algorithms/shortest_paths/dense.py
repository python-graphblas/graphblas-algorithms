from graphblas import Matrix, Vector, binary
from graphblas.semiring import any_plus

__all__ = ["floyd_warshall"]


def floyd_warshall(G, is_weighted=False):
    # By using `offdiag` instead of `G._A`, we ensure that D will not become dense.
    # Dense D may be better at times, but not including the diagonal will result in less work.
    # Typically, Floyd-Warshall algorithms sets the diagonal of D to 0 at the beginning.
    # This is unnecessary with sparse matrices, and we set the diagonal to 0 at the end.
    A = G.get_property("offdiag")
    if A.dtype == bool or not is_weighted:
        dtype = int
    else:
        dtype = A.dtype
    n = A.nrows
    D = Matrix(dtype, nrows=n, ncols=n, name="floyd_warshall")
    if is_weighted:
        D << A
    else:
        D(A.S) << 1  # Like `D << unary.one[int](A)`
    del A

    Row = Matrix(dtype, nrows=1, ncols=n, name="Row")
    Col = Matrix(dtype, nrows=n, ncols=1, name="Col")
    for i in range(n):
        Col << D[:, [i]]
        Row << D[[i], :]
        D(binary.min) << any_plus(Col @ Row)  # Like `col.outer(row, binary.plus)`

    # Set diagonal values to 0 (this way seems fast).
    # The missing values are implied to be infinity, so we set diagonals explicitly to 0.
    v = Vector(bool, size=n)
    v << True
    D(v.diag().S) << 0
    return D
