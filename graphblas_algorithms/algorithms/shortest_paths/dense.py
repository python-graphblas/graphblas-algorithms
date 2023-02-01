from graphblas import Matrix, Vector, binary
from graphblas.select import offdiag
from graphblas.semiring import any_plus

__all__ = ["floyd_warshall"]


def floyd_warshall(G, is_weighted=False):
    # By using `offdiag` instead of `G._A`, we ensure that D will not become dense.
    # Dense D may be better at times, but not including the diagonal will result in less work.
    # Typically, Floyd-Warshall algorithms sets the diagonal of D to 0 at the beginning.
    # This is unnecessary with sparse matrices, and we set the diagonal to 0 at the end.
    # We also don't iterate over index `i` if either row i or column i are empty.
    if G.is_directed():
        A, row_degrees, column_degrees = G.get_properties("offdiag row_degrees- column_degrees-")
        nonempty_nodes = binary.pair(row_degrees & column_degrees).new(name="nonempty_nodes")
    else:
        A, nonempty_nodes = G.get_properties("offdiag degrees-")

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
    Outer = Matrix(dtype, nrows=n, ncols=n, name="Outer")
    for i in nonempty_nodes:
        Col << D[:, [i]]
        Row << D[[i], :]
        Outer << any_plus(Col @ Row)  # Like `col.outer(row, binary.plus)`
        D(binary.min) << offdiag(Outer)

    # Set diagonal values to 0 (this way seems fast).
    # The missing values are implied to be infinity, so we set diagonals explicitly to 0.
    mask = Vector(bool, size=n, name="mask")
    mask << True
    Mask = mask.diag(name="Mask")
    D(Mask.S) << 0
    return D
