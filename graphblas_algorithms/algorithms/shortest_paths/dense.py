from graphblas import Matrix, Vector, binary, indexunary, replace, select
from graphblas.semiring import any_plus, any_second

__all__ = ["floyd_warshall", "floyd_warshall_predecessor_and_distance"]


def floyd_warshall(G, is_weighted=False):
    return floyd_warshall_predecessor_and_distance(G, is_weighted, compute_predecessors=False)[1]


def floyd_warshall_predecessor_and_distance(G, is_weighted=False, *, compute_predecessors=True):
    # By using `offdiag` instead of `G._A`, we ensure that D will not become dense.
    # Dense D may be better at times, but not including the diagonal will result in less work.
    # Typically, Floyd-Warshall algorithms sets the diagonal of D to 0 at the beginning.
    # This is unnecessary with sparse matrices, and we set the diagonal to 0 at the end.
    # We also don't iterate over index `i` if either row i or column i are empty.
    if is_directed := G.is_directed():
        A, row_degrees, column_degrees = G.get_properties("offdiag row_degrees- column_degrees-")
        nonempty_nodes = binary.pair(row_degrees & column_degrees).new(name="nonempty_nodes")
    else:
        A, nonempty_nodes = G.get_properties("U- degrees-")

    if A.dtype == bool or not is_weighted:
        dtype = int
    else:
        dtype = A.dtype
    n = A.nrows
    D = Matrix(dtype, nrows=n, ncols=n, name="floyd_warshall_dist")
    if is_weighted:
        D << A
    else:
        D(A.S) << 1  # Like `D << unary.one[int](A)`
    del A
    Row = Matrix(dtype, nrows=1, ncols=n, name="Row")
    if is_directed:
        Col = Matrix(dtype, nrows=n, ncols=1, name="Col")
    else:
        Col = None
    Outer = Matrix(dtype, nrows=n, ncols=n, name="Outer")
    if compute_predecessors:
        Mask = Matrix(bool, nrows=n, ncols=n, name="Mask")
        P = indexunary.rowindex(D).new(name="floyd_warshall_pred")
        if P.dtype == dtype:
            P_row = Row
        else:
            P_row = Matrix(P.dtype, nrows=1, ncols=n, name="P_row")
    else:
        Mask = P = P_row = None

    for i in nonempty_nodes:
        Row << D[[i], :]
        if is_directed:
            Col << D[:, [i]]
        else:
            Row(binary.any) << D.T[[i], :]
            Col = Row.T
        Outer << any_plus(Col @ Row)  # Like `col.outer(row, binary.plus)`

        if not compute_predecessors:
            # It is faster (approx 10%-30%) to use a mask as is done below when computing
            # predecessors, but we choose to use less memory here by not using a mask.
            if is_directed:
                D(binary.min) << select.offdiag(Outer)
            else:
                D(binary.min) << select.triu(Outer, 1)
        else:
            # Update Outer to only include off-diagonal values that will update D and P.
            if is_directed:
                Mask << indexunary.offdiag(Outer)
            else:
                Mask << indexunary.triu(Outer, 1)
            Mask(binary.second) << binary.lt(Outer & D)
            Outer(Mask.V, replace) << Outer

            # Update distances; like `D(binary.min) << offdiag(any_plus(Col @ Row))`
            D(Outer.S) << Outer

            # Broadcast predecessors in P_row to updated values
            P_row << P[[i], :]
            if not is_directed:
                P_row(binary.any) << P.T[[i], :]
                Col = P_row.T
            P(Outer.S) << any_second(Col @ P_row)
    del Outer, Mask, Col, Row, P_row

    if not is_directed:
        # Symmetrize the results.
        # It may be nice to be able to return these as upper-triangular.
        D(binary.any) << D.T
        if compute_predecessors:
            P(binary.any) << P.T

    # Set diagonal values to 0 (this way seems fast).
    # The missing values are implied to be infinity, so we set diagonals explicitly to 0.
    diag_mask = Vector(bool, size=n, name="diag_mask")
    diag_mask << True
    Diag_mask = diag_mask.diag(name="Diag_mask")
    D(Diag_mask.S) << 0

    return P, D
