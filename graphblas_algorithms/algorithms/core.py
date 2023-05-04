from graphblas import Matrix, monoid, replace, select, semiring

from graphblas_algorithms import Graph

__all__ = ["k_truss"]


def k_truss(G: Graph, k) -> Graph:
    # TODO: should we have an option to keep the output matrix the same size?
    # Ignore self-edges
    S = G.get_property("offdiag")

    if k < 3:
        # Most implementations consider k < 3 invalid,
        # but networkx leaves the graph unchanged
        C = S
    else:
        # Remove edges not in k-truss
        nvals_last = S.nvals
        # TODO: choose dtype based on max number of triangles
        plus_pair = semiring.plus_pair["int32"]
        C = Matrix("int32", S.nrows, S.ncols)
        while True:
            C(S.S, replace) << plus_pair(S @ S.T)
            C << select.value(k - 2 <= C)
            if C.nvals == nvals_last:
                break
            nvals_last = C.nvals
            S = C

    # Remove isolate nodes
    indices, _ = C.reduce_rowwise(monoid.any).to_coo(values=False)
    Ktruss = C[indices, indices].new()

    # Convert back to networkx graph with correct node ids
    key_to_id = G.renumber_key_to_id(indices.tolist())
    return Graph(Ktruss, key_to_id=key_to_id)
