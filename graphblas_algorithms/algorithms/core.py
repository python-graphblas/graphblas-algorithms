from graphblas import Matrix, monoid, replace, select, semiring

from graphblas_algorithms.classes.graph import Graph

__all__ = ["k_truss"]


def k_truss(G: Graph, k) -> Graph:
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
            C << select.value(C >= k - 2)
            if C.nvals == nvals_last:
                break
            nvals_last = C.nvals
            S = C

    # Remove isolate nodes
    indices, _ = C.reduce_rowwise(monoid.any).to_values()
    Ktruss = C[indices, indices].new()

    # Convert back to networkx graph with correct node ids
    keys = G.list_to_keys(indices)
    key_to_id = dict(zip(keys, range(len(indices))))
    return Graph.from_graphblas(Ktruss, key_to_id=key_to_id)
