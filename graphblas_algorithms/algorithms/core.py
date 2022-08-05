from graphblas import Matrix, select, monoid, semiring
from graphblas_algorithms.classes.graph import to_undirected_graph, Graph
from graphblas_algorithms.utils import get_all, not_implemented_for


def k_truss_core(G, k):
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
            C(S.S, replace=True) << plus_pair(S @ S.T)
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


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def k_truss(G, k):
    G = to_undirected_graph(G, dtype=bool)
    result = k_truss_core(G, k)
    return result.to_networkx()


__all__ = get_all(__name__)
