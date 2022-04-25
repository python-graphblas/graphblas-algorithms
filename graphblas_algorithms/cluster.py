from collections import OrderedDict

import graphblas as gb
from graphblas import Matrix, Vector, binary, select
from graphblas.semiring import any_pair, plus_pair
from networkx.utils import not_implemented_for


def single_triangle_core(G, index):
    M = Matrix(bool, G.nrows, G.ncols)
    M[index, index] = False
    C = any_pair(G.T @ M.T).new(name="C")  # select.coleq(G, index)
    del C[index, index]  # Ignore self-edges
    R = C.T.new(name="R")
    return plus_pair(G @ R.T).new(mask=C.S).reduce_scalar(allow_empty=False).value // 2


def triangles_core(G, mask=None):
    # Ignores self-edges
    L = select.tril(G, -1).new(name="L")
    U = select.triu(G, 1).new(name="U")
    C = plus_pair(L @ L.T).new(mask=L.S)
    return (
        C.reduce_rowwise().new(mask=mask)
        + C.reduce_columnwise().new(mask=mask)
        + plus_pair(U @ L.T).new(mask=U.S).reduce_rowwise().new(mask=mask)
    ).new(name="triangles")


def total_triangles_core(G):
    # Ignores self-edges
    L = select.tril(G, -1).new(name="L")
    U = select.triu(G, 1).new(name="U")
    return plus_pair(L @ U.T).new(mask=L.S).reduce_scalar(allow_empty=False).value


@not_implemented_for("directed")
def triangles(G, nodes=None):
    N = len(G)
    if N == 0:
        return {}
    node_ids = OrderedDict((k, i) for i, k in enumerate(G))
    A = gb.io.from_networkx(G, nodelist=node_ids, weight=None, dtype=bool)
    if nodes in G:
        return single_triangle_core(A, node_ids[nodes])
    if nodes is not None:
        id_to_key = {node_ids[key]: key for key in nodes}
        mask = Vector.from_values(list(id_to_key), True, size=N, dtype=bool, name="mask").S
    else:
        mask = None
    result = triangles_core(A, mask=mask)
    if nodes is not None:
        if result.nvals != len(id_to_key):
            result(mask, binary.first) << 0
        indices, values = result.to_values()
        return {id_to_key[index]: value for index, value in zip(indices, values)}
    elif result.nvals != N:
        # Fill with zero
        result(mask=~result.S) << 0
    return dict(zip(node_ids, result.to_values()[1]))
