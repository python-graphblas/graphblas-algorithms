from collections import OrderedDict

import graphblas as gb
from graphblas import Matrix, Vector, agg, binary, select
from graphblas.semiring import any_pair, plus_pair
from networkx.utils import not_implemented_for


def get_properties(G, names, *, L=None, U=None, degrees=None, has_self_edges=True):
    """Calculate properties of undirected graph"""
    if isinstance(names, str):
        # Separated by commas and/or spaces
        names = [name for name in names.replace(" ", ",").split(",") if name]
    rv = []
    for name in names:
        if name == "L":
            if L is None:
                L = select.tril(G, -1).new(name="L")
            rv.append(L)
        elif name == "U":
            if U is None:
                U = select.triu(G, 1).new(name="U")
            rv.append(U)
        elif name == "degrees":
            if degrees is None:
                if L is not None:
                    has_self_edges = G.nvals > 2 * L.nvals
                elif U is not None:
                    has_self_edges = G.nvals > 2 * U.nvals
                if has_self_edges:
                    if L is None or U is None:
                        L, U = get_properties(G, "L U", L=L, U=U)
                    degrees = (L.reduce_rowwise(agg.count) + U.reduce_rowwise(agg.count)).new(
                        name="degrees"
                    )
                else:
                    degrees = G.reduce_rowwise(agg.count).new(name="degrees")
            rv.append(degrees)
        elif name == "has_self_edges":
            # Compute if cheap
            if L is not None:
                has_self_edges = G.nvals > 2 * L.nvals
            elif U is not None:
                has_self_edges = G.nvals > 2 * U.nvals
            rv.append(has_self_edges)
        else:
            raise ValueError(f"Unknown property name: {name}")
    if len(rv) == 1:
        return rv[0]
    return rv


def single_triangle_core(G, index, *, L=None, has_self_edges=True):
    M = Matrix(bool, G.nrows, G.ncols)
    M[index, index] = False
    C = any_pair(G.T @ M.T).new(name="C")  # select.coleq(G.T, index)
    if has_self_edges:
        del C[index, index]  # Ignore self-edges
    R = C.T.new(name="R")
    has_self_edges = get_properties(G, "has_self_edges", L=L, has_self_edges=has_self_edges)
    if has_self_edges:
        # Pretty much all the time is spent here taking TRIL, which is used to ignore self-edges
        L = get_properties(G, "L", L=L)
        return plus_pair(L @ R.T).new(mask=C.S).reduce_scalar(allow_empty=False).value
    else:
        return plus_pair(G @ R.T).new(mask=C.S).reduce_scalar(allow_empty=False).value // 2


def triangles_core(G, mask=None, *, L=None, U=None):
    # Ignores self-edges
    L, U = get_properties(G, "L U", L=L, U=U)
    C = plus_pair(L @ L.T).new(mask=L.S)
    return (
        C.reduce_rowwise().new(mask=mask)
        + C.reduce_columnwise().new(mask=mask)
        + plus_pair(U @ L.T).new(mask=U.S).reduce_rowwise().new(mask=mask)
    ).new(name="triangles")


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


def total_triangles_core(G, *, L=None, U=None):
    # Ignores self-edges
    # We use SandiaDot method, because it's usually the fastest on large graphs.
    # For smaller graphs, Sandia method is usually faster: plus_pair(L @ L).new(mask=L.S)
    L, U = get_properties(G, "L U", L=L, U=U)
    return plus_pair(L @ U.T).new(mask=L.S).reduce_scalar(allow_empty=False).value


def transitivity_core(G, *, L=None, U=None, degrees=None):
    L, U = get_properties(G, "L U", L=L, U=U)
    numerator = total_triangles_core(G, L=L, U=U)
    if numerator == 0:
        return 0
    degrees = get_properties(G, "degrees", L=L, U=U, degrees=degrees)
    denom = (degrees * (degrees - 1)).reduce().value
    return 6 * numerator / denom


@not_implemented_for("directed")  # Should we implement it for directed?
def transitivity(G):
    if len(G) == 0:
        return 0
    A = gb.io.from_networkx(G, weight=None, dtype=bool)
    return transitivity_core(A)


def clustering_core(G, *, L=None, U=None, degrees=None):
    L, U, degrees = get_properties(G, "L U degrees", L=L, U=U, degrees=degrees)
    tri = triangles_core(G, L=L, U=U)
    denom = degrees * (degrees - 1)
    return (2 * tri / denom).new(name="clustering")


@not_implemented_for("directed")  # TODO: implement for directed
def clustering(G, nodes=None, weight=None):
    pass
