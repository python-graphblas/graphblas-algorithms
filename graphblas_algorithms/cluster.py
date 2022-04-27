import graphblas as gb
import networkx as nx
from graphblas import Matrix, agg, select
from graphblas.semiring import any_pair, plus_pair
from networkx import average_clustering as _nx_average_clustering
from networkx import clustering as _nx_clustering
from networkx.utils import not_implemented_for

from ._utils import graph_to_adjacency, list_to_mask, vector_to_dict


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
                degrees = get_degrees(G, L=L, U=U, has_self_edges=has_self_edges)
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


def get_degrees(G, mask=None, *, L=None, U=None, has_self_edges=True):
    if L is not None:
        has_self_edges = G.nvals > 2 * L.nvals
    elif U is not None:
        has_self_edges = G.nvals > 2 * U.nvals
    if has_self_edges:
        if L is None or U is None:
            L, U = get_properties(G, "L U", L=L, U=U)
        degrees = (
            L.reduce_rowwise(agg.count).new(mask=mask) + U.reduce_rowwise(agg.count).new(mask=mask)
        ).new(name="degrees")
    else:
        degrees = G.reduce_rowwise(agg.count).new(mask=mask, name="degrees")
    return degrees


def single_triangle_core(G, index, *, L=None, has_self_edges=True):
    M = Matrix(bool, G.nrows, G.ncols)
    M[index, index] = True
    C = any_pair(G.T @ M.T).new(name="C")  # select.coleq(G.T, index)
    has_self_edges = get_properties(G, "has_self_edges", L=L, has_self_edges=has_self_edges)
    if has_self_edges:
        del C[index, index]  # Ignore self-edges
    R = C.T.new(name="R")
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
    if len(G) == 0:
        return {}
    A, key_to_id = graph_to_adjacency(G, dtype=bool)
    if nodes in G:
        return single_triangle_core(A, key_to_id[nodes])
    mask, id_to_key = list_to_mask(nodes, key_to_id)
    result = triangles_core(A, mask=mask)
    return vector_to_dict(result, key_to_id, id_to_key, mask=mask, fillvalue=0)


def total_triangles_core(G, *, L=None, U=None):
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


def clustering_core(G, mask=None, *, L=None, U=None, degrees=None):
    L, U = get_properties(G, "L U", L=L, U=U)
    tri = triangles_core(G, mask=mask, L=L, U=U)
    degrees = get_degrees(G, mask=mask, L=L, U=U)
    denom = degrees * (degrees - 1)
    return (2 * tri / denom).new(name="clustering")


def single_clustering_core(G, index, *, L=None, degrees=None, has_self_edges=True):
    has_self_edges = get_properties(G, "has_self_edges", L=L, has_self_edges=has_self_edges)
    tri = single_triangle_core(G, index, L=L, has_self_edges=has_self_edges)
    if tri == 0:
        return 0
    if degrees is not None:
        degrees = degrees[index].value
    else:
        row = G[index, :].new()
        degrees = row.reduce(agg.count).value
        if has_self_edges and row[index].value is not None:
            degrees -= 1
    denom = degrees * (degrees - 1)
    return 2 * tri / denom


def clustering(G, nodes=None, weight=None):
    if len(G) == 0:
        return {}
    if isinstance(G, nx.DiGraph) or weight is not None:
        # TODO: Not yet implemented.  Clustering implemented only for undirected and unweighted.
        return _nx_clustering(G, nodes=nodes, weight=weight)
    A, key_to_id = graph_to_adjacency(G, weight=weight)
    if nodes in G:
        return single_clustering_core(A, key_to_id[nodes])
    mask, id_to_key = list_to_mask(nodes, key_to_id)
    result = clustering_core(A, mask=mask)
    return vector_to_dict(result, key_to_id, id_to_key, mask=mask, fillvalue=0.0)


def average_clustering_core(G, mask=None, count_zeros=True, *, L=None, U=None, degrees=None):
    c = clustering_core(G, mask=mask, L=L, U=U, degrees=degrees)
    val = c.reduce(allow_empty=False).value
    if not count_zeros:
        return val / c.nvals
    elif mask is not None:
        return val / mask.parent.nvals
    else:
        return val / c.size


def average_clustering(G, nodes=None, weight=None, count_zeros=True):
    if len(G) == 0 or isinstance(G, nx.DiGraph) or weight is not None:
        # TODO: Not yet implemented.  Clustering implemented only for undirected and unweighted.
        return _nx_average_clustering(G, nodes=nodes, weight=weight, count_zeros=count_zeros)
    A, key_to_id = graph_to_adjacency(G, weight=weight)
    mask, _ = list_to_mask(nodes, key_to_id)
    return average_clustering_core(A, mask=mask, count_zeros=count_zeros)
