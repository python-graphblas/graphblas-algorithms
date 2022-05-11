from graphblas import binary
from graphblas.semiring import plus_pair
from networkx import average_clustering as _nx_average_clustering
from networkx import clustering as _nx_clustering

from graphblas_algorithms.classes.digraph import to_graph
from graphblas_algorithms.classes.graph import to_undirected_graph
from graphblas_algorithms.utils import not_implemented_for


def single_triangle_core(G, node):
    index = G._key_to_id[node]
    r = G._A[index, :].new()
    # Pretty much all the time is spent here taking TRIL, which is used to ignore self-edges
    L = G.get_property("L-")
    if G.get_property("has_self_edges"):
        del r[index]  # Ignore self-edges
    return plus_pair(L @ r).new(mask=r.S).reduce(allow_empty=False).value


def triangles_core(G, mask=None):
    # Ignores self-edges
    L, U = G.get_properties("L- U-")
    C = plus_pair(L @ L.T).new(mask=L.S)
    return (
        C.reduce_rowwise().new(mask=mask)
        + C.reduce_columnwise().new(mask=mask)
        + plus_pair(U @ L.T).new(mask=U.S).reduce_rowwise().new(mask=mask)
    ).new(name="triangles")


@not_implemented_for("directed")
def triangles(G, nodes=None):
    G = to_undirected_graph(G, dtype=bool)
    if len(G) == 0:
        return {}
    if nodes in G:
        return single_triangle_core(G, nodes)
    mask = G.list_to_mask(nodes)
    result = triangles_core(G, mask=mask)
    return G.vector_to_dict(result, mask=mask, fillvalue=0)


def total_triangles_core(G):
    # We use SandiaDot method, because it's usually the fastest on large graphs.
    # For smaller graphs, Sandia method is usually faster: plus_pair(L @ L).new(mask=L.S)
    L, U = G.get_properties("L- U-")
    return plus_pair(L @ U.T).new(mask=L.S).reduce_scalar(allow_empty=False).value


def transitivity_core(G):
    numerator = total_triangles_core(G)
    if numerator == 0:
        return 0
    degrees = G.get_property("degrees-")
    denom = (degrees * (degrees - 1)).reduce().value
    return 6 * numerator / denom


def transitivity_directed_core(G):
    # XXX" is transitivity supposed to work on directed graphs like this?
    if G.get_property("has_self_edges"):
        A = G.get_property("offdiag")
    else:
        A = G._A
    numerator = plus_pair(A @ A.T).new(mask=A.S).reduce_scalar(allow_empty=False).value
    if numerator == 0:
        return 0
    deg = A.reduce_rowwise("count")
    denom = (deg * (deg - 1)).reduce().value
    return numerator / denom


def transitivity(G):
    G = to_graph(G, dtype=bool)  # directed or undirected
    if len(G) == 0:
        return 0
    if G.is_directed():
        return transitivity_directed_core(G)
    else:
        return transitivity_core(G)


def clustering_core(G, mask=None):
    tri = triangles_core(G, mask=mask)
    degrees = G.get_property("degrees-")
    denom = degrees * (degrees - 1)
    return (2 * tri / denom).new(name="clustering")


def clustering_directed_core(G, mask=None):
    if G.get_property("has_self_edges"):
        A = G.get_property("offdiag")
    else:
        A = G._A
    AT = G.get_property("AT")
    temp = plus_pair(A @ A.T).new(mask=A.S)
    tri = (
        temp.reduce_rowwise().new(mask=mask)
        + temp.reduce_columnwise().new(mask=mask)
        + plus_pair(AT @ A.T).new(mask=A.S).reduce_rowwise().new(mask=mask)
        + plus_pair(AT @ AT.T).new(mask=A.S).reduce_columnwise().new(mask=mask)
    )
    recip_degrees = binary.pair(A & AT).reduce_rowwise().new(mask=mask)
    total_degrees = A.reduce_rowwise("count").new(mask=mask) + A.reduce_columnwise("count").new(
        mask=mask
    )
    return (tri / (total_degrees * (total_degrees - 1) - 2 * recip_degrees)).new(name="clustering")


def single_clustering_core(G, node):
    tri = single_triangle_core(G, node)
    if tri == 0:
        return 0
    index = G._key_to_id[node]
    if "degrees-" in G._cache:
        degrees = G.get_property("degrees-")[index].value
    elif "degrees+" in G._cache:
        degrees = G.get_property("degrees+")[index].value
        if G.get_property("has_self_edges") and G._A[index, index].value is not None:
            degrees -= 1
    else:
        row = G._A[index, :].new()
        degrees = row.nvals
        if G.get_property("has_self_edges") and row[index].value is not None:
            degrees -= 1
    denom = degrees * (degrees - 1)
    return 2 * tri / denom


def single_clustering_directed_core(G, node, *, has_self_edges=True):
    if G.get_property("has_self_edges"):
        A = G.get_property("offdiag")
    else:
        A = G._A
    index = G._key_to_id[node]
    r = A[index, :].new()
    c = A[:, index].new()
    tri = (
        plus_pair(A @ c).new(mask=c.S).reduce(allow_empty=False).value
        + plus_pair(A @ c).new(mask=r.S).reduce(allow_empty=False).value
        + plus_pair(A @ r).new(mask=c.S).reduce(allow_empty=False).value
        + plus_pair(A @ r).new(mask=r.S).reduce(allow_empty=False).value
    )
    if tri == 0:
        return 0
    total_degrees = c.nvals + r.nvals
    recip_degrees = binary.pair(c & r).nvals
    return tri / (total_degrees * (total_degrees - 1) - 2 * recip_degrees)


def clustering(G, nodes=None, weight=None):
    if weight is not None:
        # TODO: Not yet implemented.  Clustering implemented only for unweighted.
        return _nx_clustering(G, nodes=nodes, weight=weight)
    G = to_graph(G, weight=weight)  # to directed or undirected
    if len(G) == 0:
        return {}
    if nodes in G:
        if G.is_directed():
            return single_clustering_directed_core(G, nodes)
        else:
            return single_clustering_core(G, nodes)
    mask = G.list_to_mask(nodes)
    if G.is_directed():
        result = clustering_directed_core(G, mask=mask)
    else:
        result = clustering_core(G, mask=mask)
    return G.vector_to_dict(result, mask=mask, fillvalue=0.0)


def average_clustering_core(G, mask=None, count_zeros=True):
    c = clustering_core(G, mask=mask)
    val = c.reduce(allow_empty=False).value
    if not count_zeros:
        return val / c.nvals
    elif mask is not None:
        return val / mask.parent.nvals
    else:
        return val / c.size


def average_clustering_directed_core(G, mask=None, count_zeros=True):
    c = clustering_directed_core(G, mask=mask)
    val = c.reduce(allow_empty=False).value
    if not count_zeros:
        return val / c.nvals
    elif mask is not None:
        return val / mask.parent.nvals
    else:
        return val / c.size


def average_clustering(G, nodes=None, weight=None, count_zeros=True):
    if weight is not None:
        # TODO: Not yet implemented.  Clustering implemented only for unweighted.
        return _nx_average_clustering(G, nodes=nodes, weight=weight, count_zeros=count_zeros)
    G = to_graph(G, weight=weight)  # to directed or undirected
    if len(G) == 0:
        raise ZeroDivisionError()  # Not covered
    mask = G.list_to_mask(nodes)
    if G.is_directed():
        return average_clustering_directed_core(G, mask=mask, count_zeros=count_zeros)
    else:
        return average_clustering_core(G, mask=mask, count_zeros=count_zeros)
