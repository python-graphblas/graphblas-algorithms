from graphblas import binary, unary
from graphblas.semiring import plus_pair, plus_times

from graphblas_algorithms.classes.digraph import to_graph
from graphblas_algorithms.classes.graph import to_undirected_graph
from graphblas_algorithms.utils import not_implemented_for


def single_triangle_core(G, node, *, weighted=False):
    index = G._key_to_id[node]
    r = G._A[index, :].new()
    # Pretty much all the time is spent here taking TRIL, which is used to ignore self-edges
    L = G.get_property("L-")
    if G.get_property("has_self_edges"):
        del r[index]  # Ignore self-edges
    if weighted:
        maxval = G.get_property("max_element-")
        L = unary.cbrt(L / maxval)
        r = unary.cbrt(r / maxval)
        semiring = plus_times
    else:
        semiring = plus_pair
    val = semiring(L @ r).new(mask=r.S)
    if weighted:
        val *= r
    return val.reduce(allow_empty=False).value


def triangles_core(G, *, weighted=False, mask=None):
    # Ignores self-edges
    L, U = G.get_properties("L- U-")
    if weighted:
        maxval = G.get_property("max_element-")
        L = unary.cbrt(L / maxval)
        U = unary.cbrt(U / maxval)
        semiring = plus_times
    else:
        semiring = plus_pair
    C = semiring(L @ L.T).new(mask=L.S)
    D = semiring(U @ L.T).new(mask=U.S)
    if weighted:
        C *= L
        D *= U
    return (
        C.reduce_rowwise().new(mask=mask)
        + C.reduce_columnwise().new(mask=mask)
        + D.reduce_rowwise().new(mask=mask)
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
    A, AT = G.get_properties("offdiag AT")
    numerator = plus_pair(A @ A.T).new(mask=A.S).reduce_scalar(allow_empty=False).value
    if numerator == 0:
        return 0
    degrees = G.get_property("row_degrees-")
    denom = (degrees * (degrees - 1)).reduce().value
    return numerator / denom


def transitivity(G):
    G = to_graph(G, dtype=bool)  # directed or undirected
    if len(G) == 0:
        return 0
    if G.is_directed():
        func = transitivity_directed_core
    else:
        func = transitivity_core
    return G._cacheit("transitivity", func, G)


def clustering_core(G, *, weighted=False, mask=None):
    tri = triangles_core(G, weighted=weighted, mask=mask)
    degrees = G.get_property("degrees-")
    denom = degrees * (degrees - 1)
    return (2 * tri / denom).new(name="clustering")


def clustering_directed_core(G, *, weighted=False, mask=None):
    A, AT = G.get_properties("offdiag AT")
    if weighted:
        maxval = G.get_property("max_element-")
        A = unary.cbrt(A / maxval)
        AT = unary.cbrt(AT / maxval)
        semiring = plus_times
    else:
        semiring = plus_pair
    C = semiring(A @ A.T).new(mask=A.S)
    D = semiring(AT @ A.T).new(mask=A.S)
    E = semiring(AT @ AT.T).new(mask=A.S)
    if weighted:
        C *= A
        D *= A
        E *= A
    tri = (
        C.reduce_rowwise().new(mask=mask)
        + C.reduce_columnwise().new(mask=mask)
        + D.reduce_rowwise().new(mask=mask)
        + E.reduce_columnwise().new(mask=mask)
    )
    recip_degrees, total_degrees = G.get_properties("recip_degrees- total_degrees-", mask=mask)
    return (tri / (total_degrees * (total_degrees - 1) - 2 * recip_degrees)).new(name="clustering")


def single_clustering_core(G, node, *, weighted=False):
    tri = single_triangle_core(G, node, weighted=weighted)
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
        row = G._A[index, :]
        degrees = row.nvals
        if G.get_property("has_self_edges") and row[index].value is not None:
            degrees -= 1
    denom = degrees * (degrees - 1)
    return 2 * tri / denom


def single_clustering_directed_core(G, node, *, weighted=False):
    A = G.get_property("offdiag")
    index = G._key_to_id[node]
    if weighted:
        maxval = G.get_property("max_element-")
        A = unary.cbrt(A / maxval)
        semiring = plus_times
    else:
        semiring = plus_pair
    r = A[index, :]
    c = A[:, index]
    v1 = semiring(A @ c).new(mask=c.S)
    v2 = semiring(A @ c).new(mask=r.S)
    v3 = semiring(A @ r).new(mask=c.S)
    v4 = semiring(A @ r).new(mask=r.S)
    if weighted:
        v1 *= c
        v2 *= r
        v3 *= c
        v4 *= r
    tri = (
        v1.reduce(allow_empty=False).value
        + v2.reduce(allow_empty=False).value
        + v3.reduce(allow_empty=False).value
        + v4.reduce(allow_empty=False).value
    )
    if tri == 0:
        return 0
    total_degrees = c.nvals + r.nvals
    recip_degrees = binary.pair(c & r).nvals
    return tri / (total_degrees * (total_degrees - 1) - 2 * recip_degrees)


def clustering(G, nodes=None, weight=None):
    G = to_graph(G, weight=weight)  # to directed or undirected
    if len(G) == 0:
        return {}
    weighted = weight is not None
    if nodes in G:
        if G.is_directed():
            return single_clustering_directed_core(G, nodes, weighted=weighted)
        else:
            return single_clustering_core(G, nodes, weighted=weighted)
    mask = G.list_to_mask(nodes)
    if G.is_directed():
        result = clustering_directed_core(G, weighted=weighted, mask=mask)
    else:
        result = clustering_core(G, weighted=weighted, mask=mask)
    return G.vector_to_dict(result, mask=mask, fillvalue=0.0)


def average_clustering_core(G, *, count_zeros=True, weighted=False, mask=None):
    c = clustering_core(G, weighted=weighted, mask=mask)
    val = c.reduce(allow_empty=False).value
    if not count_zeros:
        return val / c.nvals
    elif mask is not None:
        return val / mask.parent.nvals
    else:
        return val / c.size


def average_clustering_directed_core(G, *, count_zeros=True, weighted=False, mask=None):
    c = clustering_directed_core(G, weighted=weighted, mask=mask)
    val = c.reduce(allow_empty=False).value
    if not count_zeros:
        return val / c.nvals
    elif mask is not None:
        return val / mask.parent.nvals
    else:
        return val / c.size


def average_clustering(G, nodes=None, weight=None, count_zeros=True):
    G = to_graph(G, weight=weight)  # to directed or undirected
    if len(G) == 0:
        raise ZeroDivisionError()  # Not covered
    weighted = weight is not None
    mask = G.list_to_mask(nodes)
    if G.is_directed():
        func = average_clustering_directed_core
    else:
        func = average_clustering_core
    if mask is None:
        return G._cacheit(
            f"average_clustering(count_zeros={count_zeros})",
            func,
            G,
            weighted=weighted,
            count_zeros=count_zeros,
        )
    else:
        return func(G, weighted=weighted, count_zeros=count_zeros, mask=mask)
