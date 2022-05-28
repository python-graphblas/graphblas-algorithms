from graphblas import Matrix, Vector, binary, monoid, replace, select, unary
from graphblas.semiring import plus_pair, plus_times

from graphblas_algorithms.classes.digraph import to_graph
from graphblas_algorithms.classes.graph import to_undirected_graph
from graphblas_algorithms.utils import get_all, not_implemented_for


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
    return val.reduce().get(0)


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
    return plus_pair(L @ U.T).new(mask=L.S).reduce_scalar().get(0)


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
    numerator = plus_pair(A @ A.T).new(mask=A.S).reduce_scalar().get(0)
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
    # TODO: it would be nice if we could clean this up, but still be fast
    if "degrees-" in G._cache:
        degrees = G.get_property("degrees-").get(index)
    elif "degrees+" in G._cache:
        degrees = G.get_property("degrees+").get(index)
        if G.get_property("has_self_edges") and G._A.get(index, index) is not None:
            degrees -= 1
    else:
        row = G._A[index, :]
        degrees = row.nvals
        if G.get_property("has_self_edges") and row.get(index) is not None:
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
    tris = []
    for x, y in [(c, c), (c, r), (r, c), (r, r)]:
        v = semiring(A @ x).new(mask=y.S)
        if weighted:
            v *= y
        tris.append(v.reduce().new())
    # Getting Python scalars are blocking operations, so we do them last
    tri = sum(t.get(0) for t in tris)
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
    val = c.reduce().get(0)
    if not count_zeros:
        return val / c.nvals
    elif mask is not None:
        return val / mask.parent.nvals
    else:
        return val / c.size


def average_clustering_directed_core(G, *, count_zeros=True, weighted=False, mask=None):
    c = clustering_directed_core(G, weighted=weighted, mask=mask)
    val = c.reduce().get(0)
    if not count_zeros:
        return val / c.nvals
    elif mask is not None:
        return val / mask.parent.nvals
    else:
        return val / c.size


def average_clustering(G, nodes=None, weight=None, count_zeros=True):
    G = to_graph(G, weight=weight)  # to directed or undirected
    if len(G) == 0:
        raise ZeroDivisionError()
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


def square_clustering_core(G, node_ids=None):
    # node_ids argument is a bit different from what we do elsewhere.
    # Normally, we take a mask or vector in a `_core` function.
    # By accepting an iterable here, it could be of node ids or node keys.
    A, degrees = G.get_properties("A degrees+")  # TODO" how to handle self-edges?
    if node_ids is None:
        # Can we do this better using SuiteSparse:GraphBLAS iteration?
        node_ids = A.reduce_rowwise(monoid.any).new(name="node_ids")  # all nodes with edges
    C = unary.positionj(A).new(name="C")
    rv = Vector(float, A.nrows, name="square_clustering")
    row = Vector(A.dtype, A.ncols, name="row")
    M = Matrix(int, A.nrows, A.ncols, name="M")
    Q = Matrix(int, A.nrows, A.ncols, name="Q")
    for v in node_ids:
        # Th mask M indicates the u and w neighbors of v to "iterate" over
        row << A[v, :]
        M << row.outer(row, binary.pair)
        M << select.tril(M, -1)
        # To compute q_v(u, w), the number of common neighbors of u and w other than v (squares),
        # we first set the v'th column to zero, which lets us ignore v as a common neighbor.
        Q << binary.isne(C, v)  # `isne` keeps the dtype as int
        # Q: count the number of squares for each u-w combination!
        Q(M.S, replace) << plus_times(Q @ Q.T)
        # Total squares for v
        squares = Q.reduce_scalar().get(0)
        if squares == 0:
            rv[v] = 0
            continue
        # Denominator is the total number of squares that could exist.
        # First contribution is degrees[u] + degrees[w] for each u-w combo.
        Q(M.S, replace) << degrees.outer(degrees, binary.plus)
        deg_uw = Q.reduce_scalar().new()
        # Then we subtract off # squares, 1 for each u and 1 for each w for all combos,
        # and 1 for each edge where u-w or w-u are connected (which would make triangles).
        Q << binary.pair(A & M)  # Are u-w connected?  Can skip if bipartite
        denom = deg_uw.get(0) - (squares + 2 * M.nvals + 2 * Q.nvals)
        rv[v] = squares / denom
    return rv


def square_clustering(G, nodes=None):
    G = to_undirected_graph(G)
    if len(G) == 0:
        return {}
    if nodes in G:
        idx = G._key_to_id[nodes]
        result = square_clustering_core(G, [idx])
        return result.get(idx)
    ids = G.list_to_ids(nodes)
    result = square_clustering_core(G, ids)
    return G.vector_to_dict(result)


__all__ = get_all(__name__)
