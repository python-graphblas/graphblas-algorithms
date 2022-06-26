import numpy as np
from graphblas import Matrix, Vector, binary, monoid, replace, unary
from graphblas.semiring import plus_first, plus_pair, plus_times

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
    # Can we apply the mask earlier in the computation?
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
    # Can we apply the mask earlier in the computation?
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
    denom = total_degrees * (total_degrees - 1) - 2 * recip_degrees
    return (tri / denom).new(name="clustering")


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


def single_square_clustering_core(G, idx):
    A, degrees = G.get_properties("A degrees+")  # TODO" how to handle self-edges?
    deg = degrees.get(idx, 0)
    if deg <= 1:
        return 0
    # P2 from https://arxiv.org/pdf/2007.11111.pdf; we'll also use it as scratch
    v = A[idx, :].new(name="v")
    p2 = plus_pair(A.T @ v).new(name="p2")
    del p2[idx]
    # Denominator is thought of as the total number of squares that could exist.
    # We use the definition from https://arxiv.org/pdf/0710.0117v1.pdf (equation 2).
    #
    # (1) Subtract 1 for each edge where u-w or w-u are connected (which would make triangles)
    denom = -plus_first(p2 @ v).get(0)
    # Numerator: number of squares
    # Based on https://arxiv.org/pdf/2007.11111.pdf (sigma_12, c_4)
    p2(binary.times) << p2 - 1
    squares = p2.reduce().get(0) // 2
    if squares == 0:
        return 0
    # (2) Subtract 1 for each u and 1 for each w for all combos: degrees * (degrees - 1)
    denom -= deg * (deg - 1)
    # (3) The main contribution to the denominator: degrees[u] + degrees[w] for each u-w combo.
    # This is the only positive term.
    denom += plus_times(v @ degrees).value * (deg - 1)
    # (4) Subtract the number of squares
    denom -= squares
    # And we're done!
    return squares / denom


def square_clustering_core(G, node_ids=None):
    # Warning: only tested on undirected graphs.
    # Also, it may use a lot of memory, because we compute `P2 = A @ A.T`
    #
    # Pseudocode:
    #   P2(~degrees.diag().S) = plus_pair(A @ A.T)
    #   tri = first(P2 & A).reduce_rowwise()
    #   squares = (P2 * (P2 - 1)).reduce_rowwise() / 2
    #   uw_count = degrees * (degrees - 1)
    #   uw_degrees = plus_times(A @ degrees) * (degrees - 1)
    #   square_clustering = squares / (uw_degrees - uw_count - tri - squares)
    #
    A, degrees = G.get_properties("A degrees+")  # TODO" how to handle self-edges?
    # P2 from https://arxiv.org/pdf/2007.11111.pdf; we'll also use it as scratch
    if node_ids is not None:
        v = Vector.from_values(node_ids, True, size=degrees.size)
        Asubset = binary.second(v & A).new(name="A_subset")
    else:
        Asubset = A
        A = A.T
    D = degrees.diag(name="D")
    P2 = plus_pair(Asubset @ A).new(mask=~D.S, name="P2")

    # Denominator is thought of as the total number of squares that could exist.
    # We use the definition from https://arxiv.org/pdf/0710.0117v1.pdf (equation 2).
    #   denom = uw_degrees - uw_count - tri - squares
    #
    # (1) Subtract 1 for each edge where u-w or w-u are connected (i.e., triangles)
    #   tri = first(P2 & A).reduce_rowwise()
    D << binary.first(P2 & Asubset)
    neg_denom = D.reduce_rowwise().new(name="neg_denom")
    del D

    # Numerator: number of squares
    # Based on https://arxiv.org/pdf/2007.11111.pdf (sigma_12, c_4)
    #   squares = (P2 * (P2 - 1)).reduce_rowwise() / 2
    P2(binary.times) << P2 - 1
    squares = P2.reduce_rowwise().new(name="squares")
    del P2
    squares(squares.V, replace) << binary.cdiv(squares, 2)  # Drop zeros

    # (2) Subtract 1 for each u and 1 for each w for all combos: degrees * (degrees - 1)
    #   uw_count = degrees * (degrees - 1)
    denom = (degrees - 1).new(mask=squares.S, name="denom")
    neg_denom(binary.plus) << degrees * denom

    # (3) The main contribution to the denominator: degrees[u] + degrees[w] for each u-w combo.
    #   uw_degrees = plus_times(A @ degrees) * (degrees - 1)
    # denom(binary.times) << plus_times(A @ degrees)
    denom(binary.times, denom.S) << plus_times(Asubset @ degrees)

    # (4) Subtract the number of squares
    denom(binary.minus) << binary.plus(neg_denom & squares)

    # And we're done!  This result does not include 0s
    return (squares / denom).new(name="square_clustering")


def _split(L, k):
    """Split a list into approximately-equal parts"""
    N = len(L)
    start = 0
    for i in range(1, k):
        stop = (N * i + k - 1) // k
        if stop != start:
            yield L[start:stop]
            start = stop
    if stop != N:
        yield L[stop:]


def _square_clustering_split(G, node_ids=None, *, nsplits):
    if node_ids is None:
        node_ids = G._A.reduce_rowwise(monoid.any).to_values()[0]
    result = None
    for chunk_ids in _split(node_ids, nsplits):
        res = square_clustering_core(G, chunk_ids)
        if result is None:
            result = res
        else:
            result << monoid.any(result | res)
    return result


def square_clustering(G, nodes=None, *, nsplits=None):
    G = to_undirected_graph(G)
    if len(G) == 0:
        return {}
    elif nodes is None:
        # Should we use this one for subsets of nodes as well?
        if nsplits is None:
            result = square_clustering_core(G)
        else:
            result = _square_clustering_split(G, nsplits=nsplits)
        return G.vector_to_dict(result, fillvalue=0)
    elif nodes in G:
        idx = G._key_to_id[nodes]
        return single_square_clustering_core(G, idx)
    else:
        ids = G.list_to_ids(nodes)
        if nsplits is None:
            result = square_clustering_core(G, ids)
        else:
            result = _square_clustering_split(G, ids, nsplits=nsplits)
        return G.vector_to_dict(result)


def generalized_degree_core(G, *, mask=None):
    # Not benchmarked or optimized
    A = G.get_property("offdiag")
    Tri = Matrix(int, A.nrows, A.ncols, name="Tri")
    if mask is not None:
        if mask.structure and not mask.value:
            v_mask = mask.parent
        else:
            v_mask = mask.new()  # Not covered
        Tri << binary.pair(v_mask & A)  # Mask out rows
        Tri(Tri.S) << 0
    else:
        Tri(A.S) << 0
    Tri(Tri.S, binary.second) << plus_pair(Tri @ A.T)
    rows, cols, vals = Tri.to_values()
    # The column index indicates the number of triangles an edge participates in.
    # The largest this can be is `A.ncols - 1`.  Values is count of edges.
    return Matrix.from_values(
        rows,
        vals,
        np.ones(vals.size, dtype=int),
        dup_op=binary.plus,
        nrows=A.nrows,
        ncols=A.ncols - 1,
        name="generalized_degree",
    )


def single_generalized_degree_core(G, node):
    # Not benchmarked or optimized
    index = G._key_to_id[node]
    v = Vector(bool, len(G))
    v[index] = True
    return generalized_degree_core(G, mask=v.S)[index, :].new(name=f"generalized_degree_{index}")


@not_implemented_for("directed")
def generalized_degree(G, nodes=None):
    G = to_undirected_graph(G)
    if len(G) == 0:
        return {}
    if nodes in G:
        result = single_generalized_degree_core(G, nodes)
        return G.vector_to_dict(result)
    mask = G.list_to_mask(nodes)
    result = generalized_degree_core(G, mask=mask)
    return G.matrix_to_dicts(result)


__all__ = get_all(__name__)
