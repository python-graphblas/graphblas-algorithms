from graphblas import monoid

from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph
from graphblas_algorithms.classes.graph import to_undirected_graph
from graphblas_algorithms.utils import not_implemented_for

from ._utils import normalize_chunksize, partition

__all__ = [
    "triangles",
    "transitivity",
    "clustering",
    "average_clustering",
    "square_clustering",
    "generalized_degree",
]


@not_implemented_for("directed")
def triangles(G, nodes=None):
    G = to_undirected_graph(G, dtype=bool)
    if len(G) == 0:
        return {}
    if nodes in G:
        return algorithms.single_triangle(G, nodes)
    mask = G.list_to_mask(nodes)
    result = algorithms.triangles(G, mask=mask)
    return G.vector_to_nodemap(result, mask=mask, fill_value=0)


def transitivity(G):
    G = to_graph(G, dtype=bool)  # directed or undirected
    if len(G) == 0:
        return 0
    if G.is_directed():
        func = algorithms.transitivity_directed
    else:
        func = algorithms.transitivity
    return G._cacheit("transitivity", func, G)


def clustering(G, nodes=None, weight=None):
    G = to_graph(G, weight=weight)  # to directed or undirected
    if len(G) == 0:
        return {}
    weighted = weight is not None
    if nodes in G:
        if G.is_directed():
            return algorithms.single_clustering_directed(G, nodes, weighted=weighted)
        return algorithms.single_clustering(G, nodes, weighted=weighted)
    mask = G.list_to_mask(nodes)
    if G.is_directed():
        result = algorithms.clustering_directed(G, weighted=weighted, mask=mask)
    else:
        result = algorithms.clustering(G, weighted=weighted, mask=mask)
    return G.vector_to_nodemap(result, mask=mask, fill_value=0.0)


def average_clustering(G, nodes=None, weight=None, count_zeros=True):
    G = to_graph(G, weight=weight)  # to directed or undirected
    if len(G) == 0:
        raise ZeroDivisionError
    weighted = weight is not None
    mask = G.list_to_mask(nodes)
    if G.is_directed():
        func = algorithms.average_clustering_directed
    else:
        func = algorithms.average_clustering
    if mask is None:
        return G._cacheit(
            f"average_clustering(count_zeros={count_zeros})",
            func,
            G,
            weighted=weighted,
            count_zeros=count_zeros,
        )
    return func(G, weighted=weighted, count_zeros=count_zeros, mask=mask)


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


# TODO: should this move into algorithms?
def _square_clustering_split(G, node_ids=None, *, chunksize):
    if node_ids is None:
        node_ids, _ = G._A.reduce_rowwise(monoid.any).to_coo(values=False)
    result = None
    for chunk_ids in partition(chunksize, node_ids):
        res = algorithms.square_clustering(G, chunk_ids)
        if result is None:
            result = res
        else:
            result << monoid.any(result | res)
    return result


def square_clustering(G, nodes=None, *, chunksize="256 MiB"):
    # `chunksize` is used to split the computation into chunks.
    # square_clustering computes `A @ A`, which can get very large, even dense.
    # The default `chunksize` is to choose the number so that `Asubset @ A`
    # will be about 256 MB if dense.
    G = to_undirected_graph(G)
    if len(G) == 0:
        return {}

    chunksize = normalize_chunksize(chunksize, len(G) * G._A.dtype.np_type.itemsize, len(G))

    if nodes is None:
        # Should we use this one for subsets of nodes as well?
        if chunksize is None:
            result = algorithms.square_clustering(G)
        else:
            result = _square_clustering_split(G, chunksize=chunksize)
        return G.vector_to_nodemap(result, fill_value=0)
    if nodes in G:
        idx = G._key_to_id[nodes]
        return algorithms.single_square_clustering(G, idx)
    ids = G.list_to_ids(nodes)
    if chunksize is None:
        result = algorithms.square_clustering(G, ids)
    else:
        result = _square_clustering_split(G, ids, chunksize=chunksize)
    return G.vector_to_nodemap(result)


@not_implemented_for("directed")
def generalized_degree(G, nodes=None):
    G = to_undirected_graph(G)
    if len(G) == 0:
        return {}
    if nodes in G:
        result = algorithms.single_generalized_degree(G, nodes)
        return G.vector_to_nodemap(result)
    mask = G.list_to_mask(nodes)
    result = algorithms.generalized_degree(G, mask=mask)
    return G.matrix_to_vectornodemap(result)
