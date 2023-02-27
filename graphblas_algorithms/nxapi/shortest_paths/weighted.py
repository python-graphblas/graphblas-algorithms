from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph

from .._utils import normalize_chunksize, partition
from ..exception import NetworkXUnbounded, NodeNotFound

__all__ = [
    "all_pairs_bellman_ford_path_length",
    "single_source_bellman_ford_path_length",
]


def all_pairs_bellman_ford_path_length(G, weight="weight", *, chunksize="10 MiB"):
    # Larger chunksize offers more parallelism, but uses more memory.
    # Chunksize indicates for how many source nodes to compute at one time.
    # The default is to choose the number of rows so the result, if dense,
    # will be about 10MB.
    G = to_graph(G, weight=weight)
    chunksize = normalize_chunksize(chunksize, len(G) * G._A.dtype.np_type.itemsize, len(G))
    if chunksize is None:
        # All at once
        try:
            D = algorithms.bellman_ford_path_lengths(G)
        except algorithms.exceptions.Unbounded as e:
            raise NetworkXUnbounded(*e.args) from e
        yield from G.matrix_to_nodenodemap(D).items()
    elif chunksize < 2:
        for source in G:
            try:
                d = algorithms.single_source_bellman_ford_path_length(G, source)
            except algorithms.exceptions.Unbounded as e:
                raise NetworkXUnbounded(*e.args) from e
            yield (source, G.vector_to_nodemap(d))
    else:
        for cur_nodes in partition(chunksize, list(G)):
            try:
                D = algorithms.bellman_ford_path_lengths(G, cur_nodes)
            except algorithms.exceptions.Unbounded as e:
                raise NetworkXUnbounded(*e.args) from e
            for i, source in enumerate(cur_nodes):
                d = D[i, :].new(name=f"all_pairs_bellman_ford_path_length_{i}")
                yield (source, G.vector_to_nodemap(d))


def single_source_bellman_ford_path_length(G, source, weight="weight"):
    # TODO: what if weight is a function?
    G = to_graph(G, weight=weight)
    try:
        d = algorithms.single_source_bellman_ford_path_length(G, source)
    except algorithms.exceptions.Unbounded as e:
        raise NetworkXUnbounded(*e.args) from e
    except KeyError as e:
        raise NodeNotFound(*e.args) from e
    return G.vector_to_nodemap(d)
