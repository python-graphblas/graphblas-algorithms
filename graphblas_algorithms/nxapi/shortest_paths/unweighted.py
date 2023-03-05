from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph

from .._utils import normalize_chunksize, partition
from ..exception import NodeNotFound

__all__ = [
    "single_source_shortest_path_length",
    "single_target_shortest_path_length",
    "all_pairs_shortest_path_length",
]


def single_source_shortest_path_length(G, source, cutoff=None):
    G = to_graph(G)
    if source not in G:
        raise NodeNotFound(f"Source {source} is not in G")
    v = algorithms.single_source_shortest_path_length(G, source, cutoff)
    return G.vector_to_nodemap(v)


def single_target_shortest_path_length(G, target, cutoff=None):
    G = to_graph(G)
    if target not in G:
        raise NodeNotFound(f"Target {target} is not in G")
    v = algorithms.single_target_shortest_path_length(G, target, cutoff)
    return G.vector_to_nodemap(v)


def all_pairs_shortest_path_length(G, cutoff=None, *, chunksize="10 MiB"):
    G = to_graph(G)
    chunksize = normalize_chunksize(chunksize, len(G) * G._A.dtype.np_type.itemsize, len(G))
    if chunksize is None:
        D = algorithms.all_pairs_shortest_path_length(G, cutoff)
        yield from G.matrix_to_nodenodemap(D).items()
    elif chunksize < 2:
        for source in G:
            d = algorithms.single_source_shortest_path_length(G, source, cutoff)
            yield (source, G.vector_to_nodemap(d))
    else:
        for cur_nodes in partition(chunksize, list(G)):
            D = algorithms.all_pairs_shortest_path_length(G, cutoff, nodes=cur_nodes)
            for i, source in enumerate(cur_nodes):
                d = D[i, :].new(name=f"all_pairs_shortest_path_length_{i}")
                yield (source, G.vector_to_nodemap(d))
