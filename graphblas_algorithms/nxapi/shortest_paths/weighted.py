from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph

from ..exception import NetworkXUnbounded, NodeNotFound

__all__ = [
    "all_pairs_bellman_ford_path_length",
    "single_source_bellman_ford_path_length",
]


def all_pairs_bellman_ford_path_length(G, weight="weight", *, chunksize=128):
    # Larger chunksize offers more parallelism, but uses more memory.
    G = to_graph(G, weight=weight)
    if chunksize < 2:
        for source in G:
            try:
                d = algorithms.single_source_bellman_ford_path_length(G, source)
            except algorithms.exceptions.Unbounded as e:
                raise NetworkXUnbounded(*e.args) from e
            yield (source, G.vector_to_nodemap(d))
    else:
        # We should probably make a utility function for chunking
        nodes = list(G)
        for start, stop in zip(
            range(0, len(nodes), chunksize), range(chunksize, len(nodes) + chunksize, chunksize)
        ):
            cur_nodes = nodes[start:stop]
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
