from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph

from ..exception import NetworkXUnbounded, NodeNotFound

__all__ = [
    "all_pairs_bellman_ford_path_length",
    "single_source_bellman_ford_path_length",
]


def all_pairs_bellman_ford_path_length(G, weight="weight"):
    # TODO: what if weight is a function?
    # How should we implement and call `algorithms.all_pairs_bellman_ford_path_length`?
    # Should we compute in chunks to expose more parallelism?
    G = to_graph(G, weight=weight)
    for source in G:
        try:
            d = algorithms.single_source_bellman_ford_path_length(G, source)
        except algorithms.exceptions.Unbounded as e:
            raise NetworkXUnbounded(*e.args) from e
        except KeyError as e:
            raise NodeNotFound(*e.args) from e
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
