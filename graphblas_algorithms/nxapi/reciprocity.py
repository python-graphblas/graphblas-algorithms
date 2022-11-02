from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_directed_graph
from graphblas_algorithms.utils import not_implemented_for

from .exception import NetworkXError

__all__ = ["reciprocity", "overall_reciprocity"]


@not_implemented_for("undirected", "multigraph")
def reciprocity(G, nodes=None):
    if nodes is None:
        return overall_reciprocity(G)
    G = to_directed_graph(G, dtype=bool)
    if nodes in G:
        mask = G.list_to_mask([nodes])
        result = algorithms.reciprocity(G, mask=mask)
        rv = result.get(G._key_to_id[nodes])
        if rv is None:
            raise NetworkXError("Not defined for isolated nodes.")
        else:
            return rv
    else:
        mask = G.list_to_mask(nodes)
        result = algorithms.reciprocity(G, mask=mask)
        return G.vector_to_nodemap(result, mask=mask)


@not_implemented_for("undirected", "multigraph")
def overall_reciprocity(G):
    G = to_directed_graph(G, dtype=bool)
    try:
        return algorithms.overall_reciprocity(G)
    except algorithms.exceptions.EmptyGraphError as e:
        raise NetworkXError("Not defined for empty graphs") from e
