from graphblas import binary
from networkx import NetworkXError
from networkx.utils import not_implemented_for

from graphblas_algorithms.classes.digraph import to_directed_graph


def reciprocity_core(G, mask=None):
    overlap, total_degrees = G.get_properties("recip_degrees+ total_degrees+", mask=mask)
    return binary.truediv(2 * overlap | total_degrees, left_default=0, right_default=0).new(
        name="reciprocity"
    )


@not_implemented_for("undirected", "multigraph")
def reciprocity(G, nodes=None):
    if nodes is None:
        return overall_reciprocity(G)
    G = to_directed_graph(G, dtype=bool)
    if nodes in G:
        mask = G.list_to_mask([nodes])
        result = reciprocity_core(G, mask=mask)
        rv = result[G._key_to_id[nodes]].value
        if rv is None:
            raise NetworkXError("Not defined for isolated nodes.")
        else:
            return rv
    else:
        mask = G.list_to_mask(nodes)
        result = reciprocity_core(G, mask=mask)
        return G.vector_to_dict(result, mask=mask)


def overall_reciprocity_core(G):
    n_all_edge = G._A.nvals
    if n_all_edge == 0:
        raise NetworkXError("Not defined for empty graphs")
    n_overlap_edges, has_self_edges = G.get_properties("total_recip+ has_self_edges")
    if has_self_edges:
        n_overlap_edges -= G.get_property("diag").nvals
    return n_overlap_edges / n_all_edge


@not_implemented_for("undirected", "multigraph")
def overall_reciprocity(G):
    G = to_directed_graph(G, dtype=bool)
    return overall_reciprocity_core(G)
