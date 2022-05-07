from graphblas import binary
from networkx import NetworkXError
from networkx.utils import not_implemented_for

from ._utils import graph_to_adjacency, list_to_mask, vector_to_dict


def reciprocity_core(G, mask=None):
    # TODO: used cached properties
    overlap = binary.pair(G & G.T).reduce_rowwise().new(mask=mask)
    total_degrees = G.reduce_rowwise("count").new(mask=mask) + G.reduce_columnwise("count").new(
        mask=mask
    )
    return binary.truediv(2 * overlap | total_degrees, left_default=0, right_default=0).new(
        name="reciprocity"
    )


@not_implemented_for("undirected", "multigraph")
def reciprocity(G, nodes=None):
    if nodes is None:
        return overall_reciprocity(G)
    A, key_to_id = graph_to_adjacency(G, dtype=bool)
    if nodes in G:
        mask, id_to_key = list_to_mask([nodes], key_to_id)
        result = reciprocity_core(A, mask=mask)
        rv = result[key_to_id[nodes]].value
        if rv is None:
            raise NetworkXError("Not defined for isolated nodes.")
        else:
            return rv
    else:
        mask, id_to_key = list_to_mask(nodes, key_to_id)
        result = reciprocity_core(A, mask=mask)
        return vector_to_dict(result, key_to_id, id_to_key, mask=mask)


def overall_reciprocity_core(G, *, has_self_edges=True):
    n_all_edge = G.nvals
    if n_all_edge == 0:
        raise NetworkXError("Not defined for empty graphs")
    n_overlap_edges = binary.pair(G & G.T).reduce_scalar(allow_empty=False).value
    if has_self_edges:
        n_overlap_edges -= G.diag().nvals
    return n_overlap_edges / n_all_edge


@not_implemented_for("undirected", "multigraph")
def overall_reciprocity(G):
    A, key_to_id = graph_to_adjacency(G, dtype=bool)
    return overall_reciprocity_core(A)
