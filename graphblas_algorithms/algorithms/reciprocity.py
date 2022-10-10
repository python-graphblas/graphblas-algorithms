from graphblas import binary

from .exceptions import EmptyGraphError

__all__ = ["reciprocity", "overall_reciprocity"]


def reciprocity(G, mask=None):
    overlap, total_degrees = G.get_properties("recip_degrees+ total_degrees+", mask=mask)
    return binary.truediv(2 * overlap | total_degrees, left_default=0, right_default=0).new(
        name="reciprocity"
    )


def overall_reciprocity(G):
    n_all_edge = G._A.nvals
    if n_all_edge == 0:
        raise EmptyGraphError("Not defined for empty graphs")
    n_overlap_edges, has_self_edges = G.get_properties("total_recip+ has_self_edges")
    if has_self_edges:
        n_overlap_edges -= G.get_property("diag").nvals
    return n_overlap_edges / n_all_edge
