from graphblas import monoid
from graphblas.semiring import any_pair, plus_first

from .boundary import edge_boundary, node_boundary

__all__ = [
    "cut_size",
    "volume",
    "normalized_cut_size",
    "conductance",
    "edge_expansion",
    "mixing_expansion",
    "node_expansion",
    "boundary_expansion",
]


def cut_size(G, S, T=None, *, is_weighted=False):
    edges = edge_boundary(G, S, T, is_weighted=is_weighted)
    if is_weighted:
        rv = edges.reduce_scalar(monoid.plus).get(0)
    else:
        rv = edges.nvals
    if G.is_directed():
        edges = edge_boundary(G, T, S, is_weighted=is_weighted)
        if is_weighted:
            rv += edges.reduce_scalar(monoid.plus).get(0)
        else:
            rv += edges.nvals
    return rv


def volume(G, S, *, weighted=False):
    if weighted:
        degrees = plus_first(G._A @ S)
    else:
        degrees = G.get_property("row_degrees+", mask=S.S)
    return degrees.reduce(monoid.plus).get(0)


def normalized_cut_size(G, S, T=None):
    num_cut_edges = cut_size(G, S, T)
    volume_S = volume(G, S)
    volume_T = volume(G, T)
    return num_cut_edges * ((1 / volume_S) + (1 / volume_T))


def conductance(G, S, T=None):
    num_cut_edges = cut_size(G, S, T)
    volume_S = volume(G, S)
    volume_T = volume(G, T)
    return num_cut_edges / min(volume_S, volume_T)


def edge_expansion(G, S, T=None):
    num_cut_edges = cut_size(G, S, T)
    if T is None:
        Tnvals = S.size - S.nvals
    else:
        Tnvals = T.nvals
    return num_cut_edges / min(S.nvals, Tnvals)


def mixing_expansion(G, S, T=None):
    num_cut_edges = cut_size(G, S, T)
    return num_cut_edges / G._A.nvals  # Why no factor of 2 in denominator?


def node_expansion(G, S):
    neighborhood = any_pair(S @ G._A)
    return neighborhood.nvals / S.nvals


def boundary_expansion(G, S):
    result = node_boundary(G, S)
    return result.nvals / S.nvals
