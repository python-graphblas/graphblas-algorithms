from graphblas import monoid
from graphblas.semiring import any_pair, plus_first

from graphblas_algorithms.classes.digraph import to_graph
from graphblas_algorithms.utils import get_all

from .boundary import edge_boundary_core, node_boundary_core


def cut_size_core(G, S, T=None, *, is_weighted=False):
    edges = edge_boundary_core(G, S, T, is_weighted=is_weighted)
    if is_weighted:
        rv = edges.reduce_scalar(monoid.plus).get(0)
    else:
        rv = edges.nvals
    if G.is_directed():
        edges = edge_boundary_core(G, T, S, is_weighted=is_weighted)
        if is_weighted:
            rv += edges.reduce_scalar(monoid.plus).get(0)
        else:
            rv += edges.nvals
    return rv


def cut_size(G, S, T=None, weight=None):
    is_multigraph = G.is_multigraph()
    G = to_graph(G, weight=weight)
    S = G.set_to_vector(S, ignore_extra=True)
    T = G.set_to_vector(T, ignore_extra=True)
    return cut_size_core(G, S, T, is_weighted=is_multigraph or weight is not None)


def volume_core(G, S, *, weighted=False):
    if weighted:
        degrees = plus_first(G._A @ S)
    else:
        degrees = G.get_property("row_degrees+", mask=S.S)
    return degrees.reduce(monoid.plus).get(0)


def volume(G, S, weight=None):
    is_multigraph = G.is_multigraph()
    G = to_graph(G, weight=weight)
    S = G.list_to_vector(S)
    return volume_core(G, S, weighted=is_multigraph or weight is not None)


def normalized_cut_size_core(G, S, T=None):
    num_cut_edges = cut_size_core(G, S, T)
    volume_S = volume_core(G, S)
    volume_T = volume_core(G, T)
    return num_cut_edges * ((1 / volume_S) + (1 / volume_T))


def normalized_cut_size(G, S, T=None, weight=None):
    G = to_graph(G, weight=weight)
    S = G.set_to_vector(S, ignore_extra=True)
    if T is None:
        T = (~S.S).new()
    else:
        T = G.set_to_vector(T, ignore_extra=True)
    return normalized_cut_size_core(G, S, T)


def conductance_core(G, S, T=None):
    num_cut_edges = cut_size_core(G, S, T)
    volume_S = volume_core(G, S)
    volume_T = volume_core(G, T)
    return num_cut_edges / min(volume_S, volume_T)


def conductance(G, S, T=None, weight=None):
    G = to_graph(G, weight=weight)
    S = G.set_to_vector(S, ignore_extra=True)
    if T is None:
        T = (~S.S).new()
    else:
        T = G.set_to_vector(T, ignore_extra=True)
    return conductance_core(G, S, T)


def edge_expansion_core(G, S, T=None):
    num_cut_edges = cut_size_core(G, S, T)
    if T is None:
        Tnvals = S.size - S.nvals
    else:
        Tnvals = T.nvals
    return num_cut_edges / min(S.nvals, Tnvals)


def edge_expansion(G, S, T=None, weight=None):
    G = to_graph(G, weight=weight)
    S = G.set_to_vector(S, ignore_extra=True)
    T = G.set_to_vector(T, ignore_extra=True)
    return edge_expansion_core(G, S, T)


def mixing_expansion_core(G, S, T=None):
    num_cut_edges = cut_size_core(G, S, T)
    return num_cut_edges / G._A.nvals  # Why no factor of 2 in denominator?


def mixing_expansion(G, S, T=None, weight=None):
    G = to_graph(G, weight=weight)
    S = G.set_to_vector(S, ignore_extra=True)
    T = G.set_to_vector(T, ignore_extra=True)
    return mixing_expansion_core(G, S, T)


def node_expansion_core(G, S):
    neighborhood = any_pair(G._A.T @ S)
    return neighborhood.nvals / S.nvals


def node_expansion(G, S):
    G = to_graph(G)
    S = G.list_to_vector(S)
    return node_expansion_core(G, S)


def boundary_expansion_core(G, S):
    result = node_boundary_core(G, S)
    return result.nvals / S.nvals


def boundary_expansion(G, S):
    G = to_graph(G)
    S = G.set_to_vector(S, ignore_extra=True)
    return boundary_expansion_core(G, S)


__all__ = get_all(__name__)
