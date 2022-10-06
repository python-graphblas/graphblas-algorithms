from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph

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


def cut_size(G, S, T=None, weight=None):
    is_multigraph = G.is_multigraph()
    G = to_graph(G, weight=weight)
    S = G.set_to_vector(S, ignore_extra=True)
    T = G.set_to_vector(T, ignore_extra=True)
    return algorithms.cut_size(G, S, T, is_weighted=is_multigraph or weight is not None)


def volume(G, S, weight=None):
    is_multigraph = G.is_multigraph()
    G = to_graph(G, weight=weight)
    S = G.list_to_vector(S)
    return algorithms.volume(G, S, weighted=is_multigraph or weight is not None)


def normalized_cut_size(G, S, T=None, weight=None):
    G = to_graph(G, weight=weight)
    S = G.set_to_vector(S, ignore_extra=True)
    if T is None:
        T = (~S.S).new()
    else:
        T = G.set_to_vector(T, ignore_extra=True)
    return algorithms.normalized_cut_size(G, S, T)


def conductance(G, S, T=None, weight=None):
    G = to_graph(G, weight=weight)
    S = G.set_to_vector(S, ignore_extra=True)
    if T is None:
        T = (~S.S).new()
    else:
        T = G.set_to_vector(T, ignore_extra=True)
    return algorithms.conductance(G, S, T)


def edge_expansion(G, S, T=None, weight=None):
    G = to_graph(G, weight=weight)
    S = G.set_to_vector(S, ignore_extra=True)
    T = G.set_to_vector(T, ignore_extra=True)
    return algorithms.edge_expansion(G, S, T)


def mixing_expansion(G, S, T=None, weight=None):
    G = to_graph(G, weight=weight)
    S = G.set_to_vector(S, ignore_extra=True)
    T = G.set_to_vector(T, ignore_extra=True)
    return algorithms.mixing_expansion(G, S, T)


def node_expansion(G, S):
    G = to_graph(G)
    S = G.list_to_vector(S)
    return algorithms.node_expansion(G, S)


def boundary_expansion(G, S):
    G = to_graph(G)
    S = G.set_to_vector(S, ignore_extra=True)
    return algorithms.boundary_expansion(G, S)
