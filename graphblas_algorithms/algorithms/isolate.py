from graphblas_algorithms.classes.digraph import to_graph
from graphblas_algorithms.utils import get_all


def is_isolate_core(G, n):
    index = G._key_to_id[n]
    if G.is_directed():
        degrees = G.get_property("total_degrees+")
    else:
        degrees = G.get_property("degrees+")
    return index not in degrees


def is_isolate(G, n):
    G = to_graph(G)
    return is_isolate_core(G, n)


def isolates_core(G):
    if G.is_directed():
        degrees = G.get_property("total_degrees+")
    else:
        degrees = G.get_property("degrees+")
    return (~degrees.S).new(name="isolates")


def isolates(G):
    G = to_graph(G)
    result = isolates_core(G)
    indices, _ = result.to_values(values=False, sort=False)
    id_to_key = G.id_to_key
    return (id_to_key[index] for index in indices)


def number_of_isolates_core(G):
    if G.is_directed():
        degrees = G.get_property("total_degrees+")
    else:
        degrees = G.get_property("degrees+")
    return degrees.size - degrees.nvals


def number_of_isolates(G):
    G = to_graph(G)
    return number_of_isolates_core(G)


__all__ = get_all(__name__)
