from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph

__all__ = ["is_isolate", "isolates", "number_of_isolates"]


def is_isolate(G, n):
    G = to_graph(G)
    return algorithms.is_isolate(G, n)


def isolates(G):
    G = to_graph(G)
    result = algorithms.isolates(G)
    indices, _ = result.to_values(values=False, sort=False)
    id_to_key = G.id_to_key
    return (id_to_key[index] for index in indices)


def number_of_isolates(G):
    G = to_graph(G)
    return algorithms.number_of_isolates(G)
