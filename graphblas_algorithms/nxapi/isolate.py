from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph

__all__ = ["is_isolate", "isolates", "number_of_isolates"]


def is_isolate(G, n):
    G = to_graph(G)
    return algorithms.is_isolate(G, n)


def isolates(G):
    G = to_graph(G)
    result = algorithms.isolates(G)
    return G.vector_to_nodeset(result)  # Return type is iterable


def number_of_isolates(G):
    G = to_graph(G)
    return algorithms.number_of_isolates(G)
