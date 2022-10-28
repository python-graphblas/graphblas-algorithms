from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_directed_graph, to_graph
from graphblas_algorithms.utils import not_implemented_for

__all__ = ["degree_centrality", "in_degree_centrality", "out_degree_centrality"]


def degree_centrality(G):
    G = to_graph(G)
    result = algorithms.degree_centrality(G)
    return G.vector_to_nodemap(result, fillvalue=0.0)


@not_implemented_for("undirected")
def in_degree_centrality(G):
    G = to_directed_graph(G)
    result = algorithms.in_degree_centrality(G)
    return G.vector_to_nodemap(result, fillvalue=0.0)


@not_implemented_for("undirected")
def out_degree_centrality(G):
    G = to_directed_graph(G)
    result = algorithms.out_degree_centrality(G)
    return G.vector_to_nodemap(result, fillvalue=0.0)
