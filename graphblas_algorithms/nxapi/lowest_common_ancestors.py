from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_directed_graph
from graphblas_algorithms.utils import not_implemented_for

__all__ = ["lowest_common_ancestor"]


@not_implemented_for("undirected")
def lowest_common_ancestor(G, node1, node2, default=None):
    G = to_directed_graph(G)
    return algorithms.lowest_common_ancestor(G, node1, node2, default=default)
