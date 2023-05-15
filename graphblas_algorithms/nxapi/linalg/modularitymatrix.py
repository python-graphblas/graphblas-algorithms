from graphblas_algorithms import linalg
from graphblas_algorithms.classes.digraph import to_directed_graph
from graphblas_algorithms.classes.graph import to_undirected_graph
from graphblas_algorithms.utils import not_implemented_for

__all__ = ["modularity_matrix", "directed_modularity_matrix"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def modularity_matrix(G, nodelist=None, weight=None):
    G = to_undirected_graph(G, weight=weight)
    return linalg.modularity_matrix(G, nodelist, is_weighted=weight is not None)


@not_implemented_for("undirected")
@not_implemented_for("multigraph")
def directed_modularity_matrix(G, nodelist=None, weight=None):
    G = to_directed_graph(G, weight=weight)
    return linalg.directed_modularity_matrix(G, nodelist, is_weighted=weight is not None)
