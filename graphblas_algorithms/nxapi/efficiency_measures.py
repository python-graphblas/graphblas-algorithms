from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.graph import to_undirected_graph
from graphblas_algorithms.utils import not_implemented_for


@not_implemented_for("directed")
def efficiency(G, u, v):
    G = to_undirected_graph(G)
    return algorithms.efficiency(G, u, v)
