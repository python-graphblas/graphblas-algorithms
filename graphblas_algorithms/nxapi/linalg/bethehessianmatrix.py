from graphblas_algorithms import linalg
from graphblas_algorithms.classes.graph import to_undirected_graph
from graphblas_algorithms.utils import not_implemented_for

__all__ = ["bethe_hessian_matrix"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def bethe_hessian_matrix(G, r=None, nodelist=None):
    G = to_undirected_graph(G)
    return linalg.bethe_hessian_matrix(G, r=r, nodelist=nodelist)
