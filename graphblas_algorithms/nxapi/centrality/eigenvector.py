from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph
from graphblas_algorithms.utils import not_implemented_for

from ..exception import NetworkXError, NetworkXPointlessConcept, PowerIterationFailedConvergence

__all__ = ["eigenvector_centrality"]


@not_implemented_for("multigraph")
def eigenvector_centrality(G, max_iter=100, tol=1.0e-6, nstart=None, weight=None):
    G = to_graph(G, weight=weight, dtype=float)
    if len(G) == 0:
        raise NetworkXPointlessConcept("cannot compute centrality for the null graph")
    x = G.dict_to_vector(nstart, dtype=float, name="nstart")
    try:
        result = algorithms.eigenvector_centrality(G, max_iter=max_iter, tol=tol, nstart=x)
    except algorithms.exceptions.ConvergenceFailure as e:
        raise PowerIterationFailedConvergence(*e.args) from e
    except algorithms.exceptions.GraphBlasAlgorithmException as e:
        raise NetworkXError(*e.args) from e
    return G.vector_to_nodemap(result)
