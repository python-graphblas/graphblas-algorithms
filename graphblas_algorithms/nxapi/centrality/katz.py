from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph
from graphblas_algorithms.utils import not_implemented_for

from ..exception import NetworkXError, PowerIterationFailedConvergence

__all__ = ["katz_centrality"]


@not_implemented_for("multigraph")
def katz_centrality(
    G,
    alpha=0.1,
    beta=1.0,
    max_iter=1000,
    tol=1.0e-6,
    nstart=None,
    normalized=True,
    weight=None,
):
    G = to_graph(G, weight=weight, dtype=float)
    if len(G) == 0:
        return {}
    x = G.dict_to_vector(nstart, dtype=float, name="nstart")
    try:
        b = float(beta)
    except (TypeError, ValueError):
        try:
            b = G.dict_to_vector(beta, dtype=float, name="beta")
        except (TypeError, ValueError, AttributeError) as e:
            raise NetworkXError(*e.args) from e
    try:
        result = algorithms.katz_centrality(
            G, alpha=alpha, beta=b, max_iter=max_iter, tol=tol, nstart=x, normalized=normalized
        )
    except algorithms.exceptions.ConvergenceFailure as e:
        raise PowerIterationFailedConvergence(*e.args) from e
    except algorithms.exceptions.GraphBlasAlgorithmException as e:
        raise NetworkXError(*e.args) from e
    return G.vector_to_nodemap(result)
