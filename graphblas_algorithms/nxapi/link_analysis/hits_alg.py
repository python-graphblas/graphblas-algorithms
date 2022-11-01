from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph

from ..exception import ArpackNoConvergence

__all__ = ["hits"]


def hits(G, max_iter=100, tol=1.0e-8, nstart=None, normalized=True):
    G = to_graph(G, weight="weight", dtype=float)
    if len(G) == 0:
        return {}, {}
    x = G.dict_to_vector(nstart, dtype=float, name="nstart")
    try:
        h, a = algorithms.hits(G, max_iter=max_iter, tol=tol, nstart=x, normalized=normalized)
    except algorithms.exceptions.ConvergenceFailure as e:
        if max_iter < 1:
            raise ValueError(*e.args) from e
        else:
            raise ArpackNoConvergence(*e.args, (), ()) from e
        # TODO: it would be nice if networkx raised their own exception, such as:
        # raise nx.PowerIterationFailedConvergence(*e.args) from e
    return G.vector_to_nodemap(h, fillvalue=0), G.vector_to_nodemap(a, fillvalue=0)
