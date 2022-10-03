from graphblas import Vector, replace
from graphblas.semiring import any_pair
from networkx import NetworkXError

from graphblas_algorithms.classes.digraph import to_graph
from graphblas_algorithms.utils import get_all


# Push-pull optimization is possible, but annoying to implement
def descendants_core(G, source):
    if source not in G._key_to_id:
        raise NetworkXError(f"The node {source} is not in the graph")
    index = G._key_to_id[source]
    A = G._A
    q = Vector.from_values(index, True, size=A.nrows, name="q")
    rv = q.dup(name="descendants")
    for _ in range(A.nrows):
        q(~rv.S, replace) << any_pair(A.T @ q)
        if q.nvals == 0:
            break
        rv(q.S) << True
    del rv[index]
    return rv


def descendants(G, source):
    G = to_graph(G)
    result = descendants_core(G, source)
    return G.vector_to_set(result)


def ancestors_core(G, source):
    if source not in G._key_to_id:
        raise NetworkXError(f"The node {source} is not in the graph")
    index = G._key_to_id[source]
    A = G._A
    q = Vector.from_values(index, True, size=A.nrows, name="q")
    rv = q.dup(name="descendants")
    for _ in range(A.nrows):
        q(~rv.S, replace) << any_pair(A @ q)
        if q.nvals == 0:
            break
        rv(q.S) << True
    del rv[index]
    return rv


def ancestors(G, source):
    G = to_graph(G)
    result = ancestors_core(G, source)
    return G.vector_to_set(result)


__all__ = get_all(__name__)
