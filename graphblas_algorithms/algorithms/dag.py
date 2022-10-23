from graphblas import Vector, replace
from graphblas.semiring import lor_pair

__all__ = ["descendants", "ancestors"]


# Push-pull optimization is possible, but annoying to implement
def descendants(G, source):
    if source not in G._key_to_id:
        raise KeyError(f"The node {source} is not in the graph")
    index = G._key_to_id[source]
    A = G._A
    q = Vector.from_values(index, True, size=A.nrows, name="q")
    rv = q.dup(name="descendants")
    for _ in range(A.nrows):
        q(~rv.S, replace) << lor_pair(q @ A)
        if q.nvals == 0:
            break
        rv(q.S) << True
    del rv[index]
    return rv


def ancestors(G, source):
    if source not in G._key_to_id:
        raise KeyError(f"The node {source} is not in the graph")
    index = G._key_to_id[source]
    A = G._A
    q = Vector.from_values(index, True, size=A.nrows, name="q")
    rv = q.dup(name="descendants")
    for _ in range(A.nrows):
        q(~rv.S, replace) << lor_pair(A @ q)
        if q.nvals == 0:
            break
        rv(q.S) << True
    del rv[index]
    return rv
