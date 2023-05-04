from graphblas import Vector, replace
from graphblas.semiring import any_pair

__all__ = ["descendants", "ancestors"]


# Push-pull optimization is possible, but annoying to implement
def descendants(G, source):
    if source not in G._key_to_id:
        raise KeyError(f"The node {source} is not in the graph")
    index = G._key_to_id[source]
    A = G.get_property("offdiag")
    q = Vector(bool, size=A.nrows, name="q")
    q[index] = True
    rv = q.dup(name="descendants")
    any_pair_bool = any_pair[bool]
    for _i in range(A.nrows):
        q(~rv.S, replace) << any_pair_bool(q @ A)
        if q.nvals == 0:
            break
        rv(q.S) << True
    del rv[index]
    return rv


def ancestors(G, source):
    if source not in G._key_to_id:
        raise KeyError(f"The node {source} is not in the graph")
    index = G._key_to_id[source]
    A = G.get_property("offdiag")
    q = Vector(bool, size=A.nrows, name="q")
    q[index] = True
    rv = q.dup(name="descendants")
    any_pair_bool = any_pair[bool]
    for _i in range(A.nrows):
        q(~rv.S, replace) << any_pair_bool(A @ q)
        if q.nvals == 0:
            break
        rv(q.S) << True
    del rv[index]
    return rv
