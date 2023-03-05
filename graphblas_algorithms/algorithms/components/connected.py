from graphblas import Vector, replace
from graphblas.semiring import any_pair

from graphblas_algorithms.algorithms.exceptions import PointlessConcept


def is_connected(G):
    if len(G) == 0:
        raise PointlessConcept("Connectivity is undefined for the null graph.")
    return _plain_bfs(G, next(iter(G))).nvals == len(G)


def node_connected_component(G, n):
    return _plain_bfs(G, n)


def _plain_bfs(G, source):
    index = G._key_to_id[source]
    A = G.get_property("offdiag")
    n = A.nrows
    v = Vector(bool, n, name="bfs_plain")
    q = Vector(bool, n, name="q")
    v[index] = True
    q[index] = True
    any_pair_bool = any_pair[bool]
    for _i in range(1, n):
        q(~v.S, replace) << any_pair_bool(q @ A)
        if q.nvals == 0:
            break
        v(q.S) << True
    return v
