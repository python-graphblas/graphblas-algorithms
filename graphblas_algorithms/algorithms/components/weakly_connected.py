from graphblas import Vector, binary, replace
from graphblas.semiring import any_pair

from graphblas_algorithms.algorithms.exceptions import PointlessConcept


def is_weakly_connected(G):
    if len(G) == 0:
        raise PointlessConcept("Connectivity is undefined for the null graph.")
    return _plain_bfs(G, next(iter(G))).nvals == len(G)


# TODO: benchmark this and the version commented out below
def _plain_bfs(G, source):
    # Bi-directional BFS w/o symmetrizing the adjacency matrix
    index = G._key_to_id[source]
    A = G.get_property("offdiag")
    # XXX: should we use `AT` if available?
    n = A.nrows
    v = Vector(bool, n, name="bfs_plain")
    q_out = Vector(bool, n, name="q_out")
    q_in = Vector(bool, n, name="q_in")
    v[index] = True
    q_in[index] = True
    any_pair_bool = any_pair[bool]
    is_out_empty = True
    is_in_empty = False
    for _i in range(1, n):
        # Traverse out-edges from the most recent `q_in` and `q_out`
        if is_out_empty:
            q_out(~v.S) << any_pair_bool(q_in @ A)
        else:
            q_out << binary.any(q_out | q_in)
            q_out(~v.S, replace) << any_pair_bool(q_out @ A)
        is_out_empty = q_out.nvals == 0
        if not is_out_empty:
            v(q_out.S) << True
        elif is_in_empty:
            break
        # Traverse in-edges from the most recent `q_in` and `q_out`
        if is_in_empty:
            q_in(~v.S) << any_pair_bool(A @ q_out)
        else:
            q_in << binary.any(q_out | q_in)
            q_in(~v.S, replace) << any_pair_bool(A @ q_in)
        is_in_empty = q_in.nvals == 0
        if not is_in_empty:
            v(q_in.S) << True
        elif is_out_empty:
            break
    return v


"""
def _plain_bfs(G, source):
    # Bi-directional BFS w/o symmetrizing the adjacency matrix
    index = G._key_to_id[source]
    A = G.get_property("offdiag")
    n = A.nrows
    v = Vector(bool, n, name="bfs_plain")
    q = Vector(bool, n, name="q")
    q2 = Vector(bool, n, name="q_2")
    v[index] = True
    q[index] = True
    any_pair_bool = any_pair[bool]
    for _i in range(1, n):
        q2(~v.S, replace) << any_pair_bool(q @ A)
        v(q2.S) << True
        q(~v.S, replace) << any_pair_bool(A @ q)
        if q.nvals == 0:
            if q2.nvals == 0:
                break
            q, q2 = q2, q
        elif q2.nvals != 0:
            q << binary.any(q | q2)
    return v
"""
