"""BFS routines used by other algorithms"""

import numpy as np
from graphblas import Matrix, Vector, binary, indexunary, replace, semiring, unary
from graphblas.semiring import any_pair


def _get_cutoff(n, cutoff):
    if cutoff is None or cutoff >= n:
        return n  # Everything
    return cutoff + 1  # Inclusive


def _bfs_plain(G, source=None, target=None, *, index=None, cutoff=None):
    if source is not None:
        index = G._key_to_id[source]
    if target is not None:
        dst_id = G._key_to_id[target]
    else:
        dst_id = None
    A = G.get_property("offdiag")
    n = A.nrows
    v = Vector(bool, n, name="bfs_plain")
    q = Vector(bool, n, name="q")
    v[index] = True
    q[index] = True
    any_pair_bool = any_pair[bool]
    cutoff = _get_cutoff(n, cutoff)
    for _i in range(1, cutoff):
        q(~v.S, replace) << any_pair_bool(q @ A)
        if q.nvals == 0:
            break
        if dst_id is not None and dst_id in q:
            break
        v(q.S) << True
    return v


def _bfs_level(G, source, cutoff=None, *, transpose=False, dtype=int):
    if dtype == bool:
        dtype = int
    index = G._key_to_id[source]
    A = G.get_property("offdiag")
    if transpose and G.is_directed():
        A = A.T  # TODO: should we use "AT" instead?
    n = A.nrows
    v = Vector(dtype, n, name="bfs_level")
    q = Vector(bool, n, name="q")
    v[index] = 0
    q[index] = True
    any_pair_bool = any_pair[bool]
    cutoff = _get_cutoff(n, cutoff)
    for i in range(1, cutoff):
        q(~v.S, replace) << any_pair_bool(q @ A)
        if q.nvals == 0:
            break
        v(q.S) << i
    return v


def _bfs_levels(G, nodes, cutoff=None, *, dtype=int):
    if dtype == bool:
        dtype = int
    A = G.get_property("offdiag")
    n = A.nrows
    if nodes is None:
        # TODO: `D = Vector.from_scalar(0, n, dtype).diag()`
        D = Vector(dtype, n, name="bfs_levels_vector")
        D << 0
        D = D.diag(name="bfs_levels")
    else:
        ids = G.list_to_ids(nodes)
        D = Matrix.from_coo(
            np.arange(len(ids), dtype=np.uint64),
            ids,
            0,
            dtype,
            nrows=len(ids),
            ncols=n,
            name="bfs_levels",
        )
    Q = unary.one[bool](D).new(name="Q")
    any_pair_bool = any_pair[bool]
    cutoff = _get_cutoff(n, cutoff)
    for i in range(1, cutoff):
        Q(~D.S, replace) << any_pair_bool(Q @ A)
        if Q.nvals == 0:
            break
        D(Q.S) << i
    return D


def _bfs_parent(G, source, cutoff=None, *, target=None, transpose=False, dtype=int):
    if dtype == bool:
        dtype = int
    index = G._key_to_id[source]
    if target is not None:
        dst_id = G._key_to_id[target]
    else:
        dst_id = None
    A = G.get_property("offdiag")
    if transpose and G.is_directed():
        A = A.T  # TODO: should we use "AT" instead?
    n = A.nrows
    v = Vector(dtype, n, name="bfs_parent")
    q = Vector(dtype, n, name="q")
    v[index] = index
    q[index] = index
    min_first = semiring.min_first[v.dtype]
    index = indexunary.index[v.dtype]
    cutoff = _get_cutoff(n, cutoff)
    for _i in range(1, cutoff):
        q(~v.S, replace) << min_first(q @ A)
        if q.nvals == 0:
            break
        v(q.S) << q
        if dst_id is not None and dst_id in q:
            break
        q << index(q)
    return v


# TODO: benchmark this and the version commented out below
def _bfs_plain_bidirectional(G, source):
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
def _bfs_plain_bidirectional(G, source):
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
