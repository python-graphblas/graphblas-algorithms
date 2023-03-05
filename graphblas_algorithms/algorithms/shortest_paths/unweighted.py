import numpy as np
from graphblas import Matrix, Vector, replace, unary
from graphblas.semiring import any_pair

__all__ = [
    "single_source_shortest_path_length",
    "single_target_shortest_path_length",
    "all_pairs_shortest_path_length",
]


def single_source_shortest_path_length(G, source, cutoff=None):
    return _bfs_level(G, source, cutoff)


def single_target_shortest_path_length(G, target, cutoff=None):
    return _bfs_level(G, target, cutoff, transpose=True)


def all_pairs_shortest_path_length(G, cutoff=None, *, nodes=None, expand_output=False):
    D = _bfs_levels(G, nodes, cutoff)
    if nodes is not None and expand_output and D.ncols != D.nrows:
        ids = G.list_to_ids(nodes)
        rv = Matrix(D.dtype, D.ncols, D.ncols, name=D.name)
        rv[ids, :] = D
        return rv
    return D


def _bfs_level(G, source, cutoff, *, transpose=False):
    index = G._key_to_id[source]
    A = G.get_property("offdiag")
    if transpose and G.is_directed():
        A = A.T  # TODO: should we use "AT" instead?
    n = A.nrows
    v = Vector(int, n, name="bfs_unweighted")
    q = Vector(bool, n, name="q")
    v[index] = 0
    q[index] = True
    any_pair_bool = any_pair[bool]
    if cutoff is None or cutoff >= n:
        cutoff = n  # Everything
    else:
        cutoff += 1  # Inclusive
    for i in range(1, cutoff):
        q(~v.S, replace) << any_pair_bool(q @ A)
        if q.nvals == 0:
            break
        v(q.S) << i
    return v


def _bfs_levels(G, nodes, cutoff):
    A = G.get_property("offdiag")
    n = A.nrows
    if nodes is None:
        # TODO: `D = Vector.from_scalar(0, n, dtype).diag()`
        D = Vector(int, n, name="bfs_unweighted_vector")
        D << 0
        D = D.diag(name="bfs_unweighted")
    else:
        ids = G.list_to_ids(nodes)
        D = Matrix.from_coo(
            np.arange(len(ids), dtype=np.uint64),
            ids,
            0,
            int,
            nrows=len(ids),
            ncols=n,
            name="bfs_unweighted",
        )
    Q = unary.one[bool](D).new(name="Q")
    any_pair_bool = any_pair[bool]
    if cutoff is None or cutoff >= n:
        cutoff = n  # Everything
    else:
        cutoff += 1  # Inclusive
    for i in range(1, cutoff):
        Q(~D.S, replace) << any_pair_bool(Q @ A)
        if Q.nvals == 0:
            break
        D(Q.S) << i
    return D
