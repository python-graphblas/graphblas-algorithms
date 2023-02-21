import numpy as np
from graphblas import Matrix, Vector, binary, monoid, replace, select, unary
from graphblas.semiring import any_pair, min_plus

from ..exceptions import Unbounded

__all__ = [
    "single_source_bellman_ford_path_length",
    "bellman_ford_path_lengths",
]


def single_source_bellman_ford_path_length(G, source):
    # No need for `is_weighted=` keyword, b/c this is assumed to be weighted (I think)
    index = G._key_to_id[source]
    if G.get_property("is_iso"):
        # If the edges are iso-valued (and positive), then we can simply do level BFS
        is_negative, iso_value = G.get_properties("has_negative_edges+ iso_value")
        if not is_negative:
            d = _bfs_level(G, source, dtype=iso_value.dtype)
            if iso_value != 1:
                d *= iso_value
            return d
        # It's difficult to detect negative cycles with BFS
        if G._A[index, index].get() is not None:
            raise Unbounded("Negative cycle detected.")
        if not G.is_directed() and G._A[index, :].nvals > 0:
            # For undirected graphs, any negative edge is a cycle
            raise Unbounded("Negative cycle detected.")

    # Use `offdiag` instead of `A`, b/c self-loops don't contribute to the result,
    # and negative self-loops are easy negative cycles to avoid.
    # We check if we hit a self-loop negative cycle at the end.
    A, has_negative_diagonal = G.get_properties("offdiag has_negative_diagonal")
    if A.dtype == bool:
        # Should we upcast e.g. INT8 to INT64 as well?
        dtype = int
    else:
        dtype = A.dtype
    n = A.nrows
    d = Vector(dtype, n, name="single_source_bellman_ford_path_length")
    d[index] = 0
    cur = d.dup(name="cur")
    mask = Vector(bool, n, name="mask")
    one = unary.one[bool]
    for _i in range(n - 1):
        # This is a slightly modified Bellman-Ford algorithm.
        # `cur` is the current frontier of values that improved in the previous iteration.
        # This means that in this iteration we drop values from `cur` that are not better.
        cur << min_plus(cur @ A)

        # Mask is True where cur not in d or cur < d
        mask << one(cur)
        mask(binary.second) << binary.lt(cur & d)

        # Drop values from `cur` that didn't improve
        cur(mask.V, replace) << cur
        if cur.nvals == 0:
            break
        # Update `d` with values that improved
        d(cur.S) << cur
    else:
        # Check for negative cycle when for loop completes without breaking
        cur << min_plus(cur @ A)
        mask << binary.lt(cur & d)
        if mask.reduce(monoid.lor):
            raise Unbounded("Negative cycle detected.")
    if has_negative_diagonal:
        # We removed diagonal entries above, so check if we visited one with a negative weight
        diag = G.get_property("diag")
        cur << select.valuelt(diag, 0)
        if any_pair(d @ cur):
            raise Unbounded("Negative cycle detected.")
    return d


def bellman_ford_path_lengths(G, nodes=None, *, expand_output=False):
    """Extra parameter: expand_output

    Parameters
    ----------
    expand_output : bool, default False
        When False, the returned Matrix has one row per node in nodes.
        When True, the returned Matrix has the same shape as the input Matrix.
    """
    # Same algorithms as in `single_source_bellman_ford_path_length`, but with
    # `Cur` as a Matrix with each row corresponding to a source node.
    if G.get_property("is_iso"):
        is_negative, iso_value = G.get_properties("has_negative_edges+ iso_value")
        if not is_negative:
            D = _bfs_levels(G, nodes, dtype=iso_value.dtype)
            if iso_value != 1:
                D *= iso_value
            if nodes is not None and expand_output and D.ncols != D.nrows:
                ids = G.list_to_ids(nodes)
                rv = Matrix(D.dtype, D.ncols, D.ncols, name=D.name)
                rv[ids, :] = D
                return rv
            return D
        if not G.is_directed():
            # For undirected graphs, any negative edge is a cycle
            if nodes is not None:
                ids = G.list_to_ids(nodes)
                if G._A[ids, :].nvals > 0:
                    raise Unbounded("Negative cycle detected.")
            elif G._A.nvals > 0:
                raise Unbounded("Negative cycle detected.")

    A, has_negative_diagonal = G.get_properties("offdiag has_negative_diagonal")
    if A.dtype == bool:
        dtype = int
    else:
        dtype = A.dtype
    n = A.nrows
    if nodes is None:
        # TODO: `D = Vector.from_scalar(0, n, dtype).diag()`
        D = Vector(dtype, n, name="bellman_ford_path_lengths_vector")
        D << 0
        D = D.diag(name="bellman_ford_path_lengths")
    else:
        ids = G.list_to_ids(nodes)
        D = Matrix.from_coo(
            np.arange(len(ids), dtype=np.uint64),
            ids,
            0,
            dtype,
            nrows=len(ids),
            ncols=n,
            name="bellman_ford_path_lengths",
        )
    Cur = D.dup(name="Cur")
    Mask = Matrix(bool, D.nrows, D.ncols, name="Mask")
    one = unary.one[bool]
    for _i in range(n - 1):
        Cur << min_plus(Cur @ A)
        Mask << one(Cur)
        Mask(binary.second) << binary.lt(Cur & D)
        Cur(Mask.V, replace) << Cur
        if Cur.nvals == 0:
            break
        D(Cur.S) << Cur
    else:
        Cur << min_plus(Cur @ A)
        Mask << binary.lt(Cur & D)
        if Mask.reduce_scalar(monoid.lor):
            raise Unbounded("Negative cycle detected.")
    if has_negative_diagonal:
        diag = G.get_property("diag")
        cur = select.valuelt(diag, 0)
        if any_pair(D @ cur).nvals > 0:
            raise Unbounded("Negative cycle detected.")
    if nodes is not None and expand_output and D.ncols != D.nrows:
        rv = Matrix(D.dtype, n, n, name=D.name)
        rv[ids, :] = D
        return rv
    return D


def _bfs_level(G, source, *, dtype=int):
    if dtype == bool:
        dtype = int
    index = G._key_to_id[source]
    A = G.get_property("offdiag")
    n = A.nrows
    v = Vector(dtype, n, name="bfs_level")
    q = Vector(bool, n, name="q")
    v[index] = 0
    q[index] = True
    any_pair_bool = any_pair[bool]
    for i in range(1, n):
        q(~v.S, replace) << any_pair_bool(q @ A)
        if q.nvals == 0:
            break
        v(q.S) << i
    return v


def _bfs_levels(G, nodes=None, *, dtype=int):
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
    Q = Matrix(bool, D.nrows, D.ncols, name="Q")
    Q << unary.one[bool](D)
    any_pair_bool = any_pair[bool]
    for i in range(1, n):
        Q(~D.S, replace) << any_pair_bool(Q @ A)
        if Q.nvals == 0:
            break
        D(Q.S) << i
    return D
