import numpy as np
from graphblas import Matrix, Vector, binary, monoid, replace, unary
from graphblas.semiring import min_plus

from ..exceptions import Unbounded

__all__ = [
    "single_source_bellman_ford_path_length",
    "bellman_ford_path_lengths",
]


def single_source_bellman_ford_path_length(G, source):
    # No need for `is_weighted=` keyword, b/c this is assumed to be weighted (I think)
    index = G._key_to_id[source]
    A = G._A
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
    return d


def bellman_ford_path_lengths(G, nodes=None, *, expand_output=False):
    """

    Parameters
    ----------
    expand_output : bool, default False
        When False, the returned Matrix has one row per node in nodes.
        When True, the returned Matrix has the same shape as the input Matrix.
    """
    # Same algorithms as in `single_source_bellman_ford_path_length`, but with
    # `Cur` as a Matrix with each row corresponding to a source node.
    A = G._A
    if A.dtype == bool:
        dtype = int
    else:
        dtype = A.dtype
    n = A.nrows
    if nodes is None:
        # TODO: `D = Vector.from_iso_value(0, n, dtype).diag()`
        D = Vector(dtype, n, name="bellman_ford_path_lengths_vector")
        D << 0
        D = D.diag(name="bellman_ford_path_lengths")
    else:
        ids = G.list_to_ids(nodes)
        D = Matrix.from_coo(
            np.arange(len(ids), dtype=np.uint64), ids, 0, dtype, nrows=len(ids), ncols=n
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
    if nodes is not None and expand_output:
        rv = Matrix(D.dtype, n, n, name=D.name)
        rv[ids, :] = D
        return rv
    return D
