from graphblas import Vector, binary, monoid, replace
from graphblas.semiring import min_plus

from ..exceptions import Unbounded

__all__ = ["single_source_bellman_ford_path_length"]


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
    for _i in range(n - 1):
        # This is a slightly modified Bellman-Ford algorithm.
        # `cur` is the current frontier of values that improved in the previous iteration.
        # This means that in this iteration we drop values from `cur` that are not better.
        cur << min_plus(cur @ A)

        # Mask is True where cur not in d or cur < d
        mask(cur.S, replace) << True  # or: `mask << unary.one[bool](cur)`
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
