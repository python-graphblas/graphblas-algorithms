from graphblas import Vector, replace
from graphblas.semiring import any_pair

__all__ = ["has_path"]


def has_path(G, source, target):
    # Perform BFS from source to target
    # TODO: can delta-stepping help? What about traversing target to source?
    src = G._key_to_id[source]
    dst = G._key_to_id[target]
    if src == dst:
        return True
    A = G._A
    q = Vector.from_values(src, True, size=A.nrows, name="q")
    v = q.dup(name="v")
    for _ in range(A.nrows - 1):
        q(~v.S, replace) << any_pair(q @ A)
        if q.nvals == 0:
            return False
        if q[dst]:
            return True
        v(q.S) << True
    return False
