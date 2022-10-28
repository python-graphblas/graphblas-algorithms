from graphblas import Vector, replace
from graphblas.semiring import lor_pair

__all__ = ["has_path"]


def has_path(G, source, target):
    # Perform bidirectional BFS from source to target and target to source
    src = G._key_to_id[source]
    dst = G._key_to_id[target]
    if src == dst:
        return True
    A = G._A
    q_src = Vector.from_values(src, True, size=A.nrows, name="q_src")
    seen_src = q_src.dup(name="seen_src")
    q_dst = Vector.from_values(dst, True, size=A.nrows, name="q_dst")
    seen_dst = q_dst.dup(name="seen_dst")
    for _ in range(A.nrows // 2):
        q_src(~seen_src.S, replace) << lor_pair(q_src @ A)
        if q_src.nvals == 0:
            return False
        if lor_pair(q_src @ q_dst):
            return True

        q_dst(~seen_dst.S, replace) << lor_pair(A @ q_dst)
        if q_dst.nvals == 0:
            return False
        if lor_pair(q_src @ q_dst):
            return True

        seen_src(q_src.S) << True
        seen_dst(q_dst.S) << True
    return False
