from graphblas import Vector, replace
from graphblas.semiring import any_pair

__all__ = ["has_path"]


def has_path(G, source, target):
    # Perform bidirectional BFS from source to target and target to source
    src = G._key_to_id[source]
    dst = G._key_to_id[target]
    if src == dst:
        return True
    A = G.get_property("offdiag")
    q_src = Vector(bool, size=A.nrows, name="q_src")
    q_src[src] = True
    seen_src = q_src.dup(name="seen_src")
    q_dst = Vector(bool, size=A.nrows, name="q_dst")
    q_dst[dst] = True
    seen_dst = q_dst.dup(name="seen_dst", clear=True)
    any_pair_bool = any_pair[bool]
    for _i in range(A.nrows // 2):
        q_src(~seen_src.S, replace) << any_pair_bool(q_src @ A)
        if q_src.nvals == 0:
            return False
        if any_pair_bool(q_src @ q_dst):
            return True

        seen_dst(q_dst.S) << True
        q_dst(~seen_dst.S, replace) << any_pair_bool(A @ q_dst)
        if q_dst.nvals == 0:
            return False
        if any_pair_bool(q_src @ q_dst):
            return True

        seen_src(q_src.S) << True
    return False
