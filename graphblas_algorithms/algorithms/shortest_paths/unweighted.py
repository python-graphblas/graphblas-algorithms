from graphblas import Matrix, Vector, replace
from graphblas.semiring import any_pair

from .._bfs import _bfs_level, _bfs_levels
from ..exceptions import NoPath

__all__ = [
    "single_source_shortest_path_length",
    "single_target_shortest_path_length",
    "all_pairs_shortest_path_length",
]


def single_source_shortest_path_length(G, source, cutoff=None):
    return _bfs_level(G, source, cutoff=cutoff)


def single_target_shortest_path_length(G, target, cutoff=None):
    return _bfs_level(G, target, cutoff=cutoff, transpose=True)


def all_pairs_shortest_path_length(G, cutoff=None, *, nodes=None, expand_output=False):
    D = _bfs_levels(G, nodes, cutoff=cutoff)
    if nodes is not None and expand_output and D.ncols != D.nrows:
        ids = G.list_to_ids(nodes)
        rv = Matrix(D.dtype, D.ncols, D.ncols, name=D.name)
        rv[ids, :] = D
        return rv
    return D


def bidirectional_shortest_path_length(G, source, target):
    # Perform bidirectional BFS from source to target and target to source
    # TODO: have this raise NodeNotFound?
    if source not in G or target not in G:
        raise KeyError(f"Either source {source} or target {target} is not in G")  # NodeNotFound
    src = G._key_to_id[source]
    dst = G._key_to_id[target]
    if src == dst:
        return 0
    A = G.get_property("offdiag")
    q_src = Vector(bool, size=A.nrows, name="q_src")
    q_src[src] = True
    seen_src = q_src.dup(name="seen_src")
    q_dst = Vector(bool, size=A.nrows, name="q_dst")
    q_dst[dst] = True
    seen_dst = q_dst.dup(name="seen_dst", clear=True)
    any_pair_bool = any_pair[bool]
    for i in range(1, A.nrows + 1, 2):
        q_src(~seen_src.S, replace) << any_pair_bool(q_src @ A)
        if q_src.nvals == 0:
            raise NoPath(f"No path between {source} and {target}.")
        if any_pair_bool(q_src @ q_dst):
            return i

        seen_dst(q_dst.S) << True
        q_dst(~seen_dst.S, replace) << any_pair_bool(A @ q_dst)
        if q_dst.nvals == 0:
            raise NoPath(f"No path between {source} and {target}.")
        if any_pair_bool(q_src @ q_dst):
            return i + 1

        seen_src(q_src.S) << True
    raise NoPath(f"No path between {source} and {target}.")
