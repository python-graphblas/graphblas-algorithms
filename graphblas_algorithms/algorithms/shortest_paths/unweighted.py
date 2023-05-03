from graphblas import Matrix

from .._bfs import _bfs_level, _bfs_levels

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
