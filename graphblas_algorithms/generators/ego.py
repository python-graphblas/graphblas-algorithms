from ..algorithms.components.connected import _bfs_plain
from ..algorithms.shortest_paths.weighted import single_source_bellman_ford_path_length

__all__ = ["ego_graph"]


def ego_graph(G, n, radius=1, center=True, undirected=False, is_weighted=False):
    # TODO: should we have an option to keep the output matrix the same size?
    if undirected and G.is_directed():
        # NOT COVERED
        G2 = G.to_undirected()
    else:
        G2 = G
    if is_weighted:
        v = single_source_bellman_ford_path_length(G2, n, cutoff=radius)
    else:
        v = _bfs_plain(G2, n, cutoff=radius)
    if not center:
        del v[G._key_to_id[n]]

    indices, _ = v.to_coo(values=False)
    A = G._A[indices, indices].new(name="ego")
    key_to_id = G.renumber_key_to_id(indices.tolist())
    return type(G)(A, key_to_id=key_to_id)
