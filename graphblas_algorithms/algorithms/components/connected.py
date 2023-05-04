from .._bfs import _bfs_plain
from ..exceptions import PointlessConcept


def is_connected(G):
    if len(G) == 0:
        raise PointlessConcept("Connectivity is undefined for the null graph.")
    return _bfs_plain(G, next(iter(G))).nvals == len(G)


def node_connected_component(G, n):
    return _bfs_plain(G, n)
