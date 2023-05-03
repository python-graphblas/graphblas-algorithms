from .._bfs import _bfs_plain_bidirectional
from ..exceptions import PointlessConcept


def is_weakly_connected(G):
    if len(G) == 0:
        raise PointlessConcept("Connectivity is undefined for the null graph.")
    return _bfs_plain_bidirectional(G, next(iter(G))).nvals == len(G)
