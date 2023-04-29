from .._bfs import _plain_bfs_bidirectional
from ..exceptions import PointlessConcept


def is_weakly_connected(G):
    if len(G) == 0:
        raise PointlessConcept("Connectivity is undefined for the null graph.")
    return _plain_bfs_bidirectional(G, next(iter(G))).nvals == len(G)
