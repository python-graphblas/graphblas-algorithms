from .._bfs import _plain_bfs
from ..exceptions import PointlessConcept


def is_connected(G):
    if len(G) == 0:
        raise PointlessConcept("Connectivity is undefined for the null graph.")
    return _plain_bfs(G, next(iter(G))).nvals == len(G)


def node_connected_component(G, n):
    return _plain_bfs(G, n)
