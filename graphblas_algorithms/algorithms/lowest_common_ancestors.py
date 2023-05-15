from graphblas import binary, replace
from graphblas.semiring import any_pair

from ._bfs import _bfs_plain

__all__ = ["lowest_common_ancestor"]


def lowest_common_ancestor(G, node1, node2, default=None):
    common_ancestors = _bfs_plain(G, node1, name="common_ancestors", transpose=True)
    other_ancestors = _bfs_plain(G, node2, name="other_ancestors", transpose=True)
    common_ancestors << binary.pair(common_ancestors & other_ancestors)
    if common_ancestors.nvals == 0:
        return default
    # Take one BFS step along predecessors. The lowest common ancestor is one we don't visit.
    # An alternative strategy would be to walk along successors until there are no more.
    other_ancestors(common_ancestors.S, replace) << any_pair[bool](G._A @ common_ancestors)
    common_ancestors(~other_ancestors.S, replace) << common_ancestors
    index = common_ancestors.to_coo(values=False)[0][0]
    # XXX: should we return index or key?
    return G.id_to_key[index]
