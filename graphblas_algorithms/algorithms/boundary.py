from graphblas import binary
from graphblas.semiring import any_pair

__all__ = ["edge_boundary", "node_boundary"]


def edge_boundary(G, nbunch1, nbunch2=None, *, is_weighted=False):
    if is_weighted:
        B = binary.second(nbunch1 & G._A).new(name="boundary")
    else:
        B = binary.pair(nbunch1 & G._A).new(name="boundary")
    if nbunch2 is None:
        # Default nbunch2 is the complement of nbunch1.
        # We get the row_degrees to better handle hypersparse data.
        nbunch2 = G.get_property("row_degrees+", mask=~nbunch1.S)
    if is_weighted:
        B << binary.first(B & nbunch2)
    else:
        B << binary.pair(B & nbunch2)
    return B


def node_boundary(G, nbunch1, *, mask=None):
    if mask is None:
        mask = ~nbunch1.S
    else:
        mask = mask & (~nbunch1.S)
    bdy = any_pair(nbunch1 @ G._A).new(mask=mask, name="boundary")
    return bdy
