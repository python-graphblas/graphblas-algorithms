import numpy as np
from graphblas import binary

from ..cluster import triangles

__all__ = [
    "fast_could_be_isomorphic",
    "faster_could_be_isomorphic",
]


def fast_could_be_isomorphic(G1, G2):
    if len(G1) != len(G2):
        return False
    d1 = G1.get_property("total_degrees+" if G1.is_directed() else "degrees+")
    d2 = G2.get_property("total_degrees+" if G2.is_directed() else "degrees+")
    if d1.nvals != d2.nvals:
        return False
    t1 = triangles(G1)
    t2 = triangles(G2)
    if t1.nvals != t2.nvals:
        return False
    # Make ds and ts the same shape as numpy arrays so we can sort them lexicographically.
    if t1.nvals != d1.nvals:
        # Assign 0 to t1 where present in d1 but not t1
        t1(~t1.S) << binary.second(d1, 0)
    if t2.nvals != d2.nvals:
        # Assign 0 to t2 where present in d2 but not t2
        t2(~t2.S) << binary.second(d2, 0)
    d1 = d1.to_coo(indices=False)[1]
    d2 = d2.to_coo(indices=False)[1]
    t1 = t1.to_coo(indices=False)[1]
    t2 = t2.to_coo(indices=False)[1]
    ind1 = np.lexsort((d1, t1))
    ind2 = np.lexsort((d2, t2))
    if not np.array_equal(d1[ind1], d2[ind2]):
        return False
    if not np.array_equal(t1[ind1], t2[ind2]):
        return False
    return True


def faster_could_be_isomorphic(G1, G2):
    if len(G1) != len(G2):
        return False
    d1 = G1.get_property("total_degrees+" if G1.is_directed() else "degrees+")
    d2 = G2.get_property("total_degrees+" if G2.is_directed() else "degrees+")
    if d1.nvals != d2.nvals:
        return False
    d1 = d1.to_coo(indices=False)[1]
    d2 = d2.to_coo(indices=False)[1]
    d1.sort()
    d2.sort()
    if not np.array_equal(d1, d2):
        return False
    return True
