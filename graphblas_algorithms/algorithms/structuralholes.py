from graphblas_algorithms.classes.digraph import to_graph
from graphblas_algorithms.utils import get_all


def mutual_weight_core(G, u, v):
    key_to_id = G._key_to_id
    if u not in key_to_id or v not in key_to_id:
        return 0
    u = key_to_id[u]
    v = key_to_id[v]
    A = G._A
    return A.get(u, v, 0) + A.get(v, u, 0)


def mutual_weight(G, u, v, weight=None):
    G = to_graph(G, weight=weight)
    return mutual_weight_core(G, u, v)


__all__ = get_all(__name__)
