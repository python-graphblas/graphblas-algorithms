__all__ = ["mutual_weight"]


def mutual_weight(G, u, v):
    key_to_id = G._key_to_id
    if u not in key_to_id or v not in key_to_id:
        return 0
    u = key_to_id[u]
    v = key_to_id[v]
    A = G._A
    return A.get(u, v, 0) + A.get(v, u, 0)
