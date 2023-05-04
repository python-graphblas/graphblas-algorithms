from graphblas import Vector, replace
from graphblas.semiring import any_pair

__all__ = [
    "bfs_layers",
    "descendants_at_distance",
]


def bfs_layers(G, sources):
    if sources in G:
        sources = [sources]
    ids = G.list_to_ids(sources)
    if not ids:
        return
    A = G.get_property("offdiag")
    n = A.nrows
    v = Vector(bool, size=n, name="bfs_layers")
    q = Vector.from_coo(ids, True, size=n, name="q")
    any_pair_bool = any_pair[bool]
    yield q.dup(name="bfs_layer_0")
    for i in range(1, n):
        v(q.S) << True
        q(~v.S, replace) << any_pair_bool(q @ A)
        if q.nvals == 0:
            return
        yield q.dup(name=f"bfs_layer_{i}")


def descendants_at_distance(G, source, distance):
    index = G._key_to_id[source]
    A = G.get_property("offdiag")
    n = A.nrows
    q = Vector(bool, size=n, name=f"descendants_at_distance_{distance}")
    q[index] = True
    if distance == 0:
        return q
    v = Vector(bool, size=n, name="bfs_seen")
    any_pair_bool = any_pair[bool]
    for _i in range(1, distance + 1):
        v(q.S) << True
        q(~v.S, replace) << any_pair_bool(q @ A)
        if q.nvals == 0:
            break
    return q
