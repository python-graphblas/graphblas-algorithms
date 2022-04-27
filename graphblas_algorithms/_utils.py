import graphblas as gb
from graphblas import Vector, binary


def graph_to_adjacency(G, weight=None, dtype=None, *, name=None):
    key_to_id = {k: i for i, k in enumerate(G)}
    A = gb.io.from_networkx(G, nodelist=key_to_id, weight=weight, dtype=dtype, name=name)
    return A, key_to_id


def dict_to_vector(d, key_to_id, *, size=None, dtype=None, name=None):
    if d is None:
        return None
    if size is None:
        size = len(key_to_id)
    indices, values = zip(*((key_to_id[key], val) for key, val in d.items()))
    return Vector.from_values(indices, values, size=size, dtype=dtype, name=name)


def list_to_vector(nodes, key_to_id, *, size=None, name=None):
    if nodes is None:
        return None, None
    if size is None:
        size = len(key_to_id)
    id_to_key = {key_to_id[key]: key for key in nodes}
    v = Vector.from_values(list(id_to_key), True, size=size, dtype=bool, name=name)
    return v, id_to_key


def list_to_mask(nodes, key_to_id, *, size=None, name="mask"):
    if nodes is None:
        return None, None
    v, id_to_key = list_to_vector(nodes, key_to_id, size=size, name=name)
    return v.S, id_to_key


def vector_to_dict(v, key_to_id, id_to_key=None, *, mask=None, fillvalue=None):
    # This mutates the vector to fill it!
    if id_to_key is None:
        id_to_key = {key_to_id[key]: key for key in key_to_id}
    if mask is not None:
        if fillvalue is not None and v.nvals < mask.parent.nvals:
            v(mask, binary.first) << fillvalue
    elif fillvalue is not None and v.nvals < v.size:
        v(mask=~v.S) << fillvalue
    return {id_to_key[index]: value for index, value in zip(*v.to_values(sort=False))}
