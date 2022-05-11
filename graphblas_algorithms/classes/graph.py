import networkx as nx
from graphblas import Matrix, Vector, agg, binary, select

import graphblas_algorithms as ga

from . import _utils


def get_AT(A, cache, mask=None):
    cache["AT"] = A
    return A


def get_Up(A, cache, mask=None):
    if "U+" not in cache:
        if "U-" in cache and not has_self_edges(A, cache):
            cache["U+"] = cache["U-"]
        else:
            cache["U+"] = select.triu(A).new(name="U+")
    if "has_self_edges" not in cache:
        cache["has_self_edges"] = 2 * cache["U+"].nvals > A.nvals
    if not cache["has_self_edges"]:
        cache["U-"] = cache["U+"]
    return cache["U+"]


def get_Lp(A, cache, mask=None):
    if "L+" not in cache:
        if "L-" in cache and not has_self_edges(A, cache):
            cache["L+"] = cache["L-"]
        else:
            cache["L+"] = select.tril(A).new(name="L+")
    if "has_self_edges" not in cache:
        cache["has_self_edges"] = 2 * cache["L+"].nvals > A.nvals
    if not cache["has_self_edges"]:
        cache["L-"] = cache["L+"]
    return cache["L+"]


def get_Um(A, cache, mask=None):
    if "U-" not in cache:
        if "U+" in cache:
            if has_self_edges(A, cache):
                cache["U-"] = select.triu(cache["U+"], 1).new(name="U-")
            else:
                cache["U-"] = cache["U+"]
        else:
            cache["U-"] = select.triu(A, 1).new(name="U-")
    if "has_self_edges" not in cache:
        cache["has_self_edges"] = 2 * cache["U-"].nvals < A.nvals
    if not cache["has_self_edges"]:
        cache["U+"] = cache["U-"]
    return cache["U-"]


def get_Lm(A, cache, mask=None):
    if "L-" not in cache:
        if "L+" in cache:
            if has_self_edges(A, cache):
                cache["L-"] = select.tril(cache["L+"], -1).new(name="L-")
            else:
                cache["L-"] = cache["L+"]
        else:
            cache["L-"] = select.tril(A, -1).new(name="L-")
    if "has_self_edges" not in cache:
        cache["has_self_edges"] = 2 * cache["L-"].nvals < A.nvals
    if not cache["has_self_edges"]:
        cache["L+"] = cache["L-"]
    return cache["L-"]


def get_diag(A, cache, mask=None):
    if "diag" not in cache:
        if cache.get("has_self_edges") is False:
            cache["diag"] = Vector(A.dtype, size=A.nrows, name="diag")
        elif "U+" in cache:
            cache["diag"] = cache["U+"].diag(name="diag")
        elif "L+" in cache:
            cache["diag"] = cache["L+"].diag(name="diag")
        else:
            cache["diag"] = A.diag(name="diag")
    if "has_self_edges" not in cache:
        cache["has_self_edges"] = cache["diag"].nvals > 0
    return cache["diag"]


def has_self_edges(A, cache, mask=None):
    if "has_self_edges" not in cache:
        if "L+" in cache:
            cache["has_self_edges"] = 2 * cache["L+"].nvals > A.nvals
        elif "L-" in cache:
            cache["has_self_edges"] = 2 * cache["L-"].nvals < A.nvals
        elif "U+" in cache:
            cache["has_self_edges"] = 2 * cache["U+"].nvals > A.nvals
        elif "U-" in cache:
            cache["has_self_edges"] = 2 * cache["U-"].nvals < A.nvals
        else:
            get_diag(A, cache)
    return cache["has_self_edges"]


def get_degreesp(A, cache, mask=None):
    if mask is not None:
        if "degrees+" not in cache:
            return A.reduce_rowwise(agg.count).new(mask=mask, name="degrees+")
        else:
            return cache["degrees+"].dup(mask=mask)
    if "degrees+" not in cache:
        cache["degrees+"] = A.reduce_rowwise(agg.count).new(name="degrees+")
    return cache["degrees+"]


def get_degreesm(A, cache, mask=None):
    if mask is not None:
        if "degrees-" in cache:
            return cache["degrees-"].dup(mask=mask)
        elif "L-" in cache and "U-" in cache and has_self_edges(A, cache):
            return (
                cache["L-"].reduce_rowwise(agg.count).new(mask=mask)
                + cache["U-"].reduce_rowwise(agg.count).new(mask=mask)
            ).new(name="degrees-")
        else:
            return get_degreesm(A, cache).dup(mask=mask)
    if "degrees-" not in cache:
        if has_self_edges(A, cache):
            if "L-" in cache and "U-" in cache:
                cache["degrees-"] = (
                    cache["L-"].reduce_rowwise(agg.count) + cache["U-"].reduce_rowwise(agg.count)
                ).new(name="degrees-")
            else:
                # Is there a better way to do this?
                degrees = get_degreesp(A, cache).dup()
                diag = get_diag(A, cache)
                degrees(binary.plus, diag.S) << -1
                degrees(degrees.V, replace=True) << degrees  # drop 0s
                cache["degrees-"] = degrees
        else:
            cache["degrees-"] = get_degreesp(A, cache)
    return cache["degrees-"]


def to_undirected_graph(G, weight=None, dtype=None):
    # We should do some sanity checks here to ensure we're returning a valid undirected graph
    if isinstance(G, Graph):
        return G
    elif isinstance(G, nx.Graph):
        return Graph.from_networkx(G, weight=weight, dtype=dtype)
    elif isinstance(G, Matrix):
        return Graph.from_graphblas(G)
    else:
        raise TypeError()


class Graph:
    # "-" properties ignore self-edges, "+" properties include self-edges
    _property_priority = {
        key: i
        for i, key in enumerate(
            [
                "AT",
                "U+",
                "L+",
                "U-",
                "L-",
                "diag",
                "has_self_edges",
                "degrees+",
                "degrees-",
            ]
        )
    }
    _get_property = {
        "AT": get_AT,
        "U+": get_Up,
        "L+": get_Lp,
        "U-": get_Um,
        "L-": get_Lm,
        "diag": get_diag,
        "has_self_edges": has_self_edges,
        "degrees+": get_degreesp,
        "degrees-": get_degreesm,
    }
    graph_attr_dict_factory = dict

    def __init__(self, incoming_graph_data=None, **attr):
        if incoming_graph_data is not None:
            raise NotImplementedError("incoming_graph_data is not None")
        self.graph_attr_dict_factory = self.graph_attr_dict_factory
        self.graph = self.graph_attr_dict_factory()  # dictionary for graph attributes
        self.graph.update(attr)

        # Graphblas-specific properties
        self._A = Matrix()
        self._key_to_id = {}
        self._id_to_key = None
        self._cache = {}

    # Graphblas-specific methods
    from_networkx = classmethod(_utils.from_networkx)
    from_graphblas = classmethod(_utils.from_graphblas)
    get_property = _utils.get_property
    get_properties = _utils.get_properties
    dict_to_vector = _utils.dict_to_vector
    list_to_vector = _utils.list_to_vector
    list_to_mask = _utils.list_to_mask
    vector_to_dict = _utils.vector_to_dict

    def to_directed_class(self):
        return ga.DiGraph

    def to_undirected_class(self):
        return Graph

    @property
    def name(self):
        return self.graph.get("name", "")

    @name.setter
    def name(self, s):
        self._A.name = s
        self.graph["name"] = s

    def __iter__(self):
        return iter(self._key_to_id)

    def __contains__(self, n):
        try:
            return n in self._key_to_id
        except TypeError:
            return False

    def __len__(self):
        return self._A.nrows

    def number_of_nodes(self):
        return self._A.nrows

    def order(self):
        return self._A.nrows

    def is_multigraph(self):
        return False

    def is_directed(self):
        return False
