from collections import defaultdict

from graphblas import Matrix, Vector, select

import graphblas_algorithms as ga

from . import _utils
from ._caching import get_reduce_to_scalar, get_reduce_to_vector


def get_A(G, mask=None):
    """A"""
    return G._A


def get_AT(G, mask=None):
    """A.T"""
    A = G._A
    G._cache["AT"] = A
    return A


def get_offdiag(G, mask=None):
    """select.offdiag(A)"""
    A = G._A
    cache = G._cache
    if "offdiag" not in cache:
        if cache.get("has_self_edges") is False:
            cache["offdiag"] = A
        else:
            cache["offdiag"] = select.offdiag(A).new(name="offdiag")
    if "has_self_edges" not in cache:
        cache["has_self_edges"] = A.nvals > cache["offdiag"].nvals
    if not cache["has_self_edges"]:
        cache["offdiag"] = A
    return cache["offdiag"]


def get_Up(G, mask=None):
    """select.triu(A)"""
    A = G._A
    cache = G._cache
    if "U+" not in cache:
        if "U-" in cache and not G.get_property("has_self_edges"):
            cache["U+"] = cache["U-"]
        else:
            cache["U+"] = select.triu(A).new(name="U+")
    if "has_self_edges" not in cache:
        cache["has_self_edges"] = 2 * cache["U+"].nvals > A.nvals
    if not cache["has_self_edges"]:
        cache["U-"] = cache["U+"]
    return cache["U+"]


def get_Lp(G, mask=None):
    """select.tril(A)"""
    A = G._A
    cache = G._cache
    if "L+" not in cache:
        if "L-" in cache and not G.get_property("has_self_edges"):
            cache["L+"] = cache["L-"]
        else:
            cache["L+"] = select.tril(A).new(name="L+")
    if "has_self_edges" not in cache:
        cache["has_self_edges"] = 2 * cache["L+"].nvals > A.nvals
    if not cache["has_self_edges"]:
        cache["L-"] = cache["L+"]
    return cache["L+"]


def get_Um(G, mask=None):
    """select.triu(A, 1)"""
    A = G._A
    cache = G._cache
    if "U-" not in cache:
        if "U+" in cache:
            if G.get_property("has_self_edges"):
                cache["U-"] = select.triu(cache["U+"], 1).new(name="U-")
            else:
                cache["U-"] = cache["U+"]
        elif "offdiag" in cache:
            cache["U-"] = select.triu(cache["offdiag"], 1).new(name="U-")
        else:
            cache["U-"] = select.triu(A, 1).new(name="U-")
    if "has_self_edges" not in cache:
        cache["has_self_edges"] = 2 * cache["U-"].nvals < A.nvals
    if not cache["has_self_edges"]:
        cache["U+"] = cache["U-"]
    return cache["U-"]


def get_Lm(G, mask=None):
    """select.tril(A, -1)"""
    A = G._A
    cache = G._cache
    if "L-" not in cache:
        if "L+" in cache:
            if G.get_property("has_self_edges"):
                cache["L-"] = select.tril(cache["L+"], -1).new(name="L-")
            else:
                cache["L-"] = cache["L+"]
        elif "offdiag" in cache:
            cache["L-"] = select.tril(cache["offdiag"], -1).new(name="L-")
        else:
            cache["L-"] = select.tril(A, -1).new(name="L-")
    if "has_self_edges" not in cache:
        cache["has_self_edges"] = 2 * cache["L-"].nvals < A.nvals
    if not cache["has_self_edges"]:
        cache["L+"] = cache["L-"]
    return cache["L-"]


def get_diag(G, mask=None):
    """A.diag()"""
    A = G._A
    cache = G._cache
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


def has_self_edges(G, mask=None):
    """A.diag().nvals > 0"""
    A = G._A
    cache = G._cache
    if "has_self_edges" not in cache:
        if "L+" in cache:
            cache["has_self_edges"] = 2 * cache["L+"].nvals > A.nvals
        elif "L-" in cache:
            cache["has_self_edges"] = 2 * cache["L-"].nvals < A.nvals
        elif "U+" in cache:
            cache["has_self_edges"] = 2 * cache["U+"].nvals > A.nvals
        elif "U-" in cache:
            cache["has_self_edges"] = 2 * cache["U-"].nvals < A.nvals
        elif "offdiag" in cache:
            cache["has_self_edges"] = A.nvals > cache["offdiag"].nvals
        else:
            G.get_property("diag")
    return cache["has_self_edges"]


def to_undirected_graph(G, weight=None, dtype=None):
    # We should do some sanity checks here to ensure we're returning a valid undirected graph
    if isinstance(G, Graph):
        return G
    try:
        return Graph.from_graphblas(G)
    except TypeError:
        pass

    try:
        import networkx as nx

        if isinstance(G, nx.Graph):
            return Graph.from_networkx(G, weight=weight, dtype=dtype)
    except ImportError:
        pass

    raise TypeError()


class AutoDict(dict):
    def __missing__(self, key):
        # Automatically compute keys such as "plus_rowwise-" and "max_element+"
        if key[-1] in {"-", "+"}:
            keybase = key[:-1]
            if keybase.endswith("_rowwise"):
                opname = keybase[: -len("_rowwise")]
                methodname = "reduce_rowwise"
            elif keybase.endswith("_columnwise"):
                opname = keybase[: -len("_columnwise")]
                methodname = "reduce_rowwise"
            elif keybase.endswith("_element"):
                opname = keybase[: -len("_element")]
                methodname = "reduce_scalar"
            else:
                raise KeyError(key)
            if methodname == "reduce_scalar":
                get_reduction = get_reduce_to_scalar(key, opname)
            else:
                get_reduction = get_reduce_to_vector(key, opname, methodname)
                self[f"{opname}_columnwise{key[-1]}"] = get_reduction
            self[key] = get_reduction
            return get_reduction
        raise KeyError(key)


class Graph:
    __networkx_plugin__ = "graphblas"

    # "-" properties ignore self-edges, "+" properties include self-edges
    # Ideally, we would have "max_rowwise+" come before "max_element+".
    _property_priority = defaultdict(
        lambda: Graph._property_priority["has_self_edges"] - 0.5,
        {
            key: i
            for i, key in enumerate(
                [
                    "A",
                    "AT",
                    "offdiag",
                    "U+",
                    "L+",
                    "U-",
                    "L-",
                    "diag",
                    "count_rowwise+",
                    "count_rowwise-",
                    "has_self_edges",
                ]
            )
        },
    )
    _get_property = AutoDict(
        {
            "A": get_A,
            "AT": get_AT,
            "offdiag": get_offdiag,
            "U+": get_Up,
            "L+": get_Lp,
            "U-": get_Um,
            "L-": get_Lm,
            "diag": get_diag,
            "has_self_edges": has_self_edges,
        }
    )
    _cache_aliases = {
        "degrees+": "count_rowwise+",
        "degrees-": "count_rowwise-",
        "row_degrees+": "count_rowwise+",
        "row_degrees-": "count_rowwise-",
        "column_degrees+": "count_rowwise+",
        "column_degrees-": "count_rowwise-",
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
    id_to_key = property(_utils.id_to_key)
    get_property = _utils.get_property
    get_properties = _utils.get_properties
    dict_to_vector = _utils.dict_to_vector
    list_to_vector = _utils.list_to_vector
    list_to_mask = _utils.list_to_mask
    list_to_ids = _utils.list_to_ids
    list_to_keys = _utils.list_to_keys
    matrix_to_dicts = _utils.matrix_to_dicts
    set_to_vector = _utils.set_to_vector
    to_networkx = _utils.to_networkx
    vector_to_dict = _utils.vector_to_dict
    vector_to_nodemap = _utils.vector_to_nodemap
    vector_to_nodeset = _utils.vector_to_nodeset
    vector_to_set = _utils.vector_to_set
    _cacheit = _utils._cacheit

    # NetworkX methods
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


class MultiGraph(Graph):
    def is_multigraph(self):
        return True


__all__ = ["Graph", "MultiGraph"]
