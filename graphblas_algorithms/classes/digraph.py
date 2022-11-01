from collections import defaultdict

from graphblas import Matrix, Vector, binary, replace, select, unary

import graphblas_algorithms as ga

from . import _utils
from ._caching import get_reduce_to_scalar, get_reduce_to_vector
from .graph import Graph


def get_A(G, mask=None):
    """A"""
    return G._A


def get_AT(G, mask=None):
    """A.T"""
    A = G._A
    cache = G._cache
    if "AT" not in cache:
        cache["AT"] = A.T.new()
    return cache["AT"]


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
        if "U-" in cache:
            cache["has_self_edges"] = cache["U+"].nvals > cache["U-"].nvals
        elif "L+" in cache:
            cache["has_self_edges"] = cache["U+"].nvals + cache["L+"].nvals > A.nvals
    if cache.get("has_self_edges") is False:
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
        if "L-" in cache:
            cache["has_self_edges"] = cache["L+"].nvals > cache["L-"].nvals
        elif "U+" in cache:
            cache["has_self_edges"] = cache["L+"].nvals + cache["U+"].nvals > A.nvals
    if cache.get("has_self_edges") is False:
        cache["L-"] = cache["L+"]
    return cache["L+"]


def get_Um(G, mask=None):
    """select.triu(A, 1)"""
    A = G._A
    cache = G._cache
    if "U-" not in cache:
        if "U+" in cache:
            if cache.get("has_self_edges") is False:
                cache["U-"] = cache["U+"]
            else:
                cache["U-"] = select.triu(cache["U+"], 1).new(name="U-")
        elif "offdiag" in cache:
            cache["U-"] = select.triu(cache["offdiag"], 1).new(name="U-")
        else:
            cache["U-"] = select.triu(A, 1).new(name="U-")
    if "has_self_edges" not in cache:
        if "U+" in cache:
            cache["has_self_edges"] = cache["U-"].nvals < cache["U+"].nvals
        elif "L-" in cache:
            cache["has_self_edges"] = cache["U-"].nvals + cache["L-"].nvals < A.nvals
    if cache.get("has_self_edges") is False:
        cache["U+"] = cache["U-"]
    return cache["U-"]


def get_Lm(G, mask=None):
    """select.tril(A, -1)"""
    A = G._A
    cache = G._cache
    if "L-" not in cache:
        if "L+" in cache:
            if cache.get("has_self_edges") is False:
                cache["L-"] = cache["L+"]
            else:
                cache["L-"] = select.tril(cache["L+"], -1).new(name="L-")
        elif "offdiag" in cache:
            cache["L-"] = select.tril(cache["offdiag"], -1).new(name="L-")
        else:
            cache["L-"] = select.tril(A, -1).new(name="L-")
    if "has_self_edges" not in cache:
        if "L+" in cache:
            cache["has_self_edges"] = cache["L-"].nvals < cache["L+"].nvals
        elif "U-" in cache:
            cache["has_self_edges"] = cache["L-"].nvals + cache["U-"].nvals < A.nvals
    if cache.get("has_self_edges") is False:
        cache["L+"] = cache["L-"]
    return cache["L-"]


def get_diag(G, mask=None):
    """select.diag(A)"""
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
    if mask is not None:
        return cache["diag"].dup(mask=mask)
    return cache["diag"]


def get_recip_degreesp(G, mask=None):
    """pair(A & A.T).reduce_rowwise()"""
    A = G._A
    cache = G._cache
    if "AT" in cache:
        AT = cache["AT"]
    else:
        AT = A.T
    if mask is not None:
        if "recip_degrees+" in cache:
            return cache["recip_degrees+"].dup(mask=mask)
        elif cache.get("has_self_edges") is False and "recip_degrees-" in cache:
            cache["recip_degrees+"] = cache["recip_degrees-"]
            return cache["recip_degrees-"].dup(mask=mask)
        elif "recip_degrees-" in cache and "diag" in cache:
            return (unary.one(cache["diag"]) + cache["recip_degrees-"]).new(
                mask=mask, name="recip_degrees+"
            )
        elif "recip_degrees-" in cache and not G.get_property("has_self_edges"):
            return cache["recip_degrees-"].dup(mask=mask)
        else:
            return binary.pair(A & AT).reduce_rowwise().new(mask=mask, name="recip_degrees+")
    if "recip_degrees+" not in cache:
        if cache.get("has_self_edges") is False and "recip_degrees-" in cache:
            cache["recip_degrees+"] = cache["recip_degrees-"]
        elif "recip_degrees-" in cache and "diag" in cache:
            cache["recip_degrees+"] = (unary.one(cache["diag"]) + cache["recip_degrees-"]).new(
                name="recip_degrees+"
            )
        elif "recip_degrees-" in cache and not G.get_property("has_self_edges"):
            cache["recip_degrees+"] = cache["recip_degrees-"]
        else:
            cache["recip_degrees+"] = (
                binary.pair(A & AT).reduce_rowwise().new(name="recip_degrees+")
            )
    if (
        "has_self_edges" not in cache
        and "recip_degrees-" in cache
        and cache["recip_degrees-"].nvals != cache["recip_degrees+"].nvals
    ):
        cache["has_self_edges"] = True
    elif cache.get("has_self_edges") is False:
        cache["recip_degrees-"] = cache["recip_degrees+"]
    return cache["recip_degrees+"]


def get_recip_degreesm(G, mask=None):
    """C = select.offdiag(A) ; pair(C & C.T).reduce_rowwise()"""
    A = G._A
    cache = G._cache
    if "AT" in cache:
        AT = cache["AT"]
    elif "offdiag" in cache:
        AT = cache["offdiag"].T
    else:
        AT = A.T
    if mask is not None:
        if "recip_degrees-" in cache:
            return cache["recip_degrees-"].dup(mask=mask)
        elif cache.get("has_self_edges") is False and "recip_degrees+" in cache:
            cache["recip_degrees-"] = cache["recip_degrees+"]
            return cache["recip_degrees-"].dup(mask=mask)
        elif "recip_degrees+" in cache and "diag" in cache:
            rv = binary.minus(cache["recip_degrees+"] | unary.one(cache["diag"])).new(
                mask=mask, name="recip_degrees-"
            )
            rv(rv.V, replace) << rv  # drop 0s
            return rv
        elif not G.get_property("has_self_edges"):
            return G.get_property("recip_degrees+", mask=mask)
        elif "offdiag" in cache:
            return (
                binary.pair(cache["offdiag"] & AT)
                .reduce_rowwise()
                .new(mask=mask, name="recip_degrees-")
            )
        elif "L-" in cache and "U-" in cache:
            return (
                binary.pair(cache["L-"] & AT).reduce_rowwise().new(mask=mask)
                + binary.pair(cache["U-"] & AT).reduce_rowwise().new(mask=mask)
            ).new(name="recip_degrees-")
        else:
            diag = G.get_property("diag", mask=mask)
            overlap = binary.pair(A & AT).reduce_rowwise().new(mask=mask)
            rv = binary.minus(overlap | unary.one(diag)).new(name="recip_degrees-")
            rv(rv.V, replace) << rv  # drop 0s
            return rv
    if "recip_degrees-" not in cache:
        if cache.get("has_self_edges") is False and "recip_degrees+" in cache:
            cache["recip_degrees-"] = cache["recip_degrees+"]
        elif "recip_degrees+" in cache and "diag" in cache:
            rv = binary.minus(cache["recip_degrees+"] | unary.one(cache["diag"])).new(
                name="recip_degrees-"
            )
            rv(rv.V, replace) << rv  # drop 0s
            cache["recip_degrees-"] = rv
        elif not G.get_property("has_self_edges"):
            cache["recip_degrees-"] = G.get_property("recip_degrees+")
        elif "offdiag" in cache:
            cache["recip_degrees-"] = (
                binary.pair(cache["offdiag"] & AT).reduce_rowwise().new(name="recip_degrees-")
            )
        elif "L-" in cache and "U-" in cache:
            cache["recip_degrees-"] = (
                binary.pair(cache["L-"] & AT).reduce_rowwise().new()
                + binary.pair(cache["U-"] & AT).reduce_rowwise().new()
            ).new(name="recip_degrees-")
        else:
            diag = G.get_property("diag")
            overlap = binary.pair(A & AT).reduce_rowwise().new()
            rv = binary.minus(overlap | unary.one(diag)).new(name="recip_degrees-")
            rv(rv.V, replace) << rv  # drop 0s
            cache["recip_degrees-"] = rv
    if (
        "has_self_edges" not in cache
        and "recip_degrees+" in cache
        and cache["recip_degrees-"].nvals != cache["recip_degrees+"].nvals
    ):
        cache["has_self_edges"] = True
    elif cache.get("has_self_edges") is False:
        cache["recip_degrees+"] = cache["recip_degrees-"]
    return cache["recip_degrees-"]


def get_total_degreesp(G, mask=None):
    """A.reduce_rowwise(agg.count) + A.reduce_columnwise(agg.count)"""
    cache = G._cache
    if mask is not None:
        if "total_degrees+" in cache:
            return cache["total_degrees+"].dup(mask=mask)
        elif cache.get("has_self_edges") is False and "total_degrees-" in cache:
            cache["total_degrees+"] = cache["total_degrees-"]
            return cache["total_degrees+"].dup(mask=mask)
        else:
            return (
                G.get_property("row_degrees+", mask=mask)
                + G.get_property("column_degrees+", mask=mask)
            ).new(name="total_degrees+")
    if "total_degrees+" not in cache:
        if cache.get("has_self_edges") is False and "total_degrees-" in cache:
            cache["total_degrees+"] = cache["total_degrees-"]
        else:
            cache["total_degrees+"] = (
                G.get_property("row_degrees+") + G.get_property("column_degrees+")
            ).new(name="total_degrees+")
    if (
        "has_self_edges" not in cache
        and "total_degrees-" in cache
        and cache["total_degrees-"].nvals != cache["total_degrees+"].nvals
    ):
        cache["has_self_edges"] = True
    elif cache.get("has_self_edges") is False:
        cache["total_degrees-"] = cache["total_degrees+"]
    return cache["total_degrees+"]


def get_total_degreesm(G, mask=None):
    """C = select.offdiag(A) ; C.reduce_rowwise(agg.count) + C.reduce_columnwise(agg.count)"""
    cache = G._cache
    if mask is not None:
        if "total_degrees-" in cache:
            return cache["total_degrees-"].dup(mask=mask)
        elif cache.get("has_self_edges") is False and "total_degrees+" in cache:
            cache["total_degrees-"] = cache["total_degrees+"]
            return cache["total_degrees-"].dup(mask=mask)
        else:
            return (
                G.get_property("row_degrees-", mask=mask)
                + G.get_property("column_degrees-", mask=mask)
            ).new(name="total_degrees-")
    if "total_degrees-" not in cache:
        if cache.get("has_self_edges") is False and "total_degrees+" in cache:
            cache["total_degrees-"] = cache["total_degrees+"]
        else:
            cache["total_degrees-"] = (
                G.get_property("row_degrees-") + G.get_property("column_degrees-")
            ).new(name="total_degrees-")
    if (
        "has_self_edges" not in cache
        and "total_degrees+" in cache
        and cache["total_degrees-"].nvals != cache["total_degrees+"].nvals
    ):
        cache["has_self_edges"] = True
    elif cache.get("has_self_edges") is False:
        cache["total_degrees+"] = cache["total_degrees-"]
    return cache["total_degrees-"]


def get_total_recipp(G, mask=None):
    """pair(A & A.T).reduce_scalar()"""
    A = G._A
    cache = G._cache
    if "total_recip+" not in cache:
        if "total_recip-" in cache and cache.get("has_self_edges") is False:
            cache["total_recip+"] = cache["total_recip-"]
        elif "recip_degrees+" in cache:
            cache["total_recip+"] = cache["recip_degrees+"].reduce().get(0)
        else:
            if "AT" in cache:
                AT = cache["AT"]
            else:
                AT = A.T
            cache["total_recip+"] = binary.pair(A & AT).reduce_scalar().get(0)
    if "has_self_edges" not in cache and "total_recip-" in cache:
        cache["has_self_edges"] = cache["total_recip+"] > cache["total_recip-"]
    if cache.get("has_self_edges") is False:
        cache["total_recip-"] = cache["total_recip+"]
    return cache["total_recip+"]


def get_total_recipm(G, mask=None):
    """C = select.offdiag(A) ; pair(C & C.T).reduce_scalar()"""
    cache = G._cache
    if "total_recip-" not in cache:
        if "total_recip+" in cache and cache.get("has_self_edges") is False:
            cache["total_recip-"] = cache["total_recip+"]
        else:
            cache["total_recip-"] = G.get_property("recip_degrees-").reduce().get(0)
    if "has_self_edges" not in cache and "total_recip+" in cache:
        cache["has_self_edges"] = cache["total_recip+"] > cache["total_recip-"]
    if cache.get("has_self_edges") is False:
        cache["total_recip+"] = cache["total_recip-"]
    return cache["total_recip-"]


def has_self_edges(G, mask=None):
    """A.diag().nvals > 0"""
    A = G._A
    cache = G._cache
    if "has_self_edges" not in cache:
        if "offdiag" in cache:
            cache["has_self_edges"] = A.nvals > cache["offdiag"].nvals
        elif "L+" in cache and ("L-" in cache or "U+" in cache):
            if "L-" in cache:
                cache["has_self_edges"] = cache["L-"].nvals < cache["L+"].nvals
            else:
                cache["has_self_edges"] = cache["L+"].nvals + cache["U+"].nvals > A.nvals
        elif "U-" in cache and ("U+" in cache or "L-" in cache):
            if "U+" in cache:
                cache["has_self_edges"] = cache["U-"].nvals < cache["U+"].nvals
            else:
                cache["has_self_edges"] = cache["U-"].nvals + cache["L-"].nvals < A.nvals
        elif "total_recip-" in cache and "total_recip+" in cache:
            cache["has_self_edges"] = cache["total_recip+"] > cache["total_recip-"]
        elif "row_degrees-" in cache and "row_degrees+" in cache:
            cache["has_self_edges"] = not cache["row_degrees-"].isequal(cache["row_degrees+"])
        elif "column_degrees-" in cache and "column_degrees+" in cache:
            cache["has_self_edges"] = not cache["column_degrees-"].isequal(cache["column_degrees+"])
        elif "total_degrees-" in cache and "total_degrees+" in cache:
            cache["has_self_edges"] = not cache["total_degrees-"].isequal(cache["total_degrees+"])
        elif "recip_degrees-" in cache and "recip_degrees+" in cache:
            cache["has_self_edges"] = not cache["recip_degrees-"].isequal(cache["recip_degrees+"])
        elif "row_degrees-" in cache:
            cache["has_self_edges"] = cache["row_degrees-"].reduce().get(0) < A.nvals
        elif "column_degrees-" in cache:
            cache["has_self_edges"] = cache["column_degrees-"].reduce().get(0) < A.nvals
        elif "total_degrees-" in cache:
            cache["has_self_edges"] = cache["total_degrees-"].reduce().get(0) < 2 * A.nvals
        elif "total_degrees+" in cache:
            cache["has_self_edges"] = cache["total_degrees+"].reduce().get(0) > 2 * A.nvals
        else:
            G.get_property("diag")
    return cache["has_self_edges"]


def to_directed_graph(G, weight=None, dtype=None):
    # We should do some sanity checks here to ensure we're returning a valid directed graph
    if isinstance(G, DiGraph):
        return G
    try:
        return DiGraph.from_graphblas(G)
    except TypeError:
        pass

    try:
        import networkx as nx

        if isinstance(G, nx.DiGraph):
            return DiGraph.from_networkx(G, weight=weight, dtype=dtype)
    except ImportError:
        pass

    raise TypeError()


def to_graph(G, weight=None, dtype=None):
    if isinstance(G, (DiGraph, ga.Graph)):
        return G
    try:
        # Should we check if it can be undirected?
        return DiGraph.from_graphblas(G)
    except TypeError:
        pass

    try:
        import networkx as nx

        if isinstance(G, nx.DiGraph):
            return DiGraph.from_networkx(G, weight=weight, dtype=dtype)
        if isinstance(G, nx.Graph):
            return ga.Graph.from_networkx(G, weight=weight, dtype=dtype)
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
                methodname = "reduce_columnwise"
            elif keybase.endswith("_element"):
                opname = keybase[: -len("_element")]
                methodname = "reduce_scalar"
            else:
                raise KeyError(key)
            if methodname == "reduce_scalar":
                get_reduction = get_reduce_to_scalar(key, opname)
            else:
                get_reduction = get_reduce_to_vector(key, opname, methodname)
            self[key] = get_reduction
            return get_reduction
        raise KeyError(key)


class DiGraph(Graph):
    __networkx_plugin__ = "graphblas"

    # "-" properties ignore self-edges, "+" properties include self-edges
    # Ideally, we would have "max_rowwise+" come before "max_element+".
    _property_priority = defaultdict(
        lambda: DiGraph._property_priority["has_self_edges"] - 0.5,
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
                    "count_rowwise+",  # row_degrees
                    "count_columnwise+",  # column_degrees
                    "count_rowwise-",
                    "count_columnwise-",
                    "recip_degrees+",
                    "recip_degrees-",
                    "total_degrees+",
                    "total_degrees-",
                    "total_recip+",  # scalar; I don't like this name
                    "total_recip-",  # scalar; I don't like this name
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
            "recip_degrees+": get_recip_degreesp,
            "recip_degrees-": get_recip_degreesm,
            "total_degrees+": get_total_degreesp,
            "total_degrees-": get_total_degreesm,
            "total_recip+": get_total_recipp,
            "total_recip-": get_total_recipm,
            "has_self_edges": has_self_edges,
        }
    )
    _cache_aliases = {
        "row_degrees+": "count_rowwise+",
        "column_degrees+": "count_columnwise+",
        "row_degrees-": "count_rowwise-",
        "column_degrees-": "count_columnwise-",
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
        return DiGraph

    def to_undirected_class(self):
        return ga.Graph

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
        return True


class MultiDiGraph(DiGraph):
    def is_multigraph(self):
        return True


__all__ = ["DiGraph", "MultiDiGraph"]
