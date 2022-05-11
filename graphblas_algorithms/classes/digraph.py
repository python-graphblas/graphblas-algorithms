import networkx as nx
from graphblas import Matrix, Vector, agg, binary, select, unary

import graphblas_algorithms as ga

from . import _utils


def get_AT(A, cache, mask=None):
    if "AT" not in cache:
        cache["AT"] = A.T.new()
    return cache["AT"]


def get_offdiag(A, cache, mask=None):
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


def get_Up(A, cache, mask=None):
    if "U+" not in cache:
        if "U-" in cache and not has_self_edges(A, cache):
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


def get_Lp(A, cache, mask=None):
    if "L+" not in cache:
        if "L-" in cache and not has_self_edges(A, cache):
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


def get_Um(A, cache, mask=None):
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


def get_Lm(A, cache, mask=None):
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
    if mask is not None:
        return cache["diag"].dup(mask=mask)
    return cache["diag"]


def get_row_degreesp(A, cache, mask=None):
    if mask is not None:
        if "row_degrees+" not in cache:
            return A.reduce_rowwise(agg.count).new(mask=mask, name="row_degrees+")
        else:
            return cache["row_degrees+"].dup(mask=mask)
    if "row_degrees+" not in cache:
        cache["row_degrees+"] = A.reduce_rowwise(agg.count).new(name="row_degrees+")
    if cache.get("has_self_edges") is False:
        cache["row_degrees-"] = cache["row_degrees+"]
    return cache["row_degrees+"]


def get_row_degreesm(A, cache, mask=None):
    if mask is not None:
        if "row_degrees-" in cache:
            return cache["row_degrees-"].dup(mask=mask)
        elif "offdiag" in cache:
            return cache["offdiag"].reduce_rowwise(agg.count).new(mask=mask, name="row_degrees-")
        elif "L-" in cache and "U-" in cache and has_self_edges(A, cache):
            return (
                cache["L-"].reduce_rowwise(agg.count).new(mask=mask)
                + cache["U-"].reduce_rowwise(agg.count).new(mask=mask)
            ).new(name="row_degrees-")
        else:
            return get_row_degreesm(A, cache).dup(mask=mask)
    if "row_degrees-" not in cache:
        if "offdiag" in cache:
            cache["row_degrees-"] = (
                cache["offdiag"].reduce_rowwise(agg.count).new(name="row_degrees")
            )
        elif "L-" in cache and "U-" in cache:
            cache["row_degrees"] = (
                cache["L-"].reduce_rowwise(agg.count) + cache["U-"].reduce_rowwise(agg.count)
            ).new(name="row_degrees-")
        if has_self_edges(A, cache):
            if "L-" in cache and "U-" in cache:
                cache["row_degrees-"] = (
                    cache["L-"].reduce_rowwise(agg.count) + cache["U-"].reduce_rowwise(agg.count)
                ).new(name="row_degrees-")
            else:
                # Is there a better way to do this?
                degrees = get_row_degreesp(A, cache).dup()
                diag = get_diag(A, cache)
                degrees(binary.plus, diag.S) << -1
                degrees(degrees.V, replace=True) << degrees  # drop 0s
                cache["row_degrees-"] = degrees
        else:
            cache["row_degrees-"] = get_row_degreesp(A, cache)
    if cache.get("has_self_edges") is False:
        cache["row_degrees+"] = cache["row_degrees-"]
    return cache["row_degrees-"]


def get_column_degreesp(A, cache, mask=None):
    if mask is not None:
        if "column_degrees+" not in cache:
            if "AT" in cache:
                return cache["AT"].reduce_rowwise(agg.count).new(mask=mask, name="column_degrees+")
            else:
                return A.reduce_columnwise(agg.count).new(mask=mask, name="column_degrees+")
        else:
            return cache["column_degrees+"].dup(mask=mask)
    if "column_degrees+" not in cache:
        if "AT" in cache:
            cache["column_degrees+"] = (
                cache["AT"].reduce_rowwise(agg.count).new(name="column_degrees+")
            )
        else:
            cache["column_degrees+"] = A.reduce_columnwise(agg.count).new(name="column_degrees+")
    if cache.get("has_self_edges") is False:
        cache["column_degrees-"] = cache["column_degrees+"]
    return cache["column_degrees+"]


def get_column_degreesm(A, cache, mask=None):
    if mask is not None:
        if "column_degrees-" in cache:
            return cache["column_degrees-"].dup(mask=mask)
        elif "offdiag" in cache:
            return (
                cache["offdiag"].reduce_columnwise(agg.count).new(mask=mask, name="column_degrees-")
            )
        elif "L-" in cache and "U-" in cache and has_self_edges(A, cache):
            return (
                cache["L-"].reduce_columnwise(agg.count).new(mask=mask)
                + cache["U-"].reduce_columnwise(agg.count).new(mask=mask)
            ).new(name="column_degrees-")
        else:
            return get_column_degreesm(A, cache).dup(mask=mask)
    if "column_degrees-" not in cache:
        if "offdiag" in cache:
            cache["column_degrees-"] = (
                cache["offdiag"].reduce_columnwise(agg.count).new(name="column_degrees")
            )
        elif "L-" in cache and "U-" in cache:
            cache["column_degrees"] = (
                cache["L-"].reduce_columnwise(agg.count) + cache["U-"].reduce_columnwise(agg.count)
            ).new(name="column_degrees-")
        if has_self_edges(A, cache):
            if "L-" in cache and "U-" in cache:
                cache["column_degrees-"] = (
                    cache["L-"].reduce_columnwise(agg.count)
                    + cache["U-"].reduce_columnwise(agg.count)
                ).new(name="column_degrees-")
            else:
                # Is there a better way to do this?
                degrees = get_column_degreesp(A, cache).dup()
                diag = get_diag(A, cache)
                degrees(binary.plus, diag.S) << -1
                degrees(degrees.V, replace=True) << degrees  # drop 0s
                cache["column_degrees-"] = degrees
        else:
            cache["column_degrees-"] = get_column_degreesp(A, cache)
    if cache.get("has_self_edges") is False:
        cache["column_degrees+"] = cache["column_degrees-"]
    return cache["column_degrees-"]


def get_recip_degreesp(A, cache, mask=None):
    if "AT" in cache:
        AT = cache["AT"]
    else:
        AT = A.T
    if mask is not None:
        if "recip_degrees+" in cache:
            return cache["recip_degrees+"].dup(mask=mask)
        elif "recip_degrees-" in cache and "diag" in cache:
            return (unary.one(cache["diag"]) + cache["recip_degrees-"]).new(
                mask=mask, name="recip_degrees+"
            )
        elif "recip_degrees-" in cache and not has_self_edges(A, cache):
            return cache["recip_degrees-"].dup(mask=mask)
        else:
            return binary.pair(A & AT).reduce_rowwise().new(mask=mask, name="recip_degrees+")
    if "recip_degrees+" not in cache:
        if "recip_degrees-" in cache and "diag" in cache:
            cache["recip_degrees+"] = (unary.one(cache["diag"]) + cache["recip_degrees-"]).new(
                name="recip_degrees+"
            )
        elif "recip_degrees-" in cache and not has_self_edges(A, cache):
            cache["recip_degrees+"] = cache["recip_degrees-"]
        else:
            cache["recip_degrees+"] = (
                binary.pair(A & AT).reduce_rowwise().new(name="recip_degrees+")
            )
    if cache.get("has_self_edges") is False:
        cache["recip_degrees-"] = cache["recip_degrees+"]
    return cache["recip_degrees+"]


def get_recip_degreesm(A, cache, mask=None):
    if "AT" in cache:
        AT = cache["AT"]
    elif "offdiag" in cache:
        AT = cache["offdiag"].T
    else:
        AT = A.T
    if mask is not None:
        if "recip_degrees-" in cache:
            return cache["recip_degrees-"].dup(mask=mask)
        elif "recip_degrees+" in cache and "diag" in cache:
            rv = binary.minus(
                cache["recip_degrees+"] | unary.one(cache["diag"]), require_monoid=False
            ).new(mask=mask, name="recip_degrees-")
            rv(rv.V, replace=True) << rv  # drop 0s
            return rv
        elif not has_self_edges(A, cache):
            return get_recip_degreesp(A, cache, mask=mask)
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
            diag = get_diag(A, cache, mask=mask)
            overlap = binary.pair(A & AT).reduce_rowwise().new(mask=mask)
            rv = binary.minus(overlap & unary.one(diag), require_monoid=False).new(
                name="recip_degrees-"
            )
            rv(rv.V, replace=True) << rv  # drop 0s
            return rv
    if "recip_degrees-" not in cache:
        if "recip_degrees+" in cache and "diag" in cache:
            rv = binary.minus(
                cache["recip_degrees+"] | unary.one(cache["diag"]), require_monoid=False
            ).new(name="recip_degrees-")
            rv(rv.V, replace=True) << rv  # drop 0s
            cache["recip_degrees-"] = rv
        elif not has_self_edges(A, cache):
            cache["recip_degrees-"] = get_recip_degreesp(A, cache)
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
            diag = get_diag(A, cache)
            overlap = binary.pair(A & AT).reduce_rowwise().new()
            rv = binary.minus(overlap & unary.one(diag), require_monoid=False).new(
                name="recip_degrees-"
            )
            rv(rv.V, replace=True) << rv  # drop 0s
            cache["recip_degrees-"] = rv
    if cache.get("has_self_edges") is False:
        cache["recip_degrees+"] = cache["recip_degrees-"]
    return cache["recip_degrees-"]


def get_total_degreesp(A, cache, mask=None):
    if mask is not None:
        if "total_degrees+" in cache:
            return cache["total_degrees+"].dup(mask=mask)
        else:
            return (get_row_degreesp(A, cache, mask) + get_column_degreesp(A, cache, mask)).new(
                name="total_degrees+"
            )
    if "total_degrees+" not in cache:
        cache["total_degrees+"] = (get_row_degreesp(A, cache) + get_column_degreesp(A, cache)).new(
            name="total_degrees+"
        )
    if cache.get("has_self_edges") is False:
        cache["total_degrees-"] = cache["total_degrees+"]
    return cache["total_degrees+"]


def get_total_degreesm(A, cache, mask=None):
    if mask is not None:
        if "total_degrees-" in cache:
            return cache["total_degrees-"].dup(mask=mask)
        else:
            return (get_row_degreesm(A, cache, mask) + get_column_degreesm(A, cache, mask)).new(
                name="total_degrees-"
            )
    if "total_degrees-" not in cache:
        cache["total_degrees-"] = (get_row_degreesm(A, cache) + get_column_degreesm(A, cache)).new(
            name="total_degrees-"
        )
    if cache.get("has_self_edges") is False:
        cache["total_degrees+"] = cache["total_degrees-"]
    return cache["total_degrees-"]


def get_total_recipp(A, cache, mask=None):
    if "total_recip+" not in cache:
        if "total_recip-" in cache and cache.get("has_self_edges") is False:
            cache["total_recip+"] = cache["total_recip-"]
        elif "recip_degrees+" in cache:
            cache["total_recip+"] = cache["recip_degrees+"].reduce(allow_empty=False).value
        else:
            if "AT" in cache:
                AT = cache["AT"]
            else:
                AT = A.T
            cache["total_recip+"] = binary.pair(A & AT).reduce_scalar(allow_empty=False).value
    if "has_self_edges" not in cache and "total_recip-" in cache:
        cache["has_self_edges"] = cache["total_recip+"] > cache["total_recip-"]
    if cache.get("has_self_edges") is False:
        cache["total_recip-"] = cache["total_recip+"]
    return cache["total_recip+"]


def get_total_recipm(A, cache, mask=None):
    if "total_recip-" not in cache:
        if "total_recip+" in cache and cache.get("has_self_edges") is False:
            cache["total_recip-"] = cache["total_recip+"]
        else:
            cache["total_recip-"] = get_recip_degreesm(A, cache).reduce(allow_empty=False).value
    if "has_self_edges" not in cache and "total_recip+" in cache:
        cache["has_self_edges"] = cache["total_recip+"] > cache["total_recip-"]
    if cache.get("has_self_edges") is False:
        cache["total_recip+"] = cache["total_recip-"]
    return cache["total_recip-"]


def has_self_edges(A, cache, mask=None):
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
            cache["has_self_edges"] = (
                cache["row_degrees-"].reduce(allow_empty=False).value < A.nvals
            )
        elif "column_degrees-" in cache:
            cache["has_self_edges"] = (
                cache["column_degrees-"].reduce(allow_empty=False).value < A.nvals
            )
        elif "total_degrees-" in cache:
            cache["has_self_edges"] = (
                cache["total_degrees-"].reduce(allow_empty=False).value < 2 * A.nvals
            )
        elif "total_degrees+" in cache:
            cache["has_self_edges"] = (
                cache["total_degrees+"].reduce(allow_empty=False).value > 2 * A.nvals
            )
        else:
            get_diag(A, cache)
    return cache["has_self_edges"]


def to_directed_graph(G, weight=None, dtype=None):
    # We should do some sanity checks here to ensure we're returning a valid directed graph
    if isinstance(G, DiGraph):
        return G
    elif isinstance(G, nx.DiGraph):
        return DiGraph.from_networkx(G, weight=weight, dtype=dtype)
    elif isinstance(G, Matrix):
        return DiGraph.from_graphblas(G)
    else:
        raise TypeError()


def to_graph(G, weight=None, dtype=None):
    if isinstance(G, (DiGraph, ga.Graph)):
        return G
    elif isinstance(G, nx.DiGraph):
        return DiGraph.from_networkx(G, weight=weight, dtype=dtype)
    elif isinstance(G, nx.Graph):
        return ga.Graph.from_networkx(G, weight=weight, dtype=dtype)
    elif isinstance(G, Matrix):
        # Should we check if it can be undirected?
        return DiGraph.from_graphblas(G)
    else:
        raise TypeError()


class DiGraph:
    # "-" properties ignore self-edges, "+" properties include self-edges
    _property_priority = {
        key: i
        for i, key in enumerate(
            [
                "AT",
                "offdiag",
                "U+",
                "L+",
                "U-",
                "L-",
                "diag",
                "row_degrees+",
                "column_degrees+",
                "row_degrees-",
                "column_degrees-",
                "recip_degrees+",
                "recip_degrees-",
                "total_degrees+",
                "total_degrees-",
                "total_recip+",
                "total_recip-",
                "has_self_edges",
            ]
        )
    }
    _get_property = {
        "AT": get_AT,
        "offdiag": get_offdiag,
        "U+": get_Up,
        "L+": get_Lp,
        "U-": get_Um,
        "L-": get_Lm,
        "diag": get_diag,
        "row_degrees+": get_row_degreesp,
        "column_degrees+": get_column_degreesp,
        "row_degrees-": get_row_degreesm,
        "column_degrees-": get_column_degreesm,
        "recip_degrees+": get_recip_degreesp,
        "recip_degrees-": get_recip_degreesm,
        "total_degrees+": get_total_degreesp,
        "total_degrees-": get_total_degreesm,
        "total_recip+": get_total_recipp,
        "total_recip-": get_total_recipm,
        "has_self_edges": has_self_edges,
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
