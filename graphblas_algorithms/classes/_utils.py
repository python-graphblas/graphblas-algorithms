import graphblas as gb
import numpy as np
from graphblas import Matrix, Vector, binary
from graphblas.core.matrix import TransposedMatrix
from graphblas.core.utils import ensure_type

################
# Classmethods #
################


def from_networkx(cls, G, weight=None, dtype=None):
    rv = cls()
    rv._key_to_id = {k: i for i, k in enumerate(G)}
    if rv._key_to_id:
        rv._A = gb.io.from_networkx(G, nodelist=rv._key_to_id, weight=weight, dtype=dtype)
    else:
        rv._A = Matrix(dtype if dtype is not None else float)
    return rv


def from_graphblas(cls, A, *, key_to_id=None):
    # Does not copy if A is a Matrix!
    A = ensure_type(A, Matrix)
    if A.nrows != A.ncols:
        raise ValueError(f"Adjacency matrix must be square; got {A.nrows} x {A.ncols}")
    rv = cls()
    # If there is no mapping, it may be nice to keep this as None
    if key_to_id is None:
        rv._key_to_id = {i: i for i in range(A.nrows)}
    else:
        rv._key_to_id = key_to_id
    rv._A = A
    return rv


##############
# Properties #
##############


def id_to_key(self):
    if self._id_to_key is None:
        self._id_to_key = {val: key for key, val in self._key_to_id.items()}
    return self._id_to_key


###########
# Methods #
###########


def get_property(self, name, *, mask=None):
    return self._get_property[self._cache_aliases.get(name, name)](self, mask)


def get_properties(self, names, *, mask=None):
    if isinstance(names, str):
        # Separated by commas and/or spaces
        names = [
            self._cache_aliases.get(name, name)
            for name in names.replace(" ", ",").split(",")
            if name
        ]
    else:
        names = [self._cache_aliases.get(name, name) for name in names]
    results = {
        name: self._get_property[name](self, mask)
        for name in sorted(names, key=self._property_priority.__getitem__)
    }
    return [results[name] for name in names]


def dict_to_vector(self, d, *, size=None, dtype=None, name=None):
    if d is None:
        return None
    if size is None:
        size = len(self)
    key_to_id = self._key_to_id
    indices, values = zip(*((key_to_id[key], val) for key, val in d.items()))
    return Vector.from_values(indices, values, size=size, dtype=dtype, name=name)


def list_to_vector(self, nodes, dtype=bool, *, size=None, name=None):
    if nodes is None:
        return None
    if size is None:
        size = len(self)
    key_to_id = self._key_to_id
    index = [key_to_id[key] for key in nodes]
    return Vector.from_values(index, True, size=size, dtype=dtype, name=name)


def list_to_mask(self, nodes, *, size=None, name="mask"):
    if nodes is None:
        return None
    return self.list_to_vector(nodes, size=size, name=name).S


def list_to_ids(self, nodes):
    if nodes is None:
        return None
    key_to_id = self._key_to_id
    return [key_to_id[key] for key in nodes]


def list_to_keys(self, indices):
    if indices is None:
        return None
    id_to_key = self.id_to_key
    return [id_to_key[idx] for idx in indices]


def set_to_vector(self, nodes, dtype=bool, *, ignore_extra=False, size=None, name=None):
    if nodes is None:
        return None
    if size is None:
        size = len(self)
    key_to_id = self._key_to_id
    if ignore_extra:
        if not isinstance(nodes, set):
            nodes = set(nodes)
        nodes = nodes & key_to_id.keys()
    index = [key_to_id[key] for key in nodes]
    return Vector.from_values(index, True, size=size, dtype=dtype, name=name)


def vector_to_dict(self, v, *, mask=None, fillvalue=None):
    if mask is not None:
        if fillvalue is not None and v.nvals < mask.parent.nvals:
            v(mask, binary.first) << fillvalue
    elif fillvalue is not None and v.nvals < v.size:
        v(mask=~v.S) << fillvalue
    id_to_key = self.id_to_key
    return {id_to_key[index]: value for index, value in zip(*v.to_values(sort=False))}


def vector_to_nodemap(self, v, *, mask=None, fillvalue=None):
    from .nodemap import NodeMap

    if mask is not None:
        if fillvalue is not None and v.nvals < mask.parent.nvals:
            v(mask, binary.first) << fillvalue
    elif fillvalue is not None and v.nvals < v.size:
        v(mask=~v.S) << fillvalue

    rv = object.__new__(NodeMap)
    rv.vector = v
    rv._key_to_id = self._key_to_id
    rv._id_to_key = self._id_to_key
    return rv
    # return NodeMap.from_graphblas(v, key_to_id=self._key_to_id)


def vector_to_nodeset(self, v):
    from .nodeset import NodeSet

    rv = object.__new__(NodeSet)
    rv.vector = v
    rv._key_to_id = self._key_to_id
    rv._id_to_key = self._id_to_key
    return rv
    # return NodeSet.from_graphblas(v, key_to_id=self._key_to_id)


def vector_to_set(self, v):
    id_to_key = self.id_to_key
    indices, _ = v.to_values(values=False, sort=False)
    return {id_to_key[index] for index in indices}


def matrix_to_dicts(self, A, *, use_row_index=False, use_column_index=False):
    """Convert a Matrix to a dict of dicts of the form ``{row: {col: val}}``

    Use ``use_row_index=True`` to return the row index as keys in the dict,
    and likewise for `use_column_index=True``.

    """
    if isinstance(A, TransposedMatrix):
        # Not covered
        d = A.T.ss.export("hypercsc")
        rows = d["cols"].tolist()
        col_indices = d["row_indices"].tolist()
        use_row_index, use_column_index = use_column_index, use_row_index
    else:
        d = A.ss.export("hypercsr")
        rows = d["rows"].tolist()
        col_indices = d["col_indices"].tolist()
    indptr = d["indptr"]
    values = d["values"].tolist()
    id_to_key = self.id_to_key
    it = zip(rows, np.lib.stride_tricks.sliding_window_view(indptr, 2).tolist())
    if use_row_index and use_column_index:
        return {
            row: dict(zip(col_indices[start:stop], values[start:stop])) for row, (start, stop) in it
        }
    elif use_row_index:
        return {
            row: {
                id_to_key[col]: val for col, val in zip(col_indices[start:stop], values[start:stop])
            }
            for row, (start, stop) in it
        }
    elif use_column_index:
        return {
            id_to_key[row]: dict(zip(col_indices[start:stop], values[start:stop]))
            for row, (start, stop) in it
        }
    else:
        return {
            id_to_key[row]: {
                id_to_key[col]: val for col, val in zip(col_indices[start:stop], values[start:stop])
            }
            for row, (start, stop) in it
        }


def to_networkx(self, edge_attribute="weight"):
    import networkx as nx

    # Not covered yet, but will probably be useful soon
    if self.is_directed():
        G = nx.DiGraph()
        A = self._A
    else:
        G = nx.Graph()
        A = self.get_property("L+")
    G.add_nodes_from(self._key_to_id)
    id_to_key = self.id_to_key
    rows, cols, vals = A.to_values()
    rows = (id_to_key[row] for row in rows.tolist())
    cols = (id_to_key[col] for col in cols.tolist())
    if edge_attribute is None:
        G.add_edges_from(zip(rows, cols))
    else:
        G.add_weighted_edges_from(zip(rows, cols, vals), weight=edge_attribute)
    # What else should we copy over?
    return G


def _cacheit(self, key, func, *args, **kwargs):
    if key not in self._cache:
        self._cache[key] = func(*args, **kwargs)
    return self._cache[key]
