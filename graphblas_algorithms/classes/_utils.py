import graphblas as gb
import numpy as np
from graphblas import Matrix, Vector, binary
from graphblas.matrix import TransposedMatrix

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


def from_graphblas(cls, A):
    # Does not copy!
    if A.nrows != A.ncols:
        raise ValueError(f"Adjacency matrix must be square; got {A.nrows} x {A.ncols}")
    rv = cls()
    rv._key_to_id = {i: i for i in range(A.nrows)}
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
    indices, values = zip(*((self._key_to_id[key], val) for key, val in d.items()))
    return Vector.from_values(indices, values, size=size, dtype=dtype, name=name)


def list_to_vector(self, nodes, dtype=bool, *, size=None, name=None):
    if nodes is None:
        return None
    if size is None:
        size = len(self)
    index = [self._key_to_id[key] for key in nodes]
    return Vector.from_values(index, True, size=size, dtype=dtype, name=name)


def list_to_mask(self, nodes, *, size=None, name="mask"):
    if nodes is None:
        return None
    return self.list_to_vector(nodes, size=size, name=name).S


def list_to_ids(self, nodes):
    if nodes is None:
        return None
    return [self._key_to_id[key] for key in nodes]


def vector_to_dict(self, v, *, mask=None, fillvalue=None):
    if mask is not None:
        if fillvalue is not None and v.nvals < mask.parent.nvals:
            v(mask, binary.first) << fillvalue
    elif fillvalue is not None and v.nvals < v.size:
        v(mask=~v.S) << fillvalue
    id_to_key = self.id_to_key
    return {id_to_key[index]: value for index, value in zip(*v.to_values(sort=False))}


def matrix_to_dicts(self, A):
    """{row: {col: val}}"""
    if isinstance(A, TransposedMatrix):
        # Not covered
        d = A.T.ss.export("hypercsc")
        rows = d["cols"].tolist()
        col_indices = d["row_indices"].tolist()
    else:
        d = A.ss.export("hypercsr")
        rows = d["rows"].tolist()
        col_indices = d["col_indices"].tolist()
    indptr = d["indptr"]
    values = d["values"].tolist()
    id_to_key = self.id_to_key
    return {
        id_to_key[row]: {
            id_to_key[col]: val for col, val in zip(col_indices[start:stop], values[start:stop])
        }
        for row, (start, stop) in zip(
            rows, np.lib.stride_tricks.sliding_window_view(indptr, 2).tolist()
        )
    }


def _cacheit(self, key, func, *args, **kwargs):
    if key not in self._cache:
        self._cache[key] = func(*args, **kwargs)
    return self._cache[key]
