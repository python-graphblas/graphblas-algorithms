from collections.abc import MutableMapping

from . import _utils


class NodeMap(MutableMapping):
    def __init__(self):
        raise NotImplementedError()
        # .vector, ._key_to_id, ._id_to_key

    @classmethod
    def from_graphblas(cls, v, *, key_to_id=None):
        rv = object.__new__(cls)
        rv.vector = v
        if key_to_id is None:
            rv._key_to_id = {i: i for i in range(v.size)}
        else:
            rv._key_to_id = key_to_id
        rv._id_to_key = None
        return rv

    id_to_key = property(_utils.id_to_key)
    # get_property = _utils.get_property
    # get_properties = _utils.get_properties
    dict_to_vector = _utils.dict_to_vector
    list_to_vector = _utils.list_to_vector
    list_to_mask = _utils.list_to_mask
    list_to_ids = _utils.list_to_ids
    list_to_keys = _utils.list_to_keys
    matrix_to_dicts = _utils.matrix_to_dicts
    set_to_vector = _utils.set_to_vector
    # to_networkx = _utils.to_networkx
    vector_to_dict = _utils.vector_to_dict
    vector_to_nodemap = _utils.vector_to_nodemap
    vector_to_nodeset = _utils.vector_to_nodeset
    vector_to_set = _utils.vector_to_set
    # _cacheit = _utils._cacheit

    # Requirements for MutableMapping
    def __delitem__(self, key):
        idx = self._key_to_id[key]
        del self.vector[idx]

    def __getitem__(self, key):
        idx = self._key_to_id[key]
        if (rv := self.vector.get(idx)) is not None:
            return rv
        raise KeyError(key)

    def __iter__(self):
        # Slow if we iterate over one; fast if we iterate over all
        return map(
            self.id_to_key.__getitem__, self.vector.to_values(values=False, sort=False)[0].tolist()
        )

    def __len__(self):
        return self.vector.nvals

    def __setitem__(self, key, val):
        idx = self._key_to_id[key]
        self.vector[idx] = val

    # Override other MutableMapping methods
    def __contains__(self, key):
        idx = self._key_to_id[key]
        return idx in self.vector

    def __eq__(self, other):
        if isinstance(other, NodeMap):
            return self.vector.isequal(other.vector) and self._key_to_id == other._key_to_id
        return super().__eq__(other)

    def clear(self):
        self.vector.clear()

    def get(self, key, default=None):
        idx = self._key_to_id[key]
        return self.vector.get(idx, default)

    # items
    # keys
    # pop

    def popitem(self):
        v = self.vector
        try:
            idx, value = next(v.ss.iteritems())
        except StopIteration:
            raise KeyError from None
        del v[idx]
        return self.id_to_key[idx], value

    def setdefault(self, key, default=None):
        idx = self._key_to_id[key]
        if (value := self.vector.get(idx)) is not None:
            return value
        self.vector[idx] = default
        return default

    # update
    # values
