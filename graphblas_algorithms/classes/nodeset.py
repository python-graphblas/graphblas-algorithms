from collections.abc import MutableSet

from graphblas.semiring import lor_pair, plus_pair

from . import _utils


class NodeSet(MutableSet):
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

    # Requirements for MutableSet
    def __contains__(self, x):
        idx = self._key_to_id[x]
        return idx in self.vector

    def __iter__(self):
        # Slow if we iterate over one; fast if we iterate over all
        return map(
            self.id_to_key.__getitem__, self.vector.to_values(values=False, sort=False)[0].tolist()
        )

    def __len__(self):
        return self.vector.nvals

    def add(self, value):
        idx = self._key_to_id[value]
        self.vector[idx] = True

    def discard(self, value):
        idx = self._key_to_id[value]
        del self.vector[idx]

    # Override other MutableSet methods
    def __eq__(self, other):
        if isinstance(other, NodeSet):
            a = self.vector
            b = other.vector
            return (
                a.size == b.size
                and (nvals := a.nvals) == b.nvals
                and plus_pair(a @ b).get(0) == nvals
                and self._key_to_id == other._key_to_id
            )
        return super().__eq__(other)

    # __and__
    # __or__
    # __sub__
    # __xor__

    def clear(self):
        self.vector.clear()

    def isdisjoin(self, other):
        if isinstance(other, NodeSet):
            return not lor_pair(self.vector @ other.vector)
        return super().isdisjoint(other)

    def pop(self):
        try:
            idx = next(self.vector.ss.iterkeys())
        except StopIteration:
            raise KeyError from None
        del self.vector[idx]
        return self.id_to_key[idx]

    def remove(self, value):
        idx = self._key_to_id[value]
        if idx not in self.vector:
            raise KeyError(value)
        del self.vector[idx]

    def _from_iterable(self, it):
        # The elements in the iterable must be contained within key_to_id
        rv = object.__new__(type(self))
        rv._key_to_id = self._key_to_id
        rv._id_to_key = self._id_to_key
        rv.vector = rv.set_to_vector(it, size=self.vector.size)
        return rv

    # Add more set methods (as needed)
    def union(self, *args):
        return set(self).union(*args)  # TODO: can we make this better?
