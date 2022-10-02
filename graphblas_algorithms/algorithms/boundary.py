import itertools

from graphblas import binary
from graphblas.semiring import any_pair

from graphblas_algorithms.classes.digraph import to_graph
from graphblas_algorithms.utils import get_all


def edge_boundary_core(G, nbunch1, nbunch2=None, *, is_weighted=False):
    if is_weighted:
        B = binary.second(nbunch1 & G._A).new(name="boundary")
    else:
        B = binary.pair(nbunch1 & G._A).new(name="boundary")
    if nbunch2 is None:
        # Default nbunch2 is the complement of nbunch1.
        # We get the row_degrees to better handle hypersparse data.
        nbunch2 = G.get_property("row_degrees+", mask=~nbunch1.S)
    if is_weighted:
        B << binary.first(B & nbunch2)
    else:
        B << binary.pair(B & nbunch2)
    return B


def edge_boundary(G, nbunch1, nbunch2=None, data=False, keys=False, default=None):
    # TODO: figure out data, keys, and default arguments and handle multigraph
    # data=True will be tested in test_mst.py
    is_multigraph = G.is_multigraph()
    # This may be wrong for multi-attributed graphs
    if data is True:
        weight = "weight"
    elif not data:
        weight = None
    else:
        weight = data
    G = to_graph(G, weight=weight)
    v1 = G.set_to_vector(nbunch1, ignore_extra=True)
    v2 = G.set_to_vector(nbunch2, ignore_extra=True)
    result = edge_boundary_core(G, v1, v2, is_weighted=is_multigraph or data)
    rows, cols, vals = result.to_values(values=is_multigraph or data)
    id_to_key = G.id_to_key
    if data:
        it = zip(
            (id_to_key[row] for row in rows),
            (id_to_key[col] for col in cols),
            # Unsure about this; data argument may mean *all* edge attributes
            ({weight: val} for val in vals),
        )
    else:
        it = zip(
            (id_to_key[row] for row in rows),
            (id_to_key[col] for col in cols),
        )
    if is_multigraph:
        # Edge weights indicate number of times to repeat edges
        it = itertools.chain.from_iterable(itertools.starmap(itertools.repeat, zip(it, vals)))
    return it


def node_boundary_core(G, nbunch1, *, mask=None):
    if mask is None:
        mask = ~nbunch1.S
    else:
        mask = mask & (~nbunch1.S)
    bdy = any_pair(G._A.T @ nbunch1).new(mask=mask, name="boundary")
    return bdy


def node_boundary(G, nbunch1, nbunch2=None):
    G = to_graph(G)
    v1 = G.set_to_vector(nbunch1, ignore_extra=True)
    if nbunch2 is not None:
        mask = G.set_to_vector(nbunch2, ignore_extra=True).S
    else:
        mask = None
    result = node_boundary_core(G, v1, mask=mask)
    return G.vector_to_set(result)


__all__ = get_all(__name__)
