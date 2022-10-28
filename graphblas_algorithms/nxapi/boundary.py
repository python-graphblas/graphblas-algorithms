import itertools

from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph

__all__ = ["edge_boundary", "node_boundary"]


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
    result = algorithms.edge_boundary(G, v1, v2, is_weighted=is_multigraph or data)
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


def node_boundary(G, nbunch1, nbunch2=None):
    G = to_graph(G)
    v1 = G.set_to_vector(nbunch1, ignore_extra=True)
    if nbunch2 is not None:
        mask = G.set_to_vector(nbunch2, ignore_extra=True).S
    else:
        mask = None
    result = algorithms.node_boundary(G, v1, mask=mask)
    return G.vector_to_nodeset(result)
