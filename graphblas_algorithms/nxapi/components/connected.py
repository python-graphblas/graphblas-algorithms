from graphblas_algorithms import algorithms
from graphblas_algorithms.algorithms.exceptions import PointlessConcept
from graphblas_algorithms.classes.graph import to_undirected_graph
from graphblas_algorithms.utils import not_implemented_for

from ..exception import NetworkXPointlessConcept

__all__ = [
    "is_connected",
    "node_connected_component",
]


@not_implemented_for("directed")
def is_connected(G):
    G = to_undirected_graph(G)
    try:
        return algorithms.is_connected(G)
    except PointlessConcept as e:
        raise NetworkXPointlessConcept(*e.args) from e


@not_implemented_for("directed")
def node_connected_component(G, n):
    G = to_undirected_graph(G)
    rv = algorithms.node_connected_component(G, n)
    return G.vector_to_nodeset(rv)
