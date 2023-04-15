from graphblas_algorithms import algorithms
from graphblas_algorithms.algorithms.exceptions import PointlessConcept
from graphblas_algorithms.classes.digraph import to_directed_graph
from graphblas_algorithms.utils import not_implemented_for

from ..exception import NetworkXPointlessConcept

__all__ = [
    "is_weakly_connected",
]


@not_implemented_for("undirected")
def is_weakly_connected(G):
    G = to_directed_graph(G)
    try:
        return algorithms.is_weakly_connected(G)
    except PointlessConcept as e:
        raise NetworkXPointlessConcept(*e.args) from e
