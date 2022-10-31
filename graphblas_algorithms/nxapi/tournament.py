from graphblas import io

from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_directed_graph
from graphblas_algorithms.utils import not_implemented_for

from .simple_paths import is_simple_path as is_path  # noqa

__all__ = ["is_tournament", "score_sequence", "tournament_matrix"]


@not_implemented_for("undirected")
@not_implemented_for("multigraph")
def is_tournament(G):
    G = to_directed_graph(G)
    return algorithms.is_tournament(G)


@not_implemented_for("undirected")
@not_implemented_for("multigraph")
def score_sequence(G):
    G = to_directed_graph(G)
    # TODO: can we return a different, more native object?
    return algorithms.score_sequence(G).tolist()


@not_implemented_for("undirected")
@not_implemented_for("multigraph")
def tournament_matrix(G):
    G = to_directed_graph(G)
    T = algorithms.tournament_matrix(G)
    # TODO: can we return a different, more native object?
    return io.to_scipy_sparse(T)
