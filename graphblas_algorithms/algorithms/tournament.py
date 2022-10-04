import numpy as np
from graphblas import io, select

from graphblas_algorithms.classes.digraph import to_directed_graph
from graphblas_algorithms.utils import get_all, not_implemented_for


def is_tournament_core(G):
    A = G._A
    if A.nvals != A.nrows * (A.ncols - 1) // 2 or G.get_property("has_self_edges"):
        return False
    # Alternative: do `select.triu(A.T).new(mask=A.S)` and don't check "has_self_edges"
    val = select.triu(A.T, 1).new(mask=A.S)
    return val.nvals == 0


@not_implemented_for("undirected")
@not_implemented_for("multigraph")
def is_tournament(G):
    G = to_directed_graph(G)
    return is_tournament_core(G)


def score_sequence_core(G):
    degrees = G.get_property("row_degrees+")
    _, values = degrees.to_values(indices=False, sort=False)
    values.sort()
    if degrees.nvals != degrees.size:
        values = np.pad(values, (degrees.size - degrees.nvals, 0))
    return values


@not_implemented_for("undirected")
@not_implemented_for("multigraph")
def score_sequence(G):
    G = to_directed_graph(G)
    return score_sequence_core(G).tolist()


def tournament_matrix_core(G):
    A = G._A
    return (A - A.T).new(name="tournament_matrix")


@not_implemented_for("undirected")
@not_implemented_for("multigraph")
def tournament_matrix(G):
    G = to_directed_graph(G)
    T = tournament_matrix_core(G)
    return io.to_scipy_sparse(T)


__all__ = get_all(__name__)
