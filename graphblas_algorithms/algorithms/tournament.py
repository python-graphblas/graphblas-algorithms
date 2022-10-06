import numpy as np
from graphblas import select

__all__ = ["is_tournament", "score_sequence", "tournament_matrix"]


def is_tournament(G):
    A = G._A
    if A.nvals != A.nrows * (A.ncols - 1) // 2 or G.get_property("has_self_edges"):
        return False
    # Alternative: do `select.triu(A.T).new(mask=A.S)` and don't check "has_self_edges"
    val = select.triu(A.T, 1).new(mask=A.S)
    return val.nvals == 0


def score_sequence(G):
    degrees = G.get_property("row_degrees+")
    _, values = degrees.to_values(indices=False, sort=False)
    values.sort()
    if degrees.nvals != degrees.size:
        values = np.pad(values, (degrees.size - degrees.nvals, 0))
    return values


def tournament_matrix(G):
    A = G._A
    return (A - A.T).new(name="tournament_matrix")
