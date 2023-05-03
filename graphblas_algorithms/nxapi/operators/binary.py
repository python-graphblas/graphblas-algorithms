from graphblas_algorithms import algorithms
from graphblas_algorithms.classes.digraph import to_graph

from ..exception import NetworkXError

__all__ = [
    "compose",
    "difference",
    "disjoint_union",
    "full_join",
    "intersection",
    "symmetric_difference",
    "union",
]


def union(G, H, rename=()):
    G = to_graph(G)
    H = to_graph(H)
    try:
        return algorithms.union(G, H, rename=rename)
    except algorithms.exceptions.GraphBlasAlgorithmException as e:
        raise NetworkXError(*e.args) from e


def disjoint_union(G, H):
    G = to_graph(G)
    H = to_graph(H)
    try:
        return algorithms.disjoint_union(G, H)
    except algorithms.exceptions.GraphBlasAlgorithmException as e:
        raise NetworkXError(*e.args) from e


def intersection(G, H):
    G = to_graph(G)
    H = to_graph(H)
    try:
        return algorithms.intersection(G, H)
    except algorithms.exceptions.GraphBlasAlgorithmException as e:
        raise NetworkXError(*e.args) from e


def difference(G, H):
    G = to_graph(G)
    H = to_graph(H)
    try:
        return algorithms.difference(G, H)
    except algorithms.exceptions.GraphBlasAlgorithmException as e:
        raise NetworkXError(*e.args) from e


def symmetric_difference(G, H):
    G = to_graph(G)
    H = to_graph(H)
    try:
        return algorithms.symmetric_difference(G, H)
    except algorithms.exceptions.GraphBlasAlgorithmException as e:
        raise NetworkXError(*e.args) from e


def compose(G, H):
    G = to_graph(G)
    H = to_graph(H)
    try:
        return algorithms.compose(G, H)
    except algorithms.exceptions.GraphBlasAlgorithmException as e:
        raise NetworkXError(*e.args) from e


def full_join(G, H, rename=()):
    G = to_graph(G)
    H = to_graph(H)
    try:
        return algorithms.full_join(G, H, rename=rename)
    except algorithms.exceptions.GraphBlasAlgorithmException as e:
        raise NetworkXError(*e.args) from e
