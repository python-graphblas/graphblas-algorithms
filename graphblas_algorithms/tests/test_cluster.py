import inspect

import graphblas as gb
import networkx as nx

import graphblas_algorithms as ga
from graphblas_algorithms import transitivity, triangles

nx_triangles = nx.triangles
nx.triangles = triangles
nx.algorithms.triangles = triangles
nx.algorithms.cluster.triangles = triangles

nx_transitivity = nx.transitivity
nx.transitivity = transitivity
nx.algorithms.transitivity = transitivity
nx.algorithms.cluster.transitivity = transitivity


def test_signatures():
    nx_sig = inspect.signature(nx_triangles)
    sig = inspect.signature(triangles)
    assert nx_sig == sig
    nx_sig = inspect.signature(nx_transitivity)
    sig = inspect.signature(transitivity)
    assert nx_sig == sig


def test_triangles_full():
    # Including self-edges!
    G = gb.Matrix(bool, 5, 5)
    G[:, :] = True
    G2 = gb.select.offdiag(G).new()
    L = gb.select.tril(G, -1).new(name="L")
    U = gb.select.triu(G, 1).new(name="U")
    result = ga.cluster.triangles_core(G, L=L, U=U)
    expected = gb.Vector(int, 5)
    expected[:] = 6
    assert result.isequal(expected)
    result = ga.cluster.triangles_core(G2, L=L, U=U)
    assert result.isequal(expected)
    mask = gb.Vector(bool, 5)
    mask[0] = True
    mask[3] = True
    result = ga.cluster.triangles_core(G, mask=mask.S)
    expected = gb.Vector(int, 5)
    expected[0] = 6
    expected[3] = 6
    assert result.isequal(expected)
    result = ga.cluster.triangles_core(G2, mask=mask.S)
    assert result.isequal(expected)
    assert ga.cluster.single_triangle_core(G, 1) == 6
    assert ga.cluster.single_triangle_core(G, 0, L=L) == 6
    assert ga.cluster.single_triangle_core(G2, 0, has_self_edges=False) == 6
    assert ga.cluster.total_triangles_core(G2) == 10
    assert ga.cluster.total_triangles_core(G) == 10
    assert ga.cluster.total_triangles_core(G, L=L, U=U) == 10
    assert ga.cluster.transitivity_core(G) == 1.0
    assert ga.cluster.transitivity_core(G2, has_self_edges=False) == 1.0


from networkx.algorithms.tests.test_cluster import *  # noqa isort:skip
