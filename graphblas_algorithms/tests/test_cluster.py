import inspect

import graphblas as gb
import networkx as nx

import graphblas_algorithms as ga
from graphblas_algorithms import average_clustering, clustering, transitivity, triangles

nx_triangles = nx.triangles
nx.triangles = triangles
nx.algorithms.triangles = triangles
nx.algorithms.cluster.triangles = triangles

nx_transitivity = nx.transitivity
nx.transitivity = transitivity
nx.algorithms.transitivity = transitivity
nx.algorithms.cluster.transitivity = transitivity

nx_clustering = nx.clustering
nx.clustering = clustering
nx.algorithms.clustering = clustering
nx.algorithms.cluster.clustering = clustering

nx_average_clustering = nx.average_clustering
nx.average_clustering = average_clustering
nx.algorithms.average_clustering = average_clustering
nx.algorithms.cluster.average_clustering = average_clustering


def test_signatures():
    nx_sig = inspect.signature(nx_triangles)
    sig = inspect.signature(triangles)
    assert nx_sig == sig
    nx_sig = inspect.signature(nx_transitivity)
    sig = inspect.signature(transitivity)
    assert nx_sig == sig
    nx_sig = inspect.signature(nx_clustering)
    sig = inspect.signature(clustering)
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
    assert ga.cluster.transitivity_core(G2) == 1.0
    result = ga.cluster.clustering_core(G)
    expected = gb.Vector(float, 5)
    expected[:] = 1
    assert result.isequal(expected)
    result = ga.cluster.clustering_core(G2)
    assert result.isequal(expected)
    assert ga.cluster.single_clustering_core(G, 0) == 1
    assert ga.cluster.single_clustering_core(G2, 0) == 1
    expected(mask.S, replace=True) << 1
    result = ga.cluster.clustering_core(G, mask=mask.S)
    assert result.isequal(expected)
    result = ga.cluster.clustering_core(G2, mask=mask.S)
    assert result.isequal(expected)
    assert ga.cluster.average_clustering_core(G) == 1
    assert ga.cluster.average_clustering_core(G2) == 1
    assert ga.cluster.average_clustering_core(G, mask=mask.S) == 1
    assert ga.cluster.average_clustering_core(G2, mask=mask.S) == 1


from networkx.algorithms.tests.test_cluster import *  # noqa isort:skip
