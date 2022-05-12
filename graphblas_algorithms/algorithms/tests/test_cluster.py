import graphblas as gb
import networkx as nx

from graphblas_algorithms import (
    DiGraph,
    Graph,
    average_clustering,
    clustering,
    transitivity,
    triangles,
)
from graphblas_algorithms.algorithms import cluster


def test_triangles_full():
    # Including self-edges!
    G = gb.Matrix(bool, 5, 5)
    G[:, :] = True
    G2 = gb.select.offdiag(G).new()
    G = Graph.from_graphblas(G)
    G2 = Graph.from_graphblas(G2)
    result = cluster.triangles_core(G)
    expected = gb.Vector(int, 5)
    expected[:] = 6
    assert result.isequal(expected)
    result = cluster.triangles_core(G2)
    assert result.isequal(expected)
    mask = gb.Vector(bool, 5)
    mask[0] = True
    mask[3] = True
    result = cluster.triangles_core(G, mask=mask.S)
    expected = gb.Vector(int, 5)
    expected[0] = 6
    expected[3] = 6
    assert result.isequal(expected)
    result = cluster.triangles_core(G2, mask=mask.S)
    assert result.isequal(expected)
    assert cluster.single_triangle_core(G, 1) == 6
    assert cluster.single_triangle_core(G, 0) == 6
    assert cluster.single_triangle_core(G2, 0) == 6
    assert cluster.total_triangles_core(G2) == 10
    assert cluster.total_triangles_core(G) == 10
    assert cluster.transitivity_core(G) == 1.0
    assert cluster.transitivity_core(G2) == 1.0
    result = cluster.clustering_core(G)
    expected = gb.Vector(float, 5)
    expected[:] = 1
    assert result.isequal(expected)
    result = cluster.clustering_core(G2)
    assert result.isequal(expected)
    assert cluster.single_clustering_core(G, 0) == 1
    assert cluster.single_clustering_core(G2, 0) == 1
    expected(mask.S, replace=True) << 1
    result = cluster.clustering_core(G, mask=mask.S)
    assert result.isequal(expected)
    result = cluster.clustering_core(G2, mask=mask.S)
    assert result.isequal(expected)
    assert cluster.average_clustering_core(G) == 1
    assert cluster.average_clustering_core(G2) == 1
    assert cluster.average_clustering_core(G, mask=mask.S) == 1
    assert cluster.average_clustering_core(G2, mask=mask.S) == 1


def test_directed(orig):
    # XXX" is transitivity supposed to work on directed graphs like this?
    G = nx.complete_graph(5, create_using=nx.DiGraph())
    G.remove_edge(1, 2)
    G.remove_edge(2, 3)
    G.add_node(5)
    G2 = DiGraph.from_networkx(G)
    expected = orig.transitivity(G)
    result = transitivity(G2)
    assert expected == result
    # clustering
    expected = orig.clustering(G)
    result = clustering(G2)
    assert result == expected
    expected = orig.clustering(G, [0, 1, 2])
    result = clustering(G2, [0, 1, 2])
    assert result == expected
    for i in range(6):
        assert orig.clustering(G, i) == clustering(G2, i)
    # average_clustering
    expected = orig.average_clustering(G)
    result = average_clustering(G2)
    assert result == expected
    expected = orig.average_clustering(G, [0, 1, 2])
    result = average_clustering(G2, [0, 1, 2])
    assert result == expected
    expected = orig.average_clustering(G, count_zeros=False)
    result = average_clustering(G2, count_zeros=False)
    assert result == expected


from networkx.algorithms.tests.test_cluster import *  # noqa isort:skip
