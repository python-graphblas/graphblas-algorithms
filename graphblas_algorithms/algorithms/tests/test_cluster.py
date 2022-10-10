import graphblas as gb

from graphblas_algorithms import Graph
from graphblas_algorithms.algorithms import cluster


def test_triangles_full():
    # Including self-edges!
    G = gb.Matrix(bool, 5, 5)
    G[:, :] = True
    G2 = gb.select.offdiag(G).new()
    G = Graph.from_graphblas(G)
    G2 = Graph.from_graphblas(G2)
    result = cluster.triangles(G)
    expected = gb.Vector(int, 5)
    expected[:] = 6
    assert result.isequal(expected)
    result = cluster.triangles(G2)
    assert result.isequal(expected)
    mask = gb.Vector(bool, 5)
    mask[0] = True
    mask[3] = True
    result = cluster.triangles(G, mask=mask.S)
    expected = gb.Vector(int, 5)
    expected[0] = 6
    expected[3] = 6
    assert result.isequal(expected)
    result = cluster.triangles(G2, mask=mask.S)
    assert result.isequal(expected)
    assert cluster.single_triangle(G, 1) == 6
    assert cluster.single_triangle(G, 0) == 6
    assert cluster.single_triangle(G2, 0) == 6
    assert cluster.total_triangles(G2) == 10
    assert cluster.total_triangles(G) == 10
    assert cluster.transitivity(G) == 1.0
    assert cluster.transitivity(G2) == 1.0
    result = cluster.clustering(G)
    expected = gb.Vector(float, 5)
    expected[:] = 1
    assert result.isequal(expected)
    result = cluster.clustering(G2)
    assert result.isequal(expected)
    assert cluster.single_clustering(G, 0) == 1
    assert cluster.single_clustering(G2, 0) == 1
    expected(mask.S, replace=True) << 1
    result = cluster.clustering(G, mask=mask.S)
    assert result.isequal(expected)
    result = cluster.clustering(G2, mask=mask.S)
    assert result.isequal(expected)
    assert cluster.average_clustering(G) == 1
    assert cluster.average_clustering(G2) == 1
    assert cluster.average_clustering(G, mask=mask.S) == 1
    assert cluster.average_clustering(G2, mask=mask.S) == 1
