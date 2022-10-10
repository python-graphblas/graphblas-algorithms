import networkx as nx

from graphblas_algorithms import DiGraph, nxapi


def test_directed():
    # XXX" is transitivity supposed to work on directed graphs like this?
    G = nx.complete_graph(5, create_using=nx.DiGraph())
    G.remove_edge(1, 2)
    G.remove_edge(2, 3)
    G.add_node(5)
    G2 = DiGraph.from_networkx(G)
    expected = nx.transitivity(G)
    result = nxapi.transitivity(G2)
    assert expected == result
    # clustering
    expected = nx.clustering(G)
    result = nxapi.clustering(G2)
    assert result == expected
    expected = nx.clustering(G, [0, 1, 2])
    result = nxapi.clustering(G2, [0, 1, 2])
    assert result == expected
    for i in range(6):
        assert nx.clustering(G, i) == nxapi.clustering(G2, i)
    # average_clustering
    expected = nx.average_clustering(G)
    result = nxapi.average_clustering(G2)
    assert result == expected
    expected = nx.average_clustering(G, [0, 1, 2])
    result = nxapi.average_clustering(G2, [0, 1, 2])
    assert result == expected
    expected = nx.average_clustering(G, count_zeros=False)
    result = nxapi.average_clustering(G2, count_zeros=False)
    assert result == expected
