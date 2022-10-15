from graphblas import Vector

__all__ = ["degree_centrality", "in_degree_centrality", "out_degree_centrality"]


def _degree_centrality(G, degrees, name):
    N = len(G)
    rv = Vector(float, size=N, name=name)
    if N <= 1:
        rv << 1
    else:
        s = 1 / (N - 1)
        rv << s * degrees
    return rv


def degree_centrality(G, *, name="degree_centrality"):
    if G.is_directed():
        degrees = G.get_property("total_degrees+")
    else:
        degrees = G.get_property("degrees+")
    return _degree_centrality(G, degrees, name)


def in_degree_centrality(G, *, name="in_degree_centrality"):
    degrees = G.get_property("column_degrees+")
    return _degree_centrality(G, degrees, name)


def out_degree_centrality(G, *, name="out_degree_centrality"):
    degrees = G.get_property("row_degrees+")
    return _degree_centrality(G, degrees, name)
