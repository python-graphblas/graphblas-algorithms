import networkx as nx
from hypothesis.strategies import booleans, composite, integers, lists, tuples


@composite
def custom_graph_generators(
    draw,
    directed=booleans(),
    self_loops=booleans(),
    sym_digraph=booleans(),
    edges=tuples(integers(0, 100), integers(0, 100), integers(-100, 100)),
    edge_data=booleans(),
):
    G = nx.DiGraph() if draw(directed) else nx.Graph()
    self_loop = draw(self_loops)
    edges = draw(lists(edges, max_size=1000))
    edge_data = draw(edge_data)
    for u, v, d in edges:
        if not self_loop and u == v:
            continue
        if edge_data:
            G.add_edge(u, v, data=d)
        else:
            G.add_edge(u, v)
    if G.is_directed():
        sym_digraph = draw(sym_digraph)
        if sym_digraph:
            G = G.to_undirected()
            G = G.to_directed()
    return G
