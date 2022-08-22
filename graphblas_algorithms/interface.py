from . import algorithms


class Dispatcher:
    pagerank = algorithms.pagerank
    generalized_degree = algorithms.generalized_degree
    clustering = algorithms.clustering
    average_clustering = algorithms.average_clustering
    square_clustering = algorithms.square_clustering
    transitivity = algorithms.transitivity
    k_truss = algorithms.k_truss
    reciprocity = algorithms.reciprocity
    overall_reciprocity = algorithms.overall_reciprocity

    @staticmethod
    def convert(graph, weight=None):
        import networkx as nx
        from .classes import Graph, DiGraph

        if isinstance(graph, nx.DiGraph):
            return DiGraph.from_networkx(graph, weight=weight)
        return Graph.from_networkx(graph, weight=weight)
