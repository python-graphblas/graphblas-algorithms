import networkx as nx
import pytest

from . import nxapi


class Dispatcher:
    # Link Analysis
    pagerank = nxapi.link_analysis.pagerank_alg.pagerank
    # Boundary
    edge_boundary = nxapi.boundary.edge_boundary
    node_boundary = nxapi.boundary.node_boundary
    # Centrality
    eigenvector_centrality = nxapi.centrality.eigenvector.eigenvector_centrality
    katz_centrality = nxapi.centrality.katz.katz_centrality
    # Community
    inter_community_edges = nxapi.community.quality.inter_community_edges
    intra_community_edges = nxapi.community.quality.intra_community_edges
    # Cuts
    boundary_expansion = nxapi.cuts.boundary_expansion
    conductance = nxapi.cuts.conductance
    cut_size = nxapi.cuts.cut_size
    edge_expansion = nxapi.cuts.edge_expansion
    mixing_expansion = nxapi.cuts.mixing_expansion
    node_expansion = nxapi.cuts.node_expansion
    normalized_cut_size = nxapi.cuts.normalized_cut_size
    volume = nxapi.cuts.volume
    # Cluster
    average_clustering = nxapi.cluster.average_clustering
    clustering = nxapi.cluster.clustering
    generalized_degree = nxapi.cluster.generalized_degree
    square_clustering = nxapi.cluster.square_clustering
    transitivity = nxapi.cluster.transitivity
    triangles = nxapi.cluster.triangles
    # Core
    k_truss = nxapi.core.k_truss
    # DAG
    ancestors = nxapi.dag.ancestors
    descendants = nxapi.dag.descendants
    # Dominating
    is_dominating_set = nxapi.dominating.is_dominating_set
    # Isolate
    is_isolate = nxapi.isolate.is_isolate
    isolates = nxapi.isolate.isolates
    number_of_isolates = nxapi.isolate.number_of_isolates
    # Reciprocity
    overall_reciprocity = nxapi.overall_reciprocity
    reciprocity = nxapi.reciprocity
    # Regular
    is_k_regular = nxapi.regular.is_k_regular
    is_regular = nxapi.regular.is_regular
    # Shortest Paths
    has_path = nxapi.shortest_paths.generic.has_path
    # Simple Paths
    is_simple_path = nxapi.simple_paths.is_simple_path
    # S Metric
    s_metric = nxapi.s_metric
    # Structural Holes
    mutual_weight = nxapi.structuralholes.mutual_weight
    # Tournament
    is_tournament = nxapi.tournament.is_tournament
    score_sequence = nxapi.tournament.score_sequence
    tournament_matrix = nxapi.tournament.tournament_matrix
    # Triads
    is_triad = nxapi.triads.is_triad

    @staticmethod
    def convert(graph, weight=None):
        from .classes import DiGraph, Graph, MultiDiGraph, MultiGraph

        if isinstance(graph, nx.MultiDiGraph):
            return MultiDiGraph.from_networkx(graph, weight=weight)
        if isinstance(graph, nx.MultiGraph):
            return MultiGraph.from_networkx(graph, weight=weight)
        if isinstance(graph, nx.DiGraph):
            return DiGraph.from_networkx(graph, weight=weight)
        if isinstance(graph, nx.Graph):
            return Graph.from_networkx(graph, weight=weight)
        raise TypeError(f"Unsupported type of graph: {type(graph)}")

    @staticmethod
    def mark_nx_tests(items):
        skip = [
            ("test_attributes", {"TestBoruvka", "test_mst.py"}),
            ("test_weight_attribute", {"TestBoruvka", "test_mst.py"}),
        ]
        for item in items:
            kset = set(item.keywords)
            for test_name, keywords in skip:
                if item.name == test_name and keywords.issubset(kset):
                    item.add_marker(
                        pytest.mark.xfail(reason="unable to handle multi-attributed graphs")
                    )
