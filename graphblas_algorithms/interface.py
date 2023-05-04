from . import nxapi

#######
# NOTE: Remember to update README.md when adding or removing algorithms from Dispatcher
#######


class Dispatcher:
    # Boundary
    edge_boundary = nxapi.boundary.edge_boundary
    node_boundary = nxapi.boundary.node_boundary
    # Centrality
    degree_centrality = nxapi.centrality.degree_alg.degree_centrality
    eigenvector_centrality = nxapi.centrality.eigenvector.eigenvector_centrality
    in_degree_centrality = nxapi.centrality.degree_alg.in_degree_centrality
    katz_centrality = nxapi.centrality.katz.katz_centrality
    out_degree_centrality = nxapi.centrality.degree_alg.out_degree_centrality
    # Cluster
    average_clustering = nxapi.cluster.average_clustering
    clustering = nxapi.cluster.clustering
    generalized_degree = nxapi.cluster.generalized_degree
    square_clustering = nxapi.cluster.square_clustering
    transitivity = nxapi.cluster.transitivity
    triangles = nxapi.cluster.triangles
    # Community
    inter_community_edges = nxapi.community.quality.inter_community_edges
    intra_community_edges = nxapi.community.quality.intra_community_edges
    # Components
    is_connected = nxapi.components.connected.is_connected
    node_connected_component = nxapi.components.connected.node_connected_component
    is_weakly_connected = nxapi.components.weakly_connected.is_weakly_connected
    # Core
    k_truss = nxapi.core.k_truss
    # Cuts
    boundary_expansion = nxapi.cuts.boundary_expansion
    conductance = nxapi.cuts.conductance
    cut_size = nxapi.cuts.cut_size
    edge_expansion = nxapi.cuts.edge_expansion
    mixing_expansion = nxapi.cuts.mixing_expansion
    node_expansion = nxapi.cuts.node_expansion
    normalized_cut_size = nxapi.cuts.normalized_cut_size
    volume = nxapi.cuts.volume
    # DAG
    ancestors = nxapi.dag.ancestors
    descendants = nxapi.dag.descendants
    # Dominating
    is_dominating_set = nxapi.dominating.is_dominating_set
    # Generators
    ego_graph = nxapi.generators.ego.ego_graph
    # Isolate
    is_isolate = nxapi.isolate.is_isolate
    isolates = nxapi.isolate.isolates
    number_of_isolates = nxapi.isolate.number_of_isolates
    # Link Analysis
    hits = nxapi.link_analysis.hits_alg.hits
    google_matrix = nxapi.link_analysis.pagerank_alg.google_matrix
    pagerank = nxapi.link_analysis.pagerank_alg.pagerank
    # Operators
    compose = nxapi.operators.binary.compose
    difference = nxapi.operators.binary.difference
    disjoint_union = nxapi.operators.binary.disjoint_union
    full_join = nxapi.operators.binary.full_join
    intersection = nxapi.operators.binary.intersection
    symmetric_difference = nxapi.operators.binary.symmetric_difference
    union = nxapi.operators.binary.union
    # Reciprocity
    overall_reciprocity = nxapi.overall_reciprocity
    reciprocity = nxapi.reciprocity
    # Regular
    is_k_regular = nxapi.regular.is_k_regular
    is_regular = nxapi.regular.is_regular
    # Shortest Paths
    floyd_warshall = nxapi.shortest_paths.dense.floyd_warshall
    floyd_warshall_numpy = nxapi.shortest_paths.dense.floyd_warshall_numpy
    floyd_warshall_predecessor_and_distance = (
        nxapi.shortest_paths.dense.floyd_warshall_predecessor_and_distance
    )
    has_path = nxapi.shortest_paths.generic.has_path
    single_source_shortest_path_length = (
        nxapi.shortest_paths.unweighted.single_source_shortest_path_length
    )
    single_target_shortest_path_length = (
        nxapi.shortest_paths.unweighted.single_target_shortest_path_length
    )
    all_pairs_shortest_path_length = nxapi.shortest_paths.unweighted.all_pairs_shortest_path_length
    bellman_ford_path = nxapi.shortest_paths.weighted.bellman_ford_path
    all_pairs_bellman_ford_path_length = (
        nxapi.shortest_paths.weighted.all_pairs_bellman_ford_path_length
    )
    negative_edge_cycle = nxapi.shortest_paths.weighted.negative_edge_cycle
    single_source_bellman_ford_path_length = (
        nxapi.shortest_paths.weighted.single_source_bellman_ford_path_length
    )
    # Simple Paths
    is_simple_path = nxapi.simple_paths.is_simple_path
    # S Metric
    s_metric = nxapi.smetric.s_metric
    # Structural Holes
    mutual_weight = nxapi.structuralholes.mutual_weight
    # Tournament
    is_tournament = nxapi.tournament.is_tournament
    score_sequence = nxapi.tournament.score_sequence
    tournament_matrix = nxapi.tournament.tournament_matrix
    # Traversal
    bfs_layers = nxapi.traversal.breadth_first_search.bfs_layers
    descendants_at_distance = nxapi.traversal.breadth_first_search.descendants_at_distance
    # Triads
    is_triad = nxapi.triads.is_triad

    @staticmethod
    def convert_from_nx(graph, weight=None, *, name=None):
        import networkx as nx

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
    def convert_to_nx(obj, *, name=None):
        from graphblas import Matrix

        from .classes import Graph

        if isinstance(obj, Graph):
            obj = obj.to_networkx()
        elif isinstance(obj, Matrix):
            obj = obj.to_dense(fill_value=False)
        return obj

    @staticmethod
    def on_start_tests(items):
        try:
            import pytest
        except ImportError:  # pragma: no cover (import)
            return

        def key(testpath):
            filename, path = testpath.split(":")
            *names, testname = path.split(".")
            if names:
                [classname] = names
                return (testname, frozenset({classname, filename}))
            return (testname, frozenset({filename}))

        # Reasons to skip tests
        multi_attributed = "unable to handle multi-attributed graphs"
        multidigraph = "unable to handle MultiDiGraph"
        multigraph = "unable to handle MultiGraph"

        # Which tests to skip
        skip = {
            key("test_mst.py:TestBoruvka.test_attributes"): multi_attributed,
            key("test_mst.py:TestBoruvka.test_weight_attribute"): multi_attributed,
            key("test_dense.py:TestFloyd.test_zero_weight"): multidigraph,
            key("test_dense_numpy.py:test_zero_weight"): multidigraph,
            key("test_weighted.py:TestBellmanFordAndGoldbergRadzik.test_multigraph"): multigraph,
            key("test_binary.py:test_compose_multigraph"): multigraph,
            key("test_binary.py:test_difference_multigraph_attributes"): multigraph,
            key("test_binary.py:test_disjoint_union_multigraph"): multigraph,
            key("test_binary.py:test_full_join_multigraph"): multigraph,
            key("test_binary.py:test_intersection_multigraph_attributes"): multigraph,
            key(
                "test_binary.py:test_intersection_multigraph_attributes_node_set_different"
            ): multigraph,
            key("test_binary.py:test_symmetric_difference_multigraph"): multigraph,
            key("test_binary.py:test_union_attributes"): multi_attributed,
            # TODO: move failing assertion from `test_union_and_compose`
            key("test_binary.py:test_union_and_compose"): multi_attributed,
            key("test_binary.py:test_union_multigraph"): multigraph,
            key("test_vf2pp.py:test_custom_multigraph4_different_labels"): multigraph,
        }
        for item in items:
            kset = set(item.keywords)
            for (test_name, keywords), reason in skip.items():
                if item.name == test_name and keywords.issubset(kset):
                    item.add_marker(pytest.mark.xfail(reason=reason))
