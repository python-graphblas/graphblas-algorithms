from . import nxapi

#######
# NOTE: Remember to run `python scripts/maketree.py` when adding or removing algorithms
#       to automatically add it to README.md. You must still add algorithms below.
#######


class Dispatcher:
    # Begin auto-generated code: dispatch
    mod = nxapi.boundary
    # ==================
    edge_boundary = mod.edge_boundary
    node_boundary = mod.node_boundary

    mod = nxapi.centrality
    # ====================
    degree_centrality = mod.degree_alg.degree_centrality
    in_degree_centrality = mod.degree_alg.in_degree_centrality
    out_degree_centrality = mod.degree_alg.out_degree_centrality
    eigenvector_centrality = mod.eigenvector.eigenvector_centrality
    katz_centrality = mod.katz.katz_centrality

    mod = nxapi.cluster
    # =================
    average_clustering = mod.average_clustering
    clustering = mod.clustering
    generalized_degree = mod.generalized_degree
    square_clustering = mod.square_clustering
    transitivity = mod.transitivity
    triangles = mod.triangles

    mod = nxapi.community
    # ===================
    inter_community_edges = mod.quality.inter_community_edges
    intra_community_edges = mod.quality.intra_community_edges

    mod = nxapi.components
    # ====================
    is_connected = mod.connected.is_connected
    node_connected_component = mod.connected.node_connected_component
    is_weakly_connected = mod.weakly_connected.is_weakly_connected

    mod = nxapi.core
    # ==============
    k_truss = mod.k_truss

    mod = nxapi.cuts
    # ==============
    boundary_expansion = mod.boundary_expansion
    conductance = mod.conductance
    cut_size = mod.cut_size
    edge_expansion = mod.edge_expansion
    mixing_expansion = mod.mixing_expansion
    node_expansion = mod.node_expansion
    normalized_cut_size = mod.normalized_cut_size
    volume = mod.volume

    mod = nxapi.dag
    # =============
    ancestors = mod.ancestors
    descendants = mod.descendants

    mod = nxapi.dominating
    # ====================
    is_dominating_set = mod.is_dominating_set

    mod = nxapi.efficiency_measures
    # =============================
    efficiency = mod.efficiency

    mod = nxapi.generators
    # ====================
    ego_graph = mod.ego.ego_graph

    mod = nxapi.isolate
    # =================
    is_isolate = mod.is_isolate
    isolates = mod.isolates
    number_of_isolates = mod.number_of_isolates

    mod = nxapi.isomorphism
    # =====================
    fast_could_be_isomorphic = mod.isomorph.fast_could_be_isomorphic
    faster_could_be_isomorphic = mod.isomorph.faster_could_be_isomorphic

    mod = nxapi.linalg
    # ================
    bethe_hessian_matrix = mod.bethehessianmatrix.bethe_hessian_matrix
    adjacency_matrix = mod.graphmatrix.adjacency_matrix
    laplacian_matrix = mod.laplacianmatrix.laplacian_matrix
    normalized_laplacian_matrix = mod.laplacianmatrix.normalized_laplacian_matrix
    directed_modularity_matrix = mod.modularitymatrix.directed_modularity_matrix
    modularity_matrix = mod.modularitymatrix.modularity_matrix

    mod = nxapi.link_analysis
    # =======================
    hits = mod.hits_alg.hits
    google_matrix = mod.pagerank_alg.google_matrix
    pagerank = mod.pagerank_alg.pagerank

    mod = nxapi.lowest_common_ancestors
    # =================================
    lowest_common_ancestor = mod.lowest_common_ancestor

    mod = nxapi.operators
    # ===================
    compose = mod.binary.compose
    difference = mod.binary.difference
    disjoint_union = mod.binary.disjoint_union
    full_join = mod.binary.full_join
    intersection = mod.binary.intersection
    symmetric_difference = mod.binary.symmetric_difference
    union = mod.binary.union
    complement = mod.unary.complement
    reverse = mod.unary.reverse

    mod = nxapi.reciprocity
    # =====================
    overall_reciprocity = nxapi.overall_reciprocity
    reciprocity = nxapi.reciprocity

    mod = nxapi.regular
    # =================
    is_k_regular = mod.is_k_regular
    is_regular = mod.is_regular

    mod = nxapi.shortest_paths
    # ========================
    floyd_warshall = mod.dense.floyd_warshall
    floyd_warshall_numpy = mod.dense.floyd_warshall_numpy
    floyd_warshall_predecessor_and_distance = mod.dense.floyd_warshall_predecessor_and_distance
    has_path = mod.generic.has_path
    all_pairs_shortest_path_length = mod.unweighted.all_pairs_shortest_path_length
    single_source_shortest_path_length = mod.unweighted.single_source_shortest_path_length
    single_target_shortest_path_length = mod.unweighted.single_target_shortest_path_length
    all_pairs_bellman_ford_path_length = mod.weighted.all_pairs_bellman_ford_path_length
    bellman_ford_path = mod.weighted.bellman_ford_path
    bellman_ford_path_length = mod.weighted.bellman_ford_path_length
    negative_edge_cycle = mod.weighted.negative_edge_cycle
    single_source_bellman_ford_path_length = mod.weighted.single_source_bellman_ford_path_length

    mod = nxapi.simple_paths
    # ======================
    is_simple_path = mod.is_simple_path

    mod = nxapi.smetric
    # =================
    s_metric = mod.s_metric

    mod = nxapi.structuralholes
    # =========================
    mutual_weight = mod.mutual_weight

    mod = nxapi.tournament
    # ====================
    is_tournament = mod.is_tournament
    score_sequence = mod.score_sequence
    tournament_matrix = mod.tournament_matrix

    mod = nxapi.traversal
    # ===================
    bfs_layers = mod.breadth_first_search.bfs_layers
    descendants_at_distance = mod.breadth_first_search.descendants_at_distance

    mod = nxapi.triads
    # ================
    is_triad = mod.is_triad

    del mod
    # End auto-generated code: dispatch

    @staticmethod
    def convert_from_nx(
        graph,
        edge_attrs=None,
        node_attrs=None,
        preserve_edge_attrs=False,
        preserve_node_attrs=False,
        preserve_graph_attrs=False,
        name=None,
        graph_name=None,
        *,
        weight=None,  # For nx.__version__ <= 3.1
    ):
        import networkx as nx

        from .classes import DiGraph, Graph, MultiDiGraph, MultiGraph

        if preserve_edge_attrs:
            if graph.is_multigraph():
                attrs = set().union(
                    *(
                        datadict
                        for nbrs in graph._adj.values()
                        for keydict in nbrs.values()
                        for datadict in keydict.values()
                    )
                )
            else:
                attrs = set().union(
                    *(datadict for nbrs in graph._adj.values() for datadict in nbrs.values())
                )
            if len(attrs) == 1:
                [attr] = attrs
                edge_attrs = {attr: None}
            elif attrs:
                raise NotImplementedError("`preserve_edge_attrs=True` is not fully implemented")
        if node_attrs:
            raise NotImplementedError("non-None `node_attrs` is not yet implemented")
        if preserve_node_attrs:
            attrs = set().union(*(datadict for node, datadict in graph.nodes(data=True)))
            if attrs:
                raise NotImplementedError("`preserve_node_attrs=True` is not implemented")
        if edge_attrs:
            if len(edge_attrs) > 1:
                raise NotImplementedError(
                    "Multiple edge attributes is not implemented (bad value for edge_attrs)"
                )
            if weight is not None:
                raise TypeError("edge_attrs and weight both given")
            [[weight, default]] = edge_attrs.items()
            if default is not None and default != 1:
                raise NotImplementedError(f"edge default != 1 is not implemented; got {default}")

        if isinstance(graph, nx.MultiDiGraph):
            G = MultiDiGraph.from_networkx(graph, weight=weight)
        elif isinstance(graph, nx.MultiGraph):
            G = MultiGraph.from_networkx(graph, weight=weight)
        elif isinstance(graph, nx.DiGraph):
            G = DiGraph.from_networkx(graph, weight=weight)
        elif isinstance(graph, nx.Graph):
            G = Graph.from_networkx(graph, weight=weight)
        else:
            raise TypeError(f"Unsupported type of graph: {type(graph)}")
        if preserve_graph_attrs:
            G.graph.update(graph.graph)
        return G

    @staticmethod
    def convert_to_nx(obj, *, name=None):
        from graphblas import Matrix, io

        from .classes import Graph

        if isinstance(obj, Graph):
            obj = obj.to_networkx()
        elif isinstance(obj, Matrix):
            if name in {
                "adjacency_matrix",
                "bethe_hessian_matrix",
                "laplacian_matrix",
                "normalized_laplacian_matrix",
                "tournament_matrix",
            }:
                obj = io.to_scipy_sparse(obj)
            elif name in {
                "directed_modularity_matrix",
                "floyd_warshall_numpy",
                "google_matrix",
                "modularity_matrix",
            }:
                obj = obj.to_dense(fill_value=False)
            else:  # pragma: no cover
                raise RuntimeError(f"Should {name} return a numpy or scipy.sparse array?")
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
        # multi_attributed = "unable to handle multi-attributed graphs"
        multidigraph = "unable to handle MultiDiGraph"
        multigraph = "unable to handle MultiGraph"

        # Which tests to skip
        skip = {
            # key("test_mst.py:TestBoruvka.test_attributes"): multi_attributed,
            # key("test_mst.py:TestBoruvka.test_weight_attribute"): multi_attributed,
            key("test_dense.py:TestFloyd.test_zero_weight"): multidigraph,
            key("test_dense_numpy.py:test_zero_weight"): multidigraph,
            key("test_weighted.py:TestBellmanFordAndGoldbergRadzik.test_multigraph"): multigraph,
            # key("test_binary.py:test_compose_multigraph"): multigraph,
            # key("test_binary.py:test_difference_multigraph_attributes"): multigraph,
            # key("test_binary.py:test_disjoint_union_multigraph"): multigraph,
            # key("test_binary.py:test_full_join_multigraph"): multigraph,
            # key("test_binary.py:test_intersection_multigraph_attributes"): multigraph,
            # key(
            #     "test_binary.py:test_intersection_multigraph_attributes_node_set_different"
            # ): multigraph,
            # key("test_binary.py:test_symmetric_difference_multigraph"): multigraph,
            # key("test_binary.py:test_union_attributes"): multi_attributed,
            # TODO: move failing assertion from `test_union_and_compose`
            # key("test_binary.py:test_union_and_compose"): multi_attributed,
            # key("test_binary.py:test_union_multigraph"): multigraph,
            # key("test_vf2pp.py:test_custom_multigraph4_different_labels"): multigraph,
        }
        for item in items:
            kset = set(item.keywords)
            for (test_name, keywords), reason in skip.items():
                if item.name == test_name and keywords.issubset(kset):
                    item.add_marker(pytest.mark.xfail(reason=reason))
