def get_info():
    return {
        "backend_name": "graphblas",
        "project": "graphblas-algorithms",
        "package": "graphblas_algorithms",
        "url": "https://github.com/python-graphblas/graphblas-algorithms",
        "short_summary": "OpenMP-enabled sparse linear algebra backend.",
        # "description": "TODO",
        "functions": {
            "adjacency_matrix": {},
            "all_pairs_bellman_ford_path_length": {
                "extra_parameters": {
                    "chunksize : int or str, optional": "Split the computation into chunks; "
                    'may specify size as string or number of rows. Default "10 MiB"',
                },
            },
            "all_pairs_shortest_path_length": {
                "extra_parameters": {
                    "chunksize : int or str, optional": "Split the computation into chunks; "
                    'may specify size as string or number of rows. Default "10 MiB"',
                },
            },
            "ancestors": {},
            "average_clustering": {},
            "bellman_ford_path": {},
            "bellman_ford_path_length": {},
            "bethe_hessian_matrix": {},
            "bfs_layers": {},
            "boundary_expansion": {},
            "clustering": {},
            "complement": {},
            "compose": {},
            "conductance": {},
            "cut_size": {},
            "degree_centrality": {},
            "descendants": {},
            "descendants_at_distance": {},
            "difference": {},
            "directed_modularity_matrix": {},
            "disjoint_union": {},
            "edge_boundary": {},
            "edge_expansion": {},
            "efficiency": {},
            "ego_graph": {},
            "eigenvector_centrality": {},
            "fast_could_be_isomorphic": {},
            "faster_could_be_isomorphic": {},
            "floyd_warshall": {},
            "floyd_warshall_numpy": {},
            "floyd_warshall_predecessor_and_distance": {},
            "full_join": {},
            "generalized_degree": {},
            "google_matrix": {},
            "has_path": {},
            "hits": {},
            "in_degree_centrality": {},
            "inter_community_edges": {},
            "intersection": {},
            "intra_community_edges": {},
            "is_connected": {},
            "is_dominating_set": {},
            "is_isolate": {},
            "is_k_regular": {},
            "isolates": {},
            "is_regular": {},
            "is_simple_path": {},
            "is_tournament": {},
            "is_triad": {},
            "is_weakly_connected": {},
            "katz_centrality": {},
            "k_truss": {},
            "laplacian_matrix": {},
            "lowest_common_ancestor": {},
            "mixing_expansion": {},
            "modularity_matrix": {},
            "mutual_weight": {},
            "negative_edge_cycle": {},
            "node_boundary": {},
            "node_connected_component": {},
            "node_expansion": {},
            "normalized_cut_size": {},
            "normalized_laplacian_matrix": {},
            "number_of_isolates": {},
            "out_degree_centrality": {},
            "overall_reciprocity": {},
            "pagerank": {},
            "reciprocity": {},
            "reverse": {},
            "score_sequence": {},
            "single_source_bellman_ford_path_length": {},
            "single_source_shortest_path_length": {},
            "single_target_shortest_path_length": {},
            "s_metric": {},
            "square_clustering": {
                "extra_parameters": {
                    "chunksize : int or str, optional": "Split the computation into chunks; "
                    'may specify size as string or number of rows. Default "256 MiB"',
                },
            },
            "symmetric_difference": {},
            "tournament_matrix": {},
            "transitivity": {},
            "triangles": {},
            "union": {},
            "volume": {},
        },
    }
