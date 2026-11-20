"""Backend info NetworkX reads through the ``networkx.backend_info`` entry point.

``_functions`` and ``_extra_parameters`` are generated from the dispatcher and from our
signatures: run ``python _nx_graphblas/__init__.py``, or ``make plugin-info``, after adding
or removing an algorithm. ``_nx_graphblas/core.py`` holds the code that writes them.
"""


def get_info():
    # NetworkX looks for `extra_parameters` inside each function's own dict, so the
    # parameter table has to stay wrapped in that key rather than be the dict itself.
    functions = {
        name: {"extra_parameters": _extra_parameters[name]} if name in _extra_parameters else {}
        for name in _functions
    }
    return {
        "backend_name": "graphblas",
        "project": "graphblas-algorithms",
        "package": "graphblas_algorithms",
        "url": "https://github.com/python-graphblas/graphblas-algorithms",
        "short_summary": "OpenMP-enabled sparse linear algebra backend.",
        # "description": "TODO",
        "functions": functions,
    }


# fmt: off
_functions = (
    # BEGIN: functions
    "adjacency_matrix",
    "all_pairs_bellman_ford_path_length",
    "all_pairs_shortest_path_length",
    "ancestors",
    "average_clustering",
    "bellman_ford_path",
    "bellman_ford_path_length",
    "bethe_hessian_matrix",
    "bfs_layers",
    "boundary_expansion",
    "clustering",
    "complement",
    "compose",
    "conductance",
    "cut_size",
    "degree_centrality",
    "descendants",
    "descendants_at_distance",
    "difference",
    "directed_modularity_matrix",
    "disjoint_union",
    "edge_boundary",
    "edge_expansion",
    "efficiency",
    "ego_graph",
    "eigenvector_centrality",
    "fast_could_be_isomorphic",
    "faster_could_be_isomorphic",
    "floyd_warshall",
    "floyd_warshall_numpy",
    "floyd_warshall_predecessor_and_distance",
    "full_join",
    "generalized_degree",
    "google_matrix",
    "has_path",
    "hits",
    "in_degree_centrality",
    "inter_community_edges",
    "intersection",
    "intra_community_edges",
    "is_connected",
    "is_dominating_set",
    "is_isolate",
    "is_k_regular",
    "is_regular",
    "is_simple_path",
    "is_tournament",
    "is_triad",
    "is_weakly_connected",
    "isolates",
    "k_truss",
    "katz_centrality",
    "laplacian_matrix",
    "lowest_common_ancestor",
    "mixing_expansion",
    "modularity_matrix",
    "mutual_weight",
    "negative_edge_cycle",
    "node_boundary",
    "node_connected_component",
    "node_expansion",
    "normalized_cut_size",
    "normalized_laplacian_matrix",
    "number_of_isolates",
    "out_degree_centrality",
    "overall_reciprocity",
    "pagerank",
    "reciprocity",
    "reverse",
    "s_metric",
    "score_sequence",
    "single_source_bellman_ford_path_length",
    "single_source_shortest_path_length",
    "single_target_shortest_path_length",
    "square_clustering",
    "symmetric_difference",
    "tournament_matrix",
    "transitivity",
    "triangles",
    "union",
    "volume",
    # END: functions
)

_extra_parameters = {
    # BEGIN: extra_parameters
    "all_pairs_bellman_ford_path_length": {
        "chunksize : int or str, optional": 'Split the computation into chunks; may specify size as string or number of rows. Default "10 MiB"',
    },
    "all_pairs_shortest_path_length": {
        "chunksize : int or str, optional": 'Split the computation into chunks; may specify size as string or number of rows. Default "10 MiB"',
    },
    "square_clustering": {
        "chunksize : int or str, optional": 'Split the computation into chunks; may specify size as string or number of rows. Default "256 MiB"',
    },
    # END: extra_parameters
}
# fmt: on

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Running this file as a script puts _nx_graphblas/ on sys.path, not the repo root,
    # so the package it lives in would not be importable without this.
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from _nx_graphblas.core import main

    main()
