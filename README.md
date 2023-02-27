# **GraphBLAS Algorithms**

[![conda-forge](https://img.shields.io/conda/vn/conda-forge/graphblas-algorithms.svg)](https://anaconda.org/conda-forge/graphblas-algorithms)
[![pypi](https://img.shields.io/pypi/v/graphblas-algorithms.svg)](https://pypi.python.org/pypi/graphblas-algorithms/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/python-graphblas/graphblas-algorithms/blob/main/LICENSE)
[![Tests](https://github.com/python-graphblas/graphblas-algorithms/workflows/Tests/badge.svg?branch=main)](https://github.com/python-graphblas/graphblas-algorithms/actions)
[![Coverage](https://codecov.io/gh/python-graphblas/graphblas-algorithms/branch/main/graph/badge.svg)](https://codecov.io/gh/python-graphblas/graphblas-algorithms)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7329185.svg)](https://doi.org/10.5281/zenodo.7329185)
[![Discord](https://img.shields.io/badge/Chat-Discord-blue)](https://discord.com/invite/vur45CbwMz)
<!--- [![Docs](https://readthedocs.org/projects/graphblas-algorithms/badge/?version=latest)](https://graphblas-algorithms.readthedocs.io/en/latest/) --->

GraphBLAS algorithms written in Python with [Python-graphblas](https://python-graphblas.readthedocs.io/en/latest/).  We are trying to target the NetworkX API algorithms where possible.

### Installation
```
conda install -c conda-forge graphblas-algorithms
```
```
pip install graphblas-algorithms
```

## Basic Usage

First, create a GraphBLAS Matrix.

```python
import graphblas as gb

M = gb.Matrix.from_coo(
  [0, 0, 1, 2, 2, 3],
  [1, 3, 0, 0, 1, 2],
  [1., 2., 3., 4., 5., 6.],
  nrows=4, ncols=4, dtype='float32'
)
```

Next wrap the Matrix as `ga.Graph`.

```python
import graphblas_algorithms as ga

G = ga.Graph(M)
```

Finally call an algorithm.

```python
hubs, authorities = ga.hits(G)
```

When the result is a value per node, a `gb.Vector` will be returned.
In the case of [HITS](https://en.wikipedia.org/wiki/HITS_algorithm),
two Vectors are returned representing the hubs and authorities values.

Algorithms whose result is a subgraph will return `ga.Graph`.

## Plugin for NetworkX

Dispatching to plugins is a new feature in Networkx 3.0.
When both `networkx` and `graphblas-algorithms` are installed in an
environment, calls to NetworkX algorithms can be dispatched to the
equivalent version in `graphblas-algorithms`.

### Dispatch Example
```python
import networkx as nx
import graphblas_algorithms as ga

# Generate a random graph (5000 nodes, 1_000_000 edges)
G = nx.erdos_renyi_graph(5000, 0.08)

# Explicitly convert to ga.Graph
G2 = ga.Graph.from_networkx(G)

# Pass G2 to NetworkX's k_truss
T5 = nx.k_truss(G2, 5)
```

`G2` is not a `nx.Graph`, but it does have an attribute
`__networkx_plugin__ = "graphblas"`. This tells NetworkX to
dispatch the k_truss call to graphblas-algorithms. This link
connection exists because graphblas-algorithms registers
itself as a "networkx.plugin" entry point.

The result `T5` is a `ga.Graph` representing the 5-truss structure of the
original graph. To convert to a NetworkX Graph, use:
```python
T5.to_networkx()
```

Note that even with the conversions to and from `ga.Graph`, this example still runs 10x
faster than using the native NetworkX k-truss implementation. Speed improvements scale
with graph size, so larger graphs will see an even larger speed-up relative to NetworkX.

### Plugin Algorithms

The following NetworkX algorithms have been implemented
by graphblas-algorithms and can be used following the
dispatch pattern shown above.

- Boundary
  - edge_boundary
  - node_boundary
- Centrality
  - degree_centrality
  - eigenvector_centrality
  - in_degree_centrality
  - katz_centrality
  - out_degree_centrality
- Cluster
  - average_clustering
  - clustering
  - generalized_degree
  - square_clustering
  - transitivity
  - triangles
- Community
  - inter_community_edges
  - intra_community_edges
- Core
  - k_truss
- Cuts
  - boundary_expansion
  - conductance
  - cut_size
  - edge_expansion
  - mixing_expansion
  - node_expansion
  - normalized_cut_size
  - volume
- DAG
  - ancestors
  - descendants
- Dominating
  - is_dominating_set
- Isolate
  - is_isolate
  - isolates
  - number_of_isolates
- Link Analysis
  - hits
  - pagerank
- Reciprocity
  - overall_reciprocity
  - reciprocity
- Regular
  - is_k_regular
  - is_regular
- Shortest Paths
  - floyd_warshall
  - floyd_warshall_predecessor_and_distance
  - single_source_bellman_ford_path_length
  - all_pairs_bellman_ford_path_length
  - has_path
- Simple Paths
  - is_simple_path
- S Metric
  - s_metric
- Structural Holes
  - mutual_weight
- Tournament
  - is_tournament
  - score_sequence
  - tournament_matrix
- Triads
  - is_triad
