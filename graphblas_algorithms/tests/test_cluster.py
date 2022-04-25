import inspect

import networkx as nx

from graphblas_algorithms import triangles

nx_triangles = nx.triangles
nx.triangles = triangles
nx.algorithms.triangles = triangles
nx.algorithms.cluster.triangles = triangles


def test_signatures():
    nx_sig = inspect.signature(nx_triangles)
    sig = inspect.signature(triangles)
    assert nx_sig == sig


from networkx.algorithms.tests.test_cluster import *  # noqa isort:skip
