import inspect

import networkx as nx

from graphblas_algorithms import pagerank

nx_pagerank = nx.pagerank
nx_pagerank_scipy = nx.pagerank_scipy

nx.pagerank = pagerank
nx.pagerank_scipy = pagerank
nx.algorithms.link_analysis.pagerank_alg.pagerank_scipy = pagerank


def test_signatures():
    nx_sig = inspect.signature(nx_pagerank)
    sig = inspect.signature(pagerank)
    assert nx_sig == sig


from networkx.algorithms.link_analysis.tests.test_pagerank import *  # isort:skip
