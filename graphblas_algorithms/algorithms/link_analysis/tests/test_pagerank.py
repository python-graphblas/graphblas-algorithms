import pytest
from hypothesis import given, settings

from graphblas_algorithms import pagerank
from graphblas_algorithms.utils import custom_graph_generators


@settings(deadline=None, max_examples=500)
@given(custom_graph_generators())
def test_pagerank_gen(graph):
    assert pagerank(graph) == pytest.approx(nx.pagerank(graph))


from networkx.algorithms.link_analysis.tests.test_pagerank import *  # isort:skip
