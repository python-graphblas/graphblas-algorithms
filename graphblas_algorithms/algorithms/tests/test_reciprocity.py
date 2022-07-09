from graphblas_algorithms import overall_reciprocity, reciprocity
import pytest
from hypothesis import given, settings
from graphblas_algorithms.utils import custom_graph_generators


@settings(deadline=None, max_examples=500)
@given(custom_graph_generators())
def test_overall_reciprocity_gen(graph):
    # TODO: Fix the graph gen to not produce empty graphs and control directions
    if len(graph) > 0 and graph.is_directed():
        assert overall_reciprocity(graph) == pytest.approx(nx.overall_reciprocity(graph))


@settings(deadline=None, max_examples=500)
@given(custom_graph_generators())
def test_reciprocity_gen(graph):
    if len(graph) > 0 and graph.is_directed():
        assert reciprocity(graph) == pytest.approx(nx.reciprocity(graph))


from networkx.algorithms.tests.test_reciprocity import *  # isort:skip
