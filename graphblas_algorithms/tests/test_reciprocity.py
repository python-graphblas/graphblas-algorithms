import inspect

import networkx as nx

from graphblas_algorithms import overall_reciprocity, reciprocity

nx_reciprocity = nx.reciprocity
nx.reciprocity = reciprocity
nx.algorithms.reciprocity = reciprocity

nx_overall_reciprocity = nx.overall_reciprocity
nx.overall_reciprocity = overall_reciprocity
nx.algorithms.overall_reciprocity = overall_reciprocity


def test_signatures():
    nx_sig = inspect.signature(nx_reciprocity)
    sig = inspect.signature(reciprocity)
    assert nx_sig == sig
    nx_sig = inspect.signature(nx_overall_reciprocity)
    sig = inspect.signature(overall_reciprocity)
    assert nx_sig == sig


from networkx.algorithms.tests.test_reciprocity import *  # noqa isort:skip
