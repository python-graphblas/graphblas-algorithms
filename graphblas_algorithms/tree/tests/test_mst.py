# This helps test cuts.py
import pytest
from networkx.algorithms.tree.tests.test_mst import MinimumSpanningTreeTestBase, TestBoruvka

from networkx.algorithms.tree.tests.test_mst import *  # isort:skip

TestBoruvka.test_attributes = lambda self: pytest.xfail(
    "unable to handle multi-attributed graph"
) or MinimumSpanningTreeTestBase.test_attributes(self)
TestBoruvka.test_weight_attribute = lambda self: pytest.xfail(
    "unable to handle multi-attributed graph"
) or MinimumSpanningTreeTestBase.test_weight_attribute(self)
