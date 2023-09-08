"""
At the moment, only a smoke-test is done for graph standardization,
but the exact behaviour is not well-tested.
These tests should be added in the future.
"""

import pytest

from syntheseus.search.graph.and_or import AndOrGraph
from syntheseus.search.graph.molset import MolSetGraph
from syntheseus.search.graph.standardization import get_unique_node_andor_graph


def test_smoke_andor(andor_graph_non_minimal: AndOrGraph):
    with pytest.warns(UserWarning):
        output = get_unique_node_andor_graph(andor_graph_non_minimal)

    assert len(output) == len(andor_graph_non_minimal)  # no nodes deleted here


def test_smoke_molset(molset_tree_non_minimal: MolSetGraph):
    with pytest.warns(UserWarning):
        get_unique_node_andor_graph(molset_tree_non_minimal)
