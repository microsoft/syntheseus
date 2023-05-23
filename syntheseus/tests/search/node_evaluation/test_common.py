import pytest

from syntheseus.search.graph.and_or import AndOrGraph
from syntheseus.search.node_evaluation.common import (
    ConstantNodeEvaluator,
    HasSolutionValueFunction,
)


class TestConstantNodeEvaluator:
    @pytest.mark.parametrize("constant", [0.3, 0.8])
    def test_values(self, constant: float, andor_graph_non_minimal: AndOrGraph) -> None:
        val_fn = ConstantNodeEvaluator(constant)
        vals = val_fn(list(andor_graph_non_minimal.nodes()))
        assert all([v == constant for v in vals])  # values should match
        assert val_fn.num_calls == len(
            andor_graph_non_minimal
        )  # should have been called once per node


class TestHasSolutionValueFunction:
    def test_values(self, andor_graph_non_minimal: AndOrGraph) -> None:
        val_fn = HasSolutionValueFunction()
        nodes = list(andor_graph_non_minimal.nodes())
        vals = val_fn(nodes)
        assert all([v == float(n.has_solution) for v, n in zip(vals, nodes)])  # values should match
        assert val_fn.num_calls == len(
            andor_graph_non_minimal
        )  # should have been called once per node
