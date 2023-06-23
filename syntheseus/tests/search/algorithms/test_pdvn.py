# TODO: test everything in the PDVN file!!
from __future__ import annotations

import math

import pytest

from syntheseus.search.algorithms.pdvn import PDVN_MCTS, pdvn_extract_training_data
from syntheseus.search.analysis.solution_time import get_first_solution_time
from syntheseus.search.graph.and_or import AndNode, OrNode
from syntheseus.search.node_evaluation.base import NoCacheNodeEvaluator
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator
from syntheseus.tests.search.algorithms.test_base import BaseAlgorithmTest
from syntheseus.tests.search.conftest import RetrosynthesisTask


class TestPDVN_MCTS(BaseAlgorithmTest):
    def setup_algorithm(self, **kwargs) -> PDVN_MCTS:
        kwargs.setdefault("c_dead", 10.0)
        kwargs.setdefault("value_function_syn", ConstantNodeEvaluator(0.8))
        kwargs.setdefault("value_function_cost", ConstantNodeEvaluator(2.0))
        kwargs.setdefault("policy", ConstantNodeEvaluator(1.0))
        kwargs.setdefault("and_node_cost_fn", ConstantNodeEvaluator(1.0))
        kwargs.setdefault("bound_constant", 1e4)
        return PDVN_MCTS(**kwargs)
    
    def test_training_data(self,
        retrosynthesis_task1: RetrosynthesisTask,
    ):
        # TODO: should this test be present?
        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task1, 100,
        )
        pdvn_extract_training_data(output_graph)
        # TODO: could do some tests for training data
