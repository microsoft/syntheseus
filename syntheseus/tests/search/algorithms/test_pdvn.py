# TODO: test everything in the PDVN file!!
from __future__ import annotations
from collections import Counter

import math

import pytest

from syntheseus.search.algorithms.pdvn import PDVN_MCTS, pdvn_extract_training_data
from syntheseus.search.analysis.solution_time import get_first_solution_time
from syntheseus.search.graph.and_or import AndNode, OrNode
from syntheseus.search.node_evaluation.base import NoCacheNodeEvaluator
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator
from syntheseus.tests.search.algorithms.test_base import BaseAlgorithmTest
from syntheseus.tests.search.conftest import RetrosynthesisTask
from syntheseus.tests.search.algorithms.test_best_first import rxn_cost_fn, DictMolCost, DictRxnCost

@pytest.fixture
def policy(rxn_cost_fn) -> DictRxnCost:
    return DictRxnCost(rxn_smiles_to_cost={rxn: math.exp(-c) for rxn, c in rxn_cost_fn.rxn_to_cost.items()}, default=0.4)

@pytest.fixture
def mol_syn_estimate() -> DictMolCost:
    """Return synthesizability estimator for hand-worked example."""
    return DictMolCost(
        smiles_to_cost={
            "C": math.exp(-0.4),
            "OO": 1.0,
        },
        default=0.8,
    )

@pytest.fixture
def mol_cost_estimate() -> ConstantNodeEvaluator:
    """Return cost estimator for hand-worked example."""
    return ConstantNodeEvaluator(2.0)

# Reaction counts for hand-worked example
rxns_step1 = [
            "C>>CC" ,
            "CO>>CC",
            "CS>>CC",
]
rxns_step2 = rxns_step1  + [
            "C.O>>CO",
            "OO>>CO",
            "OS>>CO",
            "CS>>CO",
]
rxns_step3 = rxns_step2  + [
            "O>>OO",
            "CO>>OO",
            "OS>>OO",
]
rxns_step4 = rxns_step3  + [
    "C.S>>CS",
    "OS>>CS",
    "SS>>CS",
    "CO>>CS",
]
rxns_step5 = rxns_step4  + [
    "O>>C",
    "S>>C",
]


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

    def test_by_hand_step1(
        self,
        retrosynthesis_task5: RetrosynthesisTask,
        policy: DictRxnCost,
        mol_syn_estimate: DictMolCost,
        mol_cost_estimate: DictMolCost,
    ) -> None:
        
        # TODO: should expand once
        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task5, 1, policy=policy, value_function_syn=mol_syn_estimate, value_function_cost=mol_cost_estimate,
        )
        assert output_graph.reaction_smiles_counter() == Counter(rxns_step1)

    def test_by_hand_step2(  # TODO: step
        self,
        retrosynthesis_task5: RetrosynthesisTask,
        policy: DictRxnCost,
        mol_syn_estimate: DictMolCost,
        mol_cost_estimate: DictMolCost,
    ) -> None:
        
        # TODO: should expand once
        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task5, 3, policy=policy, value_function_syn=mol_syn_estimate, value_function_cost=mol_cost_estimate,
        )
        assert output_graph.reaction_smiles_counter() == Counter(rxns_step2)
    

    def test_by_hand_step3(  # TODO: step
        self,
        retrosynthesis_task5: RetrosynthesisTask,
        policy: DictRxnCost,
        mol_syn_estimate: DictMolCost,
        mol_cost_estimate: DictMolCost,
    ) -> None:
        
        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task5, 5, policy=policy, value_function_syn=mol_syn_estimate, value_function_cost=mol_cost_estimate,
        )
        assert output_graph.reaction_smiles_counter() == Counter(rxns_step3)

    def test_by_hand_step4(  # TODO: step
        self,
        retrosynthesis_task5: RetrosynthesisTask,
        policy: DictRxnCost,
        mol_syn_estimate: DictMolCost,
        mol_cost_estimate: DictMolCost,
    ) -> None:
        
        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task5, 6, policy=policy, value_function_syn=mol_syn_estimate, value_function_cost=mol_cost_estimate,
        )
        assert output_graph.reaction_smiles_counter() == Counter(rxns_step4)

    def test_by_hand_step5(  # TODO: step
        self,
        retrosynthesis_task5: RetrosynthesisTask,
        policy: DictRxnCost,
        mol_syn_estimate: DictMolCost,
        mol_cost_estimate: DictMolCost,
    ) -> None:
        
        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task5, 10, policy=policy, value_function_syn=mol_syn_estimate, value_function_cost=mol_cost_estimate,
        )
        assert output_graph.reaction_smiles_counter() == Counter(rxns_step5)