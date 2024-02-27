from __future__ import annotations

import math
from collections import Counter

import pytest

from syntheseus.interface.bag import Bag
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.reaction_prediction.inference.toy_models import ListOfReactionsToyModel
from syntheseus.search.algorithms.pdvn import PDVN_MCTS, pdvn_extract_training_data
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator
from syntheseus.tests.search.algorithms.test_base import BaseAlgorithmTest
from syntheseus.tests.search.algorithms.test_best_first import (
    DictMolCost,
    DictRxnCost,
    rxn_cost_fn,  # noqa: F401
)
from syntheseus.tests.search.conftest import RetrosynthesisTask


@pytest.fixture
def policy(
    rxn_cost_fn,  # noqa: F811  # doesn't understand pytest's use of rxn_cost_fn as fixture
) -> DictRxnCost:
    return DictRxnCost(
        rxn_smiles_to_cost={rxn: math.exp(-c) for rxn, c in rxn_cost_fn.rxn_to_cost.items()},
        default=0.4,
    )


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
    "C>>CC",
    "CO>>CC",
    "CS>>CC",
]
rxns_step2 = rxns_step1 + [
    "C.O>>CO",
    "OO>>CO",
    "OS>>CO",
    "CS>>CO",
]
rxns_step3 = rxns_step2 + [
    "O>>OO",
    "CO>>OO",
    "OS>>OO",
]
rxns_step4 = rxns_step3 + [
    "C.S>>CS",
    "OS>>CS",
    "SS>>CS",
    "CO>>CS",
]
rxns_step5 = rxns_step4 + [
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

    def test_by_hand_step1(
        self,
        retrosynthesis_task5: RetrosynthesisTask,
        policy: DictRxnCost,
        mol_syn_estimate: DictMolCost,
        mol_cost_estimate: DictMolCost,
    ) -> None:
        r"""
        Test PDVN-MCTS on a hand-designed example which closely mirrors the hand-designed
        example from the tests for retro*. The example is designed to first explore a
        sub-optimal path (O -> OO -> CO -> CC) before finding the better path (O -> C -> CC).

        The cost of all reactions is 1, V_syn is a lookup table,
        V_cost is 2 for all molecules, and the policy is another lookup table.

        In the first step, the algorithm should expand the root node (CC),
        and the tree should have the following structure:

                 ----------CC --------
                /          |          \
              C            CO         CS
        """

        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task5,
            1,
            policy=policy,
            value_function_syn=mol_syn_estimate,
            value_function_cost=mol_cost_estimate,
        )
        assert output_graph.reaction_smiles_counter() == Counter(rxns_step1)  # type: ignore[attr-defined]

        # Various checks on root node
        assert (
            output_graph.root_node.num_visit == 2
        )  # 1 pseudo-visit at the beginning, then 1 real visit
        assert math.isclose(
            output_graph.root_node.data["pdvn_mcts_v_syn"],
            0.4,
        ), "v_syn should be 0.4 average(0, 0.8)"
        assert math.isclose(
            output_graph.root_node.data["pdvn_mcts_v_cost"],
            1.5,
        ), "v_cost should be 1.5 average(0, 3)"

        # Training data
        training_data = pdvn_extract_training_data(output_graph)  # type: ignore[arg-type]  # doesn't understand it is AND/OR graph
        assert all(v == 0 for v in training_data.mol_to_synthesizability.values())  # nothing solved
        assert (
            len(training_data.mol_to_min_syn_cost)
            == len(training_data.mol_to_reactions_for_min_syn)
            == 0
        ), "no solutions found means no cost / reaction data"

    def test_by_hand_step2(
        self,
        retrosynthesis_task5: RetrosynthesisTask,
        policy: DictRxnCost,
        mol_syn_estimate: DictMolCost,
        mol_cost_estimate: DictMolCost,
    ) -> None:
        r"""
        Continuation of test above.

        In the second iteration the value function for CS should be evaluated,
        and in the third iteration CO should be expanded and the OO node visited.

        The tree should have the following structure:
                 ----------CC --------
                /          |          \
              C            CO         CS
                           |
                -----------------------
               /       /        \      \
             C+O      OO         OS     CS
        """

        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task5,
            3,
            policy=policy,
            value_function_syn=mol_syn_estimate,
            value_function_cost=mol_cost_estimate,
        )
        assert output_graph.reaction_smiles_counter() == Counter(rxns_step2)  # type: ignore[attr-defined]

        # Various checks on root node
        assert output_graph.root_node.num_visit == 4
        assert math.isclose(
            output_graph.root_node.data["pdvn_mcts_v_syn"],
            2.6 / 4,
        ), "v_syn should be 0.65 average(0, 0.8, 0.8, 1.0)"
        assert math.isclose(
            output_graph.root_node.data["pdvn_mcts_v_cost"],
            10 / 4,
        ), "v_cost should be 2.5 average(0, 3, 3, 4)"

        # Training data
        training_data = pdvn_extract_training_data(output_graph)  # type: ignore[arg-type]  # doesn't understand it is AND/OR graph
        assert all(
            v == 0 or (mol.smiles == "O" and v == 1)
            for mol, v in training_data.mol_to_synthesizability.items()
        )  # only "O" is solved (because it is purchasable)
        assert training_data.mol_to_min_syn_cost == {
            Molecule("O"): 0.0
        }, "Only O is solved, and it has cost 0.0 since it is purchasable"
        assert (
            len(training_data.mol_to_reactions_for_min_syn) == 0
        ), "Only solved molecule has no reactions"

    def test_by_hand_step3(
        self,
        retrosynthesis_task5: RetrosynthesisTask,
        policy: DictRxnCost,
        mol_syn_estimate: DictMolCost,
        mol_cost_estimate: DictMolCost,
    ) -> None:
        r"""
        Continuation of test above.

        In the fourth iteration the value function for C should be evaluated,
        and in the fifth iteration OO should be expanded.

        The tree should have the following structure:
                 ----------CC --------
                /          |          \
              C            CO         CS
                           |
                -----------------------
               /       /        \      \
             C+O      OO         OS     CS
                       |
              ---------|---------
             /         |         \
            O         CO         OS
        """

        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task5,
            5,
            policy=policy,
            value_function_syn=mol_syn_estimate,
            value_function_cost=mol_cost_estimate,
        )
        assert output_graph.reaction_smiles_counter() == Counter(rxns_step3)  # type: ignore[attr-defined]

        # Check training data is correct
        training_data = pdvn_extract_training_data(output_graph)  # type: ignore[arg-type]  # doesn't understand it is AND/OR graph
        assert all(
            v == 0 or (mol.smiles in {"O", "OO", "CO", "CC"} and v == 1)
            for mol, v in training_data.mol_to_synthesizability.items()
        )
        assert training_data.mol_to_min_syn_cost == {
            Molecule("O"): 0.0,
            Molecule("OO"): 1.0,
            Molecule("CO"): 2.0,
            Molecule("CC"): 3.0,
        }
        assert len(training_data.mol_to_reactions_for_min_syn) == 3

    def test_by_hand_step4(
        self,
        retrosynthesis_task5: RetrosynthesisTask,
        policy: DictRxnCost,
        mol_syn_estimate: DictMolCost,
        mol_cost_estimate: DictMolCost,
    ) -> None:
        r"""
        Continuation of test above.

        In the 6th iteration CS in the top row should be expanded.
        The tree should have the following structure (children under CS not drawn):
                 ----------CC -------------------
                /          |                     \
              C            CO                    CS
                           |                      |
                -----------------------          ...
               /       /        \      \
             C+O      OO         OS     CS
                       |
              ---------|---------
             /         |         \
            O         CO         OS
        """

        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task5,
            6,
            policy=policy,
            value_function_syn=mol_syn_estimate,
            value_function_cost=mol_cost_estimate,
        )
        assert output_graph.reaction_smiles_counter() == Counter(rxns_step4)  # type: ignore[attr-defined]

        # No new solutions found
        training_data = pdvn_extract_training_data(output_graph)  # type: ignore[arg-type]  # doesn't understand it is AND/OR graph
        assert len(training_data.mol_to_reactions_for_min_syn) == 3

    def test_by_hand_step5(
        self,
        retrosynthesis_task5: RetrosynthesisTask,
        policy: DictRxnCost,
        mol_syn_estimate: DictMolCost,
        mol_cost_estimate: DictMolCost,
    ) -> None:
        r"""
        Continuation of test above.

        For iterations 7-9 the value function of various other nodes will be evaluated
        without expansion. Then, on iteration 10 "C" will be expanded,
        leading to a new solution (O -> C).

        The tree should have the following structure:
                 ----------CC -------------------
                /              |                     \
              C                CO                    CS
            /  \               |                      |
           O    S   -----------------------          ...
                   /       /        \      \
                 C+O      OO         OS     CS
                           |
                  ---------|---------
                 /         |         \
                O         CO         OS
        """

        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task5,
            10,
            policy=policy,
            value_function_syn=mol_syn_estimate,
            value_function_cost=mol_cost_estimate,
        )
        assert output_graph.reaction_smiles_counter() == Counter(rxns_step5)  # type: ignore[attr-defined]

        # Check training data is correct
        training_data = pdvn_extract_training_data(output_graph)  # type: ignore[arg-type]  # doesn't understand it is AND/OR graph
        assert all(
            v == 0 or (mol.smiles in {"O", "OO", "CO", "CC", "C"} and v == 1)
            for mol, v in training_data.mol_to_synthesizability.items()
        )
        assert training_data.mol_to_min_syn_cost == {
            Molecule("O"): 0.0,
            Molecule("OO"): 1.0,
            Molecule("CO"): 2.0,
            Molecule("CC"): 2.0,
            Molecule("C"): 1.0,
        }
        assert len(training_data.mol_to_reactions_for_min_syn) == 4

    def test_by_hand_long_term(
        self,
        retrosynthesis_task5: RetrosynthesisTask,
        policy: DictRxnCost,
        mol_syn_estimate: DictMolCost,
        mol_cost_estimate: DictMolCost,
    ) -> None:
        r"""
        Continuation of test above.
        After running for a very long time, the algorithm should find the "true"
        synthesizability and cost of each node.
        """

        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task5,
            1_000,
            policy=policy,
            value_function_syn=mol_syn_estimate,
            value_function_cost=mol_cost_estimate,
        )

        # Check training data is correct
        training_data = pdvn_extract_training_data(output_graph)  # type: ignore[arg-type]  # doesn't understand it is AND/OR graph
        assert all(
            v == 1 for v in training_data.mol_to_synthesizability.values()
        )  # every molecule is synthesizable in this system
        assert training_data.mol_to_min_syn_cost == {
            # The only purchasable molecule
            Molecule("O"): 0.0,
            # All molecules distance 1 away from O
            Molecule("C"): 1.0,
            Molecule("S"): 1.0,
            Molecule("OO"): 1.0,
            # all molecules 2 away from O
            Molecule("CO"): 2.0,
            Molecule("CC"): 2.0,
            Molecule("OS"): 2.0,
            Molecule("SS"): 2.0,
            # only molecule 3 away from O
            Molecule("CS"): 3.0,
        }

        # Check that reactions are correct by comparing reaction SMILES
        mol_to_rxns_for_min_syn = {
            mol.smiles: {rxn.reaction_smiles for rxn in rxns}
            for mol, rxns in training_data.mol_to_reactions_for_min_syn.items()
        }
        assert mol_to_rxns_for_min_syn == {
            "C": {"O>>C"},
            "S": {"O>>S"},
            "OO": {"O>>OO"},
            "CO": {"C.O>>CO", "OO>>CO"},
            "CC": {
                "C>>CC",
            },
            "OS": {"OO>>OS", "O.S>>OS"},
            "SS": {"S>>SS"},
            "CS": {"SS>>CS", "C.S>>CS", "CO>>CS", "OS>>CS"},
        }

    def test_system_with_dead_ends(
        self,
        retrosynthesis_task5: RetrosynthesisTask,
    ) -> None:
        """Test PDVN MCTS on a system with dead end reactions."""

        rxn_model = ListOfReactionsToyModel(
            [
                SingleProductReaction(reactants=Bag({Molecule("C")}), product=Molecule("CC")),
                SingleProductReaction(reactants=Bag({Molecule("CO")}), product=Molecule("CC")),
                SingleProductReaction(reactants=Bag({Molecule("O")}), product=Molecule("C")),
            ],
            use_cache=True,
        )
        retrosynthesis_task = RetrosynthesisTask(
            inventory=retrosynthesis_task5.inventory,
            target_mol=retrosynthesis_task5.target_mol,
            reaction_model=rxn_model,
        )
        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task,
            1_000,
        )

        # Check training data is correct
        training_data = pdvn_extract_training_data(output_graph)  # type: ignore[arg-type]  # doesn't understand it is AND/OR graph
        assert training_data.mol_to_synthesizability == {
            Molecule("CC"): 1.0,
            Molecule("C"): 1.0,
            Molecule("O"): 1.0,
            Molecule("CO"): -1.0,
        }
