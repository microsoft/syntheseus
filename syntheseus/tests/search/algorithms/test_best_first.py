from __future__ import annotations

import math

import pytest

from syntheseus.search.algorithms.best_first.retro_star import (
    MolIsPurchasableCost,
    RetroStarSearch,
)
from syntheseus.search.analysis.solution_time import get_first_solution_time
from syntheseus.search.graph.and_or import AndNode, OrNode
from syntheseus.search.node_evaluation.base import NoCacheNodeEvaluator
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator
from syntheseus.tests.search.algorithms.test_base import BaseAlgorithmTest
from syntheseus.tests.search.conftest import RetrosynthesisTask


class DictMolCost(NoCacheNodeEvaluator[OrNode]):
    """Store a cost for each mol in a dictionary."""

    def __init__(self, smiles_to_cost: dict[str, float], default: float, **kwargs):
        super().__init__(**kwargs)
        self.mol_to_cost = smiles_to_cost
        self.default = default

    def _evaluate_nodes(self, nodes, graph=None):
        return [self.mol_to_cost.get(n.mol.smiles, self.default) for n in nodes]


class DictRxnCost(NoCacheNodeEvaluator[AndNode]):
    """Store a cost for each reaction in a dictionary."""

    def __init__(self, rxn_smiles_to_cost: dict[str, float], default: float, **kwargs):
        super().__init__(**kwargs)
        self.rxn_to_cost = rxn_smiles_to_cost
        self.default = default

    def _evaluate_nodes(self, nodes, graph=None):
        return [self.rxn_to_cost.get(n.reaction.reaction_smiles, self.default) for n in nodes]


@pytest.fixture
def rxn_cost_fn() -> DictRxnCost:
    """Return cost function for hand-worked example."""
    return DictRxnCost(
        rxn_smiles_to_cost={
            "C>>CC": 1.5,
            "CO>>CC": 0.5,
            "OO>>CO": 0.1,
            "O>>OO": 2.0,  # make cost sub-optimal
            "CO>>OO": 5.0,  # expensive to prevent further exploration
            "OS>>OO": 5.0,  # expensive to prevent further exploration
            "OS>>CO": 5.0,  # expensive to prevent further exploration
            "CS>>CO": 5.0,  # expensive to prevent further exploration
            "O>>C": 0.1,
        },
        default=1.0,
    )


@pytest.fixture
def mol_value_fn() -> DictMolCost:
    """Return cost function for hand-worked example."""
    return DictMolCost(
        smiles_to_cost={
            "C": 0.4,
            "OO": 0.0,
        },
        default=0.1,
    )


def assert_retro_star_values_in_graph(graph, values: list[float]):
    """Assert that the expected retro-star values are in the graph."""
    retro_star_values = [node.data["retro_star_value"] for node in graph.nodes()]
    for v in values:
        assert any(math.isclose(v, rv) for rv in retro_star_values)


class TestRetroStar(BaseAlgorithmTest):
    def setup_algorithm(self, **kwargs) -> RetroStarSearch:
        # Return retro*0 with a constant unit cost for each reaction
        kwargs.setdefault("and_node_cost_fn", ConstantNodeEvaluator(1.0))
        kwargs.setdefault("or_node_cost_fn", MolIsPurchasableCost())
        kwargs.setdefault("value_function", ConstantNodeEvaluator(0.0))
        return RetroStarSearch(**kwargs)

    def test_by_hand_step1(
        self,
        retrosynthesis_task5: RetrosynthesisTask,
        rxn_cost_fn: DictRxnCost,
        mol_value_fn: DictMolCost,
    ) -> None:
        r"""
        Test retro-star on a hand-designed example with a custom reaction cost/value function
        (the default cost/value functions will just act the same as breadth-first search).
        The example is designed to first explore a sub-optimal path (O -> OO -> CO -> CC)
        before finding the better path (O -> C -> CC).

        In the first step, the algorithm should expand the root node (CC),
        and the tree should have the following structure:

                 ----------CC (0.6/0.6)--------
                /(c=1.5)    | (c=0.5)          \ (c=1.0)
              C (0.4/1.9)   CO (0.1/0.6)        CS (0.1/1.1)

        (X/Y) denotes reaction number of X (min cost to purchase/synthesize)
        and retro-star value of Y (min cost of synthesis route including this node).

        (c=Z) denotes the cost cost of the reaction Z
        """
        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task5, 1, and_node_cost_fn=rxn_cost_fn, value_function=mol_value_fn
        )
        assert output_graph.reaction_smiles_counter() == {  # type: ignore  # unsure about rxn_counter
            "C>>CC": 1,
            "CO>>CC": 1,
            "CS>>CC": 1,
        }
        assert len(output_graph) == 7
        assert not output_graph.root_node.has_solution
        assert get_first_solution_time(output_graph) == math.inf
        assert math.isclose(output_graph.root_node.data["retro_star_value"], 0.6)
        assert math.isclose(output_graph.root_node.data["retro_star_reaction_number"], 0.6)
        assert math.isclose(output_graph.root_node.data["retro_star_min_cost"], math.inf)
        assert_retro_star_values_in_graph(output_graph, [1.9, 0.6, 1.1])

    def test_by_hand_step2(
        self,
        retrosynthesis_task5: RetrosynthesisTask,
        rxn_cost_fn: DictRxnCost,
        mol_value_fn: DictMolCost,
    ) -> None:
        r"""
        Continuation of test above (see docstring for notation).

        CO should be expanded next because it has the lowest retro-star value.

                 --------------------- CC (0.6/0.6)----------------
                /(c=1.5)                 | (c=0.5)                 \ (c=1.0)
              C (0.4/1.9)  -------------CO (0.1/0.6)-----------     CS (0.1/1.1)
                           /(c=1)       /(c=0.1)   \(c=5)      \(c=5)
                C (.4/1.9) + O (0/1.9)  OO (0.0/.6)  OS (0.1/5.6)  CS (0.1/5.6)

        (disclaimer: because the tree is large it is possible that some hand-calculated values are wrong)
        """
        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task5, 2, and_node_cost_fn=rxn_cost_fn, value_function=mol_value_fn
        )
        assert output_graph.reaction_smiles_counter() == {  # type: ignore  # unsure about rxn_counter
            "C>>CC": 1,
            "CO>>CC": 1,
            "CS>>CC": 1,
            "C.O>>CO": 1,
            "OO>>CO": 1,
            "OS>>CO": 1,
            "CS>>CO": 1,
        }
        assert len(output_graph) == 16
        assert not output_graph.root_node.has_solution
        assert get_first_solution_time(output_graph) == math.inf
        assert math.isclose(output_graph.root_node.data["retro_star_value"], 0.6)
        assert math.isclose(output_graph.root_node.data["retro_star_reaction_number"], 0.6)
        assert math.isclose(output_graph.root_node.data["retro_star_min_cost"], math.inf)
        assert_retro_star_values_in_graph(output_graph, [5.6, 1.1, 0.6, 1.9])

    def test_by_hand_step3(
        self,
        retrosynthesis_task5: RetrosynthesisTask,
        rxn_cost_fn: DictRxnCost,
        mol_value_fn: DictMolCost,
    ) -> None:
        r"""
        Continuation of test above (see docstring for notation).

        OO should be expanded next, yielding a solution.

                 --------------------- CC (1.1/1.1)----------------
                /(c=1.5)                 | (c=0.5)                 \ (c=1.0)
              C (0.4/1.9)  -------------CO (0.1/0.6)-----------     CS (0.1/1.1)
                           /(c=1)       /(c=0.1)   \(c=5)      \(c=5)
                C (.4/1.9) + O (0/1.9)  OO (0.0/.6)  OS (0.1/5.6)  CS (0.1/5.6)
                                         |
                                ---------|---------
                              / (c=2)    | (c=5)    \ (c=5)
                             O (0/2.6)  CO (.1/5.7) OS (.1/5.7)

        (disclaimer: because the tree is large it is possible that some hand-calculated values are wrong)
        """
        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task5, 3, and_node_cost_fn=rxn_cost_fn, value_function=mol_value_fn
        )
        assert output_graph.reaction_smiles_counter() == {  # type: ignore  # unsure about rxn_counter
            "C>>CC": 1,
            "CO>>CC": 1,
            "CS>>CC": 1,
            "C.O>>CO": 1,
            "OO>>CO": 1,
            "OS>>CO": 1,
            "CS>>CO": 1,
            "O>>OO": 1,
            "CO>>OO": 1,
            "OS>>OO": 1,
        }
        assert len(output_graph) == 22
        assert output_graph.root_node.has_solution
        assert get_first_solution_time(output_graph) == 3
        assert math.isclose(output_graph.root_node.data["retro_star_value"], 1.1)
        assert math.isclose(output_graph.root_node.data["retro_star_reaction_number"], 1.1)
        assert math.isclose(output_graph.root_node.data["retro_star_min_cost"], 2.6)
        assert_retro_star_values_in_graph(output_graph, [5.6, 1.1, 2.6, 1.9, 5.7])

    def test_by_hand_step4(
        self,
        retrosynthesis_task5: RetrosynthesisTask,
        rxn_cost_fn: DictRxnCost,
        mol_value_fn: DictMolCost,
    ) -> None:
        r"""
        Continuation of test above (see docstring for notation).

        CS should be expanded next because it has the lowest retro-star value.
        I did not draw the rest of the tree, but the retro star value should be 2.2

                 --------------------- CC (1.9/1.9)------------------------------
                /(c=1.5)                 | (c=0.5)                               \ (c=1.0)
              C (0.4/1.9)  -------------CO (0.1/2.6)-----------                  CS (*/2.1)
                           /(c=1)       /(c=0.1)   \(c=5)      \(c=5)             |
                C (.4/1.9) + O (0/1.9)  OO (2/2.6)  OS (0.1/5.6)  CS (0.1/5.6)    |
                                         |                                        |
                                ---------|---------                             ......
                              / (c=2)    | (c=5)    \ (c=5)
                             O (0/2.6)  CO (.1/5.7) OS (.1/5.7)

        (disclaimer: because the tree is large it is possible that some hand-calculated values are wrong)
        """
        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task5, 4, and_node_cost_fn=rxn_cost_fn, value_function=mol_value_fn
        )
        assert output_graph.reaction_smiles_counter() == {  # type: ignore  # unsure about rxn_counter
            "C>>CC": 1,
            "CO>>CC": 1,
            "CS>>CC": 1,
            "C.O>>CO": 1,
            "OO>>CO": 1,
            "OS>>CO": 1,
            "CS>>CO": 1,
            "O>>OO": 1,
            "CO>>OO": 1,
            "OS>>OO": 1,
            "C.S>>CS": 1,
            "OS>>CS": 1,
            "SS>>CS": 1,
            "CO>>CS": 1,
        }
        assert len(output_graph) == 31
        assert output_graph.root_node.has_solution
        assert get_first_solution_time(output_graph) == 3
        assert math.isclose(output_graph.root_node.data["retro_star_value"], 1.9)
        assert math.isclose(output_graph.root_node.data["retro_star_reaction_number"], 1.9)
        assert math.isclose(output_graph.root_node.data["retro_star_min_cost"], 2.6)
        assert_retro_star_values_in_graph(output_graph, [1.9, 5.6, 2.6, 2.1, 5.7])

    def test_by_hand_step5(
        self,
        retrosynthesis_task5: RetrosynthesisTask,
        rxn_cost_fn: DictRxnCost,
        mol_value_fn: DictMolCost,
    ) -> None:
        r"""
        Continuation of test above (see docstring for notation).

        C should now be expanded because it has the lowest retro-star value of 1.9,
        yielding a route with cost 1.6 (the minimum possible).

                 --------------------- CC (1.9/1.9)------------------------------
                /(c=1.5)                 | (c=0.5)                               \ (c=1.0)
              C (0.4/1.6)               CO (*/2.6)                              CS (*/2.1)
             -|--------                  |                                       |
            /(c=.1)    \(c=1)           ...                                     ...
          O (0/1.6)     S (0.1/2.6)

        (disclaimer: because the tree is large it is possible that some hand-calculated values are wrong)
        """
        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task5, 5, and_node_cost_fn=rxn_cost_fn, value_function=mol_value_fn
        )
        assert output_graph.reaction_smiles_counter() == {  # type: ignore  # unsure about rxn_counter
            "C>>CC": 1,
            "CO>>CC": 1,
            "CS>>CC": 1,
            "C.O>>CO": 1,
            "OO>>CO": 1,
            "OS>>CO": 1,
            "CS>>CO": 1,
            "O>>OO": 1,
            "CO>>OO": 1,
            "OS>>OO": 1,
            "C.S>>CS": 1,
            "OS>>CS": 1,
            "SS>>CS": 1,
            "CO>>CS": 1,
            "O>>C": 1,
            "S>>C": 1,
        }
        assert len(output_graph) == 35
        assert output_graph.root_node.has_solution
        assert get_first_solution_time(output_graph) == 3
        assert math.isclose(output_graph.root_node.data["retro_star_value"], 1.6)
        assert math.isclose(output_graph.root_node.data["retro_star_reaction_number"], 1.6)
        assert math.isclose(output_graph.root_node.data["retro_star_min_cost"], 1.6)
        assert_retro_star_values_in_graph(output_graph, [1.6, 2.6, 2.1])
