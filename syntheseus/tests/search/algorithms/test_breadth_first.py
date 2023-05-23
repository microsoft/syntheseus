from __future__ import annotations

import math
from collections import Counter

from syntheseus.search.algorithms.breadth_first import (
    AndOr_BreadthFirstSearch,
    MolSet_BreadthFirstSearch,
)
from syntheseus.search.analysis.solution_time import get_first_solution_time
from syntheseus.tests.search.algorithms.test_base import BaseAlgorithmTest
from syntheseus.tests.search.conftest import RetrosynthesisTask


class TestAndOrBreadthFirst(BaseAlgorithmTest):
    def setup_algorithm(self, **kwargs) -> AndOr_BreadthFirstSearch:
        return AndOr_BreadthFirstSearch(**kwargs)

    def test_by_hand_step1(self, retrosynthesis_task5: RetrosynthesisTask) -> None:
        r"""
        Test that behaviour of algorithm matches hand-calculated outcome.
        Should expand the root node into the following 3 reactions:

                                CC
                            /    |   \
                        C (+C)   CO  CS
        """
        output_graph = self.run_alg_for_n_iterations(retrosynthesis_task5, 1)
        assert len(output_graph) == 7  # should expand first node into 3 reactions with 1 product
        assert not output_graph.root_node.has_solution  # should not be solved yet
        assert get_first_solution_time(output_graph) == math.inf
        assert output_graph.smiles_counter() == {"CC": 1, "C": 1, "CO": 1, "CS": 1}  # type: ignore  # mol_counter

    def test_by_hand_step2(self, retrosynthesis_task5: RetrosynthesisTask) -> None:
        r"""
        Continuing from previous test: should expand the C node to form the following graph:

                                CC
                            /    |   \
                        C (+C)   CO  CS
                       /    \
                       O     S

        Because "O" is purchasable, a solution should be found.
        """
        output_graph = self.run_alg_for_n_iterations(retrosynthesis_task5, 2)
        assert len(output_graph) == 11  # should expand first node into 3 reactions with 1 product
        assert output_graph.root_node.has_solution  # should now be solved
        assert get_first_solution_time(output_graph) == 2
        assert output_graph.smiles_counter() == {  # type: ignore  # mol_counter
            "CC": 1,
            "C": 1,
            "CO": 1,
            "CS": 1,
            "O": 1,
            "S": 1,
        }

    def test_by_hand_step3(self, retrosynthesis_task5: RetrosynthesisTask) -> None:
        r"""
        Continuing from previous test: should expand "CO" since it was next in the queue.

                           ----------------CC ------------
                          /                |              \
                      C (+C)      -------- CO --           CS
                     /    \      /     /   |    \
                     O     S   C + O   OO  OS    CS

        Note that the reaction "CC -> CO" should *not* be included because it reproduces the root mol
        and should therefore be filtered out.
        """
        output_graph = self.run_alg_for_n_iterations(retrosynthesis_task5, 3)
        assert len(output_graph) == 20
        assert get_first_solution_time(output_graph) == 2
        assert output_graph.smiles_counter() == {  # type: ignore  # mol_counter
            "CC": 1,
            "C": 2,
            "CO": 1,
            "CS": 2,
            "O": 2,
            "S": 1,
            "OO": 1,
            "OS": 1,
        }

    def test_by_hand_step4(self, retrosynthesis_task5: RetrosynthesisTask) -> None:
        r"""
        Continuing from previous test: should expand CS

                           ----------------CC ---------------------
                          /                |                       \
                      C (+C)      -------- CO --             ------CS-----
                     /    \      /     /   |    \           /      |   \   \
                     O     S   C + O   OO  OS    CS       C + S    OS  SS   CO

        Note that the reaction "CC -> CO" should *not* be included because it reproduces the root mol
        and should therefore be filtered out.
        """
        output_graph = self.run_alg_for_n_iterations(retrosynthesis_task5, 4)
        assert len(output_graph) == 29  # should expand first node into 3 reactions with 1 product
        assert get_first_solution_time(output_graph) == 2
        assert output_graph.smiles_counter() == {  # type: ignore  # mol_counter
            "CC": 1,
            "C": 3,
            "CO": 2,
            "CS": 2,
            "O": 2,
            "S": 2,
            "OO": 1,
            "OS": 2,
            "SS": 1,
        }

    def test_by_hand_step5(self, retrosynthesis_task5: RetrosynthesisTask) -> None:
        r"""
        Continuing from previous test: because "O" is purchasable it should not be visited, so "S"
        should be the next thing expanded

                           ----------------CC ---------------------
                          /                |                       \
                      C (+C)      -------- CO --             ------CS-----
                     /    \      /     /   |    \           /      |   \   \
                     O     S   C + O   OO  OS    CS       C + S    OS  SS   CO
                         /  \
                        C    O
        """
        output_graph = self.run_alg_for_n_iterations(retrosynthesis_task5, 5)
        assert len(output_graph) == 33  # should expand first node into 3 reactions with 1 product
        assert get_first_solution_time(output_graph) == 2
        assert output_graph.smiles_counter() == {  # type: ignore  # mol_counter
            "CC": 1,
            "C": 4,
            "CO": 2,
            "CS": 2,
            "O": 3,
            "S": 2,
            "OO": 1,
            "OS": 2,
            "SS": 1,
        }


# Answers for hand-crafted example below
MOLSET_NODES_STEP1: dict[tuple[str, ...], int] = {("CC",): 1, ("C",): 1, ("CO",): 1, ("CS",): 1}
MOLSET_NODES_STEP2 = Counter(MOLSET_NODES_STEP1)
MOLSET_NODES_STEP2.update({("O",): 1, ("S",): 1})
MOLSET_NODES_STEP3 = Counter(MOLSET_NODES_STEP2)
MOLSET_NODES_STEP3.update({("C", "O"): 1, ("OO",): 1, ("OS",): 1, ("CS",): 1})
MOLSET_NODES_STEP4 = Counter(MOLSET_NODES_STEP3)
MOLSET_NODES_STEP4.update({("C", "S"): 1, ("OS",): 1, ("SS",): 1, ("CO",): 1})
MOLSET_NODES_STEP5 = Counter(MOLSET_NODES_STEP4)
MOLSET_NODES_STEP5.update({("C",): 1, ("O",): 1})
MOLSET_NODES_STEP6 = Counter(MOLSET_NODES_STEP5)
MOLSET_NODES_STEP6.update({("O",): 1, ("O", "S"): 1})
MOLSET_NODES_STEP10 = Counter(MOLSET_NODES_STEP6)
MOLSET_NODES_STEP10.update({("OS",): 1, ("O",): 1, ("CO",): 1})  # step 7 (OO)
MOLSET_NODES_STEP10.update(
    {("OO",): 1, ("CS",): 1, ("SS",): 1, ("CO",): 1, ("O", "S"): 1}
)  # step 8 (OS)
MOLSET_NODES_STEP10.update({("OS",): 1, ("SS",): 1, ("CO",): 1, ("C", "S"): 1})  # step 9 (CS)
MOLSET_NODES_STEP10.update({("O", "S"): 1, ("S",): 1, ("C", "O"): 1, ("C",): 1})  # step 10 (C, S)


class TestMolSetBreadthFirst(BaseAlgorithmTest):
    def setup_algorithm(self, **kwargs) -> MolSet_BreadthFirstSearch:
        return MolSet_BreadthFirstSearch(**kwargs)

    def test_by_hand_step1(self, retrosynthesis_task5: RetrosynthesisTask) -> None:
        r"""
        Test that behaviour of algorithm matches hand-calculated outcome.
        Should expand the root node into the following 3 reactions:

                                {CC}
                            /    |   \
                           {C}  {CO} {CS}

        NOTE: for the tests below I did not make visualizations because they are similar
        to the test above.
        """
        output_graph = self.run_alg_for_n_iterations(retrosynthesis_task5, 1)
        assert not output_graph.root_node.has_solution  # should not be solved yet
        assert get_first_solution_time(output_graph) == math.inf
        assert output_graph.smiles_set_counter() == MOLSET_NODES_STEP1  # type: ignore
        assert len(output_graph) == 4

    def test_by_hand_step2(self, retrosynthesis_task5: RetrosynthesisTask) -> None:
        """
        Continuing from previous test: should expand the C node and solve the task.
        """

        output_graph = self.run_alg_for_n_iterations(retrosynthesis_task5, 2)
        assert output_graph.root_node.has_solution  # should be solved now
        assert get_first_solution_time(output_graph) == 2
        assert output_graph.smiles_set_counter() == MOLSET_NODES_STEP2  # type: ignore
        assert len(output_graph) == 6

    def test_by_hand_step3(self, retrosynthesis_task5: RetrosynthesisTask) -> None:
        """
        Continuing from previous test: should expand CO, and now produce a node with 2 molecules.
        """
        output_graph = self.run_alg_for_n_iterations(retrosynthesis_task5, 3)
        assert output_graph.smiles_set_counter() == MOLSET_NODES_STEP3  # type: ignore
        assert len(output_graph) == 10
        assert get_first_solution_time(output_graph) == 2

    def test_by_hand_step4(self, retrosynthesis_task5: RetrosynthesisTask) -> None:
        """
        Continuing from previous test: should expand CS.
        """
        output_graph = self.run_alg_for_n_iterations(retrosynthesis_task5, 4)
        assert output_graph.smiles_set_counter() == MOLSET_NODES_STEP4  # type: ignore
        assert len(output_graph) == 14
        assert get_first_solution_time(output_graph) == 2

    def test_by_hand_step5(self, retrosynthesis_task5: RetrosynthesisTask) -> None:
        """
        Continuing from previous test: expand "S"
        """
        output_graph = self.run_alg_for_n_iterations(retrosynthesis_task5, 5)
        assert output_graph.smiles_set_counter() == MOLSET_NODES_STEP5  # type: ignore
        assert len(output_graph) == 16
        assert get_first_solution_time(output_graph) == 2

    def test_by_hand_step6(self, retrosynthesis_task5: RetrosynthesisTask) -> None:
        """
        First step where the algorithm should expand a node with 2 molecules: {C, O}.
        However, since O is purchasable it should not be expanded.
        """
        output_graph = self.run_alg_for_n_iterations(retrosynthesis_task5, 6)
        assert output_graph.smiles_set_counter() == MOLSET_NODES_STEP6  # type: ignore
        assert len(output_graph) == 18
        assert get_first_solution_time(output_graph) == 2

    def test_by_hand_step10(self, retrosynthesis_task5: RetrosynthesisTask) -> None:
        """
        Jump ahead to step 10, where the algorithm should exapnd the node {C, S},
        which has 2 non-purchasable molecules.
        """
        output_graph = self.run_alg_for_n_iterations(retrosynthesis_task5, 10)
        assert output_graph.smiles_set_counter() == MOLSET_NODES_STEP10  # type: ignore
        assert get_first_solution_time(output_graph) == 2
