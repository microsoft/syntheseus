from __future__ import annotations

import math
import random
import warnings
from collections import Counter

import pytest

from syntheseus.search.algorithms.mcts.base import pucb_bound, random_argmin
from syntheseus.search.algorithms.mcts.molset import MolSetMCTS
from syntheseus.search.analysis.solution_time import get_first_solution_time
from syntheseus.search.graph.molset import MolSetNode
from syntheseus.search.node_evaluation.base import NoCacheNodeEvaluator
from syntheseus.search.node_evaluation.common import (
    ConstantNodeEvaluator,
    HasSolutionValueFunction,
)
from syntheseus.tests.search.algorithms.test_base import BaseAlgorithmTest
from syntheseus.tests.search.conftest import RetrosynthesisTask


class TestRandomArgmin:
    # Lists with probabilities of different answers
    random_test_cases: list[tuple[list[float], dict[int, float]]] = [
        ([1.0, 2.0, 1.0], {0: 0.5, 2: 0.5}),  # 50/50 each
        ([1.0], {0: 1.0}),  # just 1 element
        ([2.0, 1.0, 3.0, 1.0, 1.0], {1: 1 / 3, 3: 1 / 3, 4: 1 / 3}),  # 1/3 chance each
        (
            [1.0, 1 + 1e-4, 1.0],
            {0: 0.5, 2: 0.5},
        ),  # the middle element is still significantly different
        ([-math.inf, 3.0, -math.inf], {0: 0.5, 2: 0.5}),  # 2 infs, should both be min
    ]

    def test_deterministic(self) -> None:
        """Test some deterministic random argmin."""
        assert random_argmin([2.0, 1.0, 3.0]) == 1
        assert random_argmin([-1.0]) == 0
        assert random_argmin([3.0, 2.0, 1.0, 0.0]) == 3
        assert random_argmin([-math.inf, 2.0, 1.0, 0.0]) == 0  # No problem with infs

    def test_errors(self) -> None:
        """Raise various errors."""
        with pytest.raises(ValueError):
            random_argmin([])
        with pytest.raises(ValueError):
            random_argmin([math.nan])
        with pytest.raises(ValueError):
            random_argmin([math.nan, 1.0, 3.0])
        with pytest.raises(ValueError):
            random_argmin([-math.inf, math.nan, 1.0, 3.0])

    def test_random(self) -> None:
        """Do a few tests where the correct answer is random."""
        N = 1_000  # how many repeats of the test to do
        for lst, answers in self.random_test_cases:
            outputs = [random_argmin(lst) for _ in range(N)]
            counter = Counter(outputs)
            answer_fractions = {k: v / N for k, v in counter.items()}
            assert answer_fractions.keys() == answers.keys()
            for k in answers.keys():
                assert math.isclose(answer_fractions[k], answers[k], abs_tol=9e-2)


class DictMolSetCost(NoCacheNodeEvaluator[MolSetNode]):
    """Stores a custom cost for each molset."""

    def __init__(self, smiles_set_to_cost: dict[frozenset[str], float], default: float, **kwargs):
        super().__init__(**kwargs)
        self.mols_to_cost = smiles_set_to_cost
        self.default = default

    def _evaluate_nodes(self, nodes, graph=None):
        return [
            self.mols_to_cost.get(frozenset([mol.smiles for mol in n.mols]), self.default)
            for n in nodes
        ]


@pytest.fixture
def value_fn() -> DictMolSetCost:
    """Return value function for hand-worked example."""
    return DictMolSetCost(
        smiles_set_to_cost={
            frozenset({"C"}): 0.3,
            frozenset({"CO"}): 0.9,
            frozenset({"CS"}): 0.1,
            frozenset({"C", "O"}): 0.1,
            frozenset({"OO"}): 0.1,
            frozenset({"OS"}): 0.1,
        },
        default=0.5,
    )


HAND_EXAMPLE_BOUND_CONSTANT = 1.0


class TestMolSetMCTS(BaseAlgorithmTest):
    time_limit_upper_bound_s = 0.1  # runtime is variable for some reason

    def setup_algorithm(
        self,
        reward_function=None,
        value_function=None,
        bound_constant: float = 1e4,  # high default to ensure lots of exploring
        **kwargs,
    ) -> MolSetMCTS:
        return MolSetMCTS(
            reward_function=reward_function or HasSolutionValueFunction(),
            value_function=value_function or ConstantNodeEvaluator(0.5),
            bound_constant=bound_constant,
            random_state=random.Random(42),  # control random seed
            **kwargs,
        )

    def test_by_hand_uct_step1(
        self, retrosynthesis_task5: RetrosynthesisTask, value_fn: DictMolSetCost
    ) -> None:
        r"""
        Test that behaviour of algorithm matches hand-calculated outcome.
        Should expand the root node into the following 3 reactions:

                                {CC}
                            /    |   \
                           {C}  {CO} {CS}

        NOTE: for the tests below I did not make visualizations because they are similar
        to the tests from breadth-first search.
        """
        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task5,
            1,
            value_function=value_fn,
            bound_constant=HAND_EXAMPLE_BOUND_CONSTANT,
        )
        assert not output_graph.root_node.has_solution  # should not be solved yet
        assert get_first_solution_time(output_graph) == math.inf
        assert output_graph.smiles_set_counter() == {("CC",): 1, ("C",): 1, ("CO",): 1, ("CS",): 1}  # type: ignore
        assert len(output_graph) == 4
        assert output_graph.root_node.num_visit == 2  # 1 + 1 pseudo-visit before MCTS starts

    def test_by_hand_step3(
        self, retrosynthesis_task5: RetrosynthesisTask, value_fn: DictMolSetCost
    ) -> None:
        """
        Continuation of test above. The next 2 iterations should visit the
        remaining 2 children nodes without expanding them. The reward should
        be the average of initial visit (0) and values of other 3 nodes.
        """
        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task5,
            3,
            value_function=value_fn,
            bound_constant=HAND_EXAMPLE_BOUND_CONSTANT,
        )
        assert not output_graph.root_node.has_solution  # should not be solved yet
        assert get_first_solution_time(output_graph) == math.inf
        assert output_graph.smiles_set_counter() == {("CC",): 1, ("C",): 1, ("CO",): 1, ("CS",): 1}  # type: ignore
        assert math.isclose(
            output_graph.root_node.data["mcts_value"],
            (0 + 0.3 + 0.9 + 0.1) / 4,
        )
        assert output_graph.root_node.num_visit == 4

    def test_by_hand_step4(
        self, retrosynthesis_task5: RetrosynthesisTask, value_fn: DictMolSetCost
    ) -> None:
        """
        Continuation of test above. Next iteration should expand "CO"
        since that node has the highest value. All its children have
        a reward of 0.1 so that should be the reward for this iteration.
        """
        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task5,
            4,
            value_function=value_fn,
            bound_constant=HAND_EXAMPLE_BOUND_CONSTANT,
        )
        assert len(output_graph) == 8
        assert not output_graph.root_node.has_solution  # should not be solved yet
        assert get_first_solution_time(output_graph) == math.inf
        assert math.isclose(
            output_graph.root_node.data["mcts_value"],
            (0 + 0.3 + 0.9 + 0.1 + 0.1) / 5,
        )
        assert math.isclose(output_graph.root_node.data["mcts_prev_reward"], 0.1)
        assert output_graph.root_node.num_visit == 5

    def test_by_hand_step5(
        self, retrosynthesis_task5: RetrosynthesisTask, value_fn: DictMolSetCost
    ) -> None:
        """
        Continuation of test above. Due to low reward it should next expand
        the "C" node at depth 1, which will result in a solution.
        """
        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task5,
            5,
            value_function=value_fn,
            bound_constant=HAND_EXAMPLE_BOUND_CONSTANT,
        )
        assert len(output_graph) == 10
        assert output_graph.root_node.has_solution
        assert get_first_solution_time(output_graph) == 3
        assert math.isclose(
            output_graph.root_node.data["mcts_value"],
            (0 + 0.3 + 0.9 + 0.2 + 1) / 6,
        )
        assert math.isclose(output_graph.root_node.data["mcts_prev_reward"], 1.0)
        assert output_graph.root_node.num_visit == 6

    @pytest.mark.parametrize("n", [1, 2, 5])
    def test_num_visit_to_expand(
        self, retrosynthesis_task5: RetrosynthesisTask, value_fn: DictMolSetCost, n: int
    ) -> None:
        """
        Test that the algorithm will visit nodes `num_visit_to_expand` times
        before expanding them.

        To do this, we use the hand-calculated example above and check that if
        `num_visit_to_expand=n` then the algorithm will visit the root node
        (n-1) times before expanding it, then visit the 3 children nodes
        n times each without expanding.
        """
        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task5,
            4 * n - 1,
            value_function=value_fn,
            min_num_visit_to_expand=n,
            bound_constant=1e3,  # higher constant so that rewards don't matter
        )
        assert not output_graph.root_node.has_solution  # should not be solved yet
        assert get_first_solution_time(output_graph) == math.inf
        assert output_graph.smiles_set_counter() == {("CC",): 1, ("C",): 1, ("CO",): 1, ("CS",): 1}  # type: ignore
        correct_num_visit = 4 * n  # 1 extra visit from start
        assert output_graph.root_node.num_visit == correct_num_visit
        assert math.isclose(
            output_graph.root_node.data["mcts_value"],
            (0 + n * (0.3 + 0.9 + 0.1) + (n - 1) * value_fn.default) / correct_num_visit,
        )

    def test_pucb_bound_function_fails(
        self,
        retrosynthesis_task5: RetrosynthesisTask,
        value_fn: DictMolSetCost,
    ) -> None:
        """Test that the P-UCB bound function fails if no policy is provided."""

        with pytest.raises(RuntimeError):
            self.run_alg_for_n_iterations(
                retrosynthesis_task5,
                10,
                value_function=value_fn,
                bound_function=pucb_bound,
            )

    def test_pucb_bound_function(
        self,
        retrosynthesis_task5: RetrosynthesisTask,
        value_fn: DictMolSetCost,
    ) -> None:
        """
        Test that the P-UCB bound function works as expected on the hand-calculated example.
        With a low enough score for the policy, the algorithm should expand the "C" node
        instead of the "CO" node.

        After many iterations, the tree should look like:

                    {CC}
                /    |    \
              {C}   {CO}   {CS}  (not expanded due to policy)
            /    \
           {O}   {S}  (not expanded due to max expansion depth)
        """

        # Assigns a really small value to all first-level nodes except {"C"}
        policy = DictMolSetCost(
            smiles_set_to_cost={
                frozenset({"CO"}): 1e-4,
                frozenset({"CS"}): 1e-4,
            },
            default=1.0,
        )

        # Run algorithm with high bound constant so rewards have less effect
        # than policy scores
        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task5,
            50,
            value_function=value_fn,
            bound_function=pucb_bound,
            policy=policy,
            bound_constant=10.0,
            max_expansion_depth=2,  # limit so it doesn't expand {S} node.
        )

        # Check that only "C" node was expanded
        assert output_graph.smiles_set_counter() == {("CC",): 1, ("C",): 1, ("CO",): 1, ("CS",): 1, ("O",): 1, ("S",): 1}  # type: ignore
        assert output_graph.root_node.has_solution
        assert get_first_solution_time(output_graph) == 2

    def test_run_limit_warning(
        self,
        retrosynthesis_task5: RetrosynthesisTask,
    ) -> None:
        """Test that a warning is raised on __init__ if the algorithm is set up to possibly run forever."""

        kwargs = dict(
            reaction_model=retrosynthesis_task5.reaction_model,
            mol_inventory=retrosynthesis_task5.inventory,
        )

        # No time limit of any sort
        with pytest.warns(UserWarning):
            self.setup_algorithm(
                time_limit_s=math.inf,
                limit_iterations=1e100,
                limit_reaction_model_calls=1e100,
                **kwargs,  # type: ignore
            )

        # Just limit reaction model calls (could still run forever)
        with pytest.warns(UserWarning):
            self.setup_algorithm(
                time_limit_s=math.inf,
                limit_iterations=1e100,
                limit_reaction_model_calls=10,
                **kwargs,  # type: ignore
            )

        # Set limits and see no warning (by catching them as errors)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("error")
            self.setup_algorithm(time_limit_s=1.0, limit_iterations=1e100, **kwargs)  # type: ignore
            self.setup_algorithm(time_limit_s=1.0, limit_iterations=1e100, **kwargs)  # type: ignore
