from __future__ import annotations

import abc
import datetime
import math
import warnings

import pytest

from syntheseus.interface.molecule import Molecule
from syntheseus.search import INT_INF
from syntheseus.search.algorithms.base import SearchAlgorithm
from syntheseus.search.analysis.route_extraction import iter_routes_cost_order
from syntheseus.search.analysis.solution_time import get_first_solution_time
from syntheseus.search.graph.and_or import AndNode, AndOrGraph, OrNode
from syntheseus.search.graph.base_graph import RetrosynthesisSearchGraph
from syntheseus.search.graph.message_passing import has_solution_update, run_message_passing
from syntheseus.search.graph.molset import MolSetGraph, MolSetNode
from syntheseus.search.graph.route import SynthesisGraph
from syntheseus.search.mol_inventory import SmilesListInventory
from syntheseus.tests.search.conftest import RetrosynthesisTask


class BaseAlgorithmTest(abc.ABC):
    """
    Base class which defines many common tests for algorithms.

    Tests for individual algorithms should inherit from this class.
    """

    # Some tolerances for tests
    time_limit_upper_bound_s = 0.05
    time_limit_multiplier = 1.0  # can be overridden by subclasses for slower algorithms

    @abc.abstractmethod
    def setup_algorithm(self, **kwargs) -> SearchAlgorithm:
        """Set up the algorithm with required kwargs, filling in sensible defaults where needed."""
        pass

    def run_alg_for_n_iterations(
        self, task: RetrosynthesisTask, n_iterations: int, **alg_kwargs
    ) -> RetrosynthesisSearchGraph:
        """Convenience method to run an algorithm for exactly n iterations and return the output graph."""

        # Run algorithm
        alg = self.setup_algorithm(
            reaction_model=task.reaction_model,
            mol_inventory=task.inventory,
            limit_iterations=n_iterations,
            **alg_kwargs,
        )
        output_graph, _ = alg.run_from_mol(task.target_mol)

        # For convenience of tests, populate the analysis_time field with num calls to rxn model
        for node in output_graph.nodes():
            node.data["analysis_time"] = node.data["num_calls_rxn_model"]
        return output_graph

    def test_smoke(self, retrosynthesis_task1: RetrosynthesisTask) -> None:
        """
        Basic "smoke test" that the algorithm runs and solves a very basic retrosynthesis task.
        """

        alg = self.setup_algorithm(
            reaction_model=retrosynthesis_task1.reaction_model,
            mol_inventory=retrosynthesis_task1.inventory,
            time_limit_s=0.1 * self.time_limit_multiplier,
            limit_iterations=1_000,
            limit_reaction_model_calls=10,
        )
        output_graph, _ = alg.run_from_mol(retrosynthesis_task1.target_mol)

        # Check that it is solved after 1 call to reaction model
        assert output_graph.root_node.has_solution
        for node in output_graph.nodes():
            node.data["analysis_time"] = node.data["num_calls_rxn_model"]
        solution_time = get_first_solution_time(output_graph)
        assert solution_time == 1

    def test_smoke2(self, retrosynthesis_task4: RetrosynthesisTask) -> None:
        """
        A second "smoke test": the algorithm should be able to fully expand and solve a small
        finite task in a reasonable number of iterations.
        """

        alg = self.setup_algorithm(
            reaction_model=retrosynthesis_task4.reaction_model,
            mol_inventory=retrosynthesis_task4.inventory,
            limit_iterations=10_000,
            time_limit_s=0.1 * self.time_limit_multiplier,
        )
        output_graph, _ = alg.run_from_mol(retrosynthesis_task4.target_mol)
        assert output_graph.root_node.has_solution

    @pytest.mark.parametrize("time_limit_s", [0.03, 0.1])
    def test_time_limit_s(
        self, retrosynthesis_task1: RetrosynthesisTask, time_limit_s: float
    ) -> None:
        """
        Test that the time limit works by checking whether the algorithm runs until the time limit.

        Unfortunately this test seems very prone to failure, so we use a very generous tolerance,
        but give a warning if the test fails at a lower tolerance.
        """

        # Run and time the algorithm
        alg = self.setup_algorithm(
            reaction_model=retrosynthesis_task1.reaction_model,
            mol_inventory=retrosynthesis_task1.inventory,
            time_limit_s=time_limit_s,
            limit_iterations=INT_INF,
            limit_reaction_model_calls=INT_INF,
        )
        start = datetime.datetime.now()
        alg.run_from_mol(retrosynthesis_task1.target_mol)
        end = datetime.datetime.now()
        time_elapsed = (end - start).total_seconds()

        # Test 1: it should not run under the time limit,
        # and should at most run 0.1s over
        assert time_limit_s - 0.005 <= time_elapsed <= time_limit_s + self.time_limit_upper_bound_s

        # Test 2: stricter test with lower tolerance (which only gives a warning)
        if not (time_limit_s - 0.001 <= time_elapsed <= time_limit_s + 0.03):
            warnings.warn(
                f"Warning: test_time_limit_s took {time_elapsed} s with a budget of {time_limit_s} s."
                " Consider re-running the test to ensure this is just a random failure."
            )

    @pytest.mark.parametrize("limit", [0, 1, 5])
    def test_limit_reaction_model_calls(
        self, retrosynthesis_task1: RetrosynthesisTask, limit: int
    ) -> None:
        """
        Test that the reaction model call limit works by running the algorithm and
        checking that it calls the reaction model exactly as many times as the limit.
        """

        # Run algorithm
        alg = self.setup_algorithm(
            reaction_model=retrosynthesis_task1.reaction_model,
            mol_inventory=retrosynthesis_task1.inventory,
            time_limit_s=1e3
            * self.time_limit_multiplier,  # set a finite but extremely large limit to avoid warnings
            limit_iterations=INT_INF,
            limit_reaction_model_calls=limit,
        )
        output_graph, _ = alg.run_from_mol(retrosynthesis_task1.target_mol)

        # Check max rxn model calls
        if isinstance(output_graph, MolSetGraph):
            # Be less strict: the algorithm may have called the reaction model very slightly more than the limit.
            # This can happen if the last node to expand has more more than one molecule to be expanded
            # (e.g. 9/10 budget used up and the last node has 2 molecules to expand)
            assert alg.reaction_model.num_calls() in {limit, limit + 1}
        else:
            # Default case is to match exactly
            assert alg.reaction_model.num_calls() == limit

    @pytest.mark.parametrize("limit", [0, 1, 2, 25, 100])
    def test_limit_graph_nodes(
        self,
        retrosynthesis_task6: RetrosynthesisTask,
        limit: int,
    ) -> None:
        """
        Test that limiting the number of nodes in the graph works as intended.
        The algorithm should run until the node limit is reached, and then stop
        (without adding *too many* extra nodes).

        `retrosynthesis_task6` is chosen because it can create a very large graph.
        """

        # Run algorithm
        alg = self.setup_algorithm(
            reaction_model=retrosynthesis_task6.reaction_model,
            mol_inventory=retrosynthesis_task6.inventory,
            limit_graph_nodes=limit,
            limit_iterations=int(1e6),  # a very high limit, but avoids MCTS warnings
        )
        output_graph, _ = alg.run_from_mol(retrosynthesis_task6.target_mol)

        # The algorithm will stop running when the graph size meets or exceeds the limit.
        # However, since multiple nodes are added during each expansion, the node count might
        # not exactly equal the limit. Therefore, we choose a variable tolerance.
        # "Tolerance" here is len(graph) - limit
        if limit == 0:
            tolerance = 1  # will stop search immediately with only the root node
        elif limit == 1:
            tolerance = 0  # should not expand root node
        elif limit == 2:
            tolerance = 19  # a very high number, since first expansion brings node count to 21 for AND/OR graphs
        else:
            tolerance = 20  # a fairly high tolerance (should always be enough for one expansion)

        # The actual test
        assert limit <= len(output_graph) <= tolerance + limit

    @pytest.mark.parametrize("limit", [0, 1, 2, 100])
    def test_limit_iterations(
        self,
        retrosynthesis_task1: RetrosynthesisTask,
        retrosynthesis_task2: RetrosynthesisTask,
        retrosynthesis_task3: RetrosynthesisTask,
        limit: int,
    ) -> None:
        """
        Because an iteration can be different for each algorithm, this test is more indirect:
        it sets the iteration limit to different values then runs the algorithm and checks
        that nothing happened which would not be possible in the given number of iterations.
        """

        # Run each algorithm on all the tasks
        output_graphs = [
            self.run_alg_for_n_iterations(task, limit)
            for task in [retrosynthesis_task1, retrosynthesis_task2, retrosynthesis_task3]
        ]

        # Check whether each task was solved, and if this matched expectations
        tasks_solved = [g.root_node.has_solution for g in output_graphs]
        if limit == 0:
            # nothing should have happened, so none of the tasks should be solved
            assert tasks_solved == [False, False, False]
        elif limit == 1:
            # The first task should be solved since the first iteration of all algorithms should be to expand the root node
            # The other 2 tasks should not be solved though
            assert tasks_solved == [True, False, False]

        elif limit == 2:
            # The first task should be solved and the last task should not be solved. Second task may or may not be solved.
            assert tasks_solved[0]
            assert not tasks_solved[2]

        elif limit >= 100:
            # With this high number of iterations all tasks should be solved
            assert tasks_solved == [True, True, True]

        else:
            # this is just to make sure no tests are skipped accidentally by raising an error if the limit list
            # is changed without adding a corresponding elif clause here
            raise ValueError(f"Case of limit={limit} is not handled.")

    @pytest.mark.parametrize("depth", [0, 2, 4])
    def test_max_expansion_depth(
        self, retrosynthesis_task1: RetrosynthesisTask, depth: int
    ) -> None:
        """
        Test that the 'max_expansion_depth' argument actually prevents nodes which are too deep from being expanded.

        To test this, we choose a system where circular reactions can be done (C -> O -> C -> O -> ...),
        meaning that the full search tree is infinitely deep. The algorithm is allowed to run for a long time
        so that it may fully expand the tree up to a given depth. Then we check whether the depth constraint was
        reached and more importantly not exceeded.
        """
        alg = self.setup_algorithm(
            reaction_model=retrosynthesis_task1.reaction_model,
            mol_inventory=retrosynthesis_task1.inventory,
            time_limit_s=0.2 * self.time_limit_multiplier,
            limit_iterations=10_000,
            limit_reaction_model_calls=1_000,
            max_expansion_depth=depth,
        )
        output_graph, _ = alg.run_from_mol(retrosynthesis_task1.target_mol)
        node_depths = [node.depth for node in output_graph.nodes()]

        # Check that the max depth was respected
        assert max(node_depths) == depth

    @pytest.mark.parametrize("expand_purchasable_mols", [False, True])
    def test_expand_purchasable_mols(
        self, retrosynthesis_task1: RetrosynthesisTask, expand_purchasable_mols: bool
    ) -> None:
        """
        Test that the 'expand_purchasable_mols' argument is respected. If False, then molecules which are purchasable
        should not be expanded. If True, then they should be.

        To test this, we choose retrosynthesis_task1 which can be solved in 1 step. If we run the algorithm for long enough
        it will have the opportunity to expand the molecules from the original solutions. Whether these molecules are actually
        expanded should be determined by "expand_purchasable_mols".
        """

        # Run the algorithm
        alg = self.setup_algorithm(
            reaction_model=retrosynthesis_task1.reaction_model,
            mol_inventory=retrosynthesis_task1.inventory,
            time_limit_s=0.1 * self.time_limit_multiplier,
            limit_iterations=10_000,
            limit_reaction_model_calls=1_000,
            max_expansion_depth=4,
            expand_purchasable_mols=expand_purchasable_mols,
        )
        output_graph, _ = alg.run_from_mol(retrosynthesis_task1.target_mol)

        # Get nodes which correspond to the purchasable mols
        # (code depends on whether it is an AND/OR or MolSet graph)
        purchasable_mols = {Molecule("CS"), Molecule("CO")}
        if isinstance(output_graph, AndOrGraph):
            expanded_purchasable_nodes = [
                node
                for node in output_graph.nodes()
                if isinstance(node, OrNode) and node.mol in purchasable_mols and node.is_expanded
            ]
        elif isinstance(output_graph, MolSetGraph):
            expanded_purchasable_nodes = [
                node  # type: ignore[misc]  # gets confused about node type
                for node in output_graph.nodes()
                if node.mols <= purchasable_mols and node.is_expanded
            ]
        else:
            raise ValueError(f"Graph type of {type(output_graph)} not handled by this test!")

        # The actual test: are there expanded purchasable nodes?
        if expand_purchasable_mols:
            assert len(expanded_purchasable_nodes) > 0
        else:
            assert len(expanded_purchasable_nodes) == 0

    @pytest.mark.parametrize("unique_nodes", [False, True])
    def test_unique_nodes(
        self, unique_nodes: bool, retrosynthesis_task5: RetrosynthesisTask
    ) -> None:
        """
        Test that the 'unique_nodes' argument is respected, if applicable.

        If False, algorithms should run for a long time and return many nodes.

        If True, then either:
        - the algorithm should raise an error that it cannot be run with unique nodes (e.g. MCTS)
        - the algorithm should fully explore the small search space and return a pre-defined number of nodes
        """

        try:
            alg = self.setup_algorithm(
                reaction_model=retrosynthesis_task5.reaction_model,
                mol_inventory=retrosynthesis_task5.inventory,
                time_limit_s=0.1 * self.time_limit_multiplier,
                limit_iterations=10_000,
                limit_reaction_model_calls=100,
                max_expansion_depth=10,
                expand_purchasable_mols=False,
                unique_nodes=unique_nodes,
            )
        except ValueError:
            # This is the case where the algorithm does not support unique nodes,
            # but it should only occur in cases where unique_nodes=True
            assert unique_nodes
        else:
            # Should run without issue in this case
            output_graph, _ = alg.run_from_mol(retrosynthesis_task5.target_mol)

            # How many nodes to expect depends on the graph type
            if isinstance(output_graph, AndOrGraph):
                expected_num_unique_nodes = 35
            elif isinstance(output_graph, MolSetGraph):
                expected_num_unique_nodes = 12
            else:
                raise ValueError(f"Graph type of {type(output_graph)} not handled by this test!")

            # Actual test #1: if unique nodes then the number of nodes should be exactly as expected.
            # Otherwise it should be strictly greater
            actual_num_nodes = len(output_graph.nodes())
            if unique_nodes:
                assert actual_num_nodes == expected_num_unique_nodes
            else:
                assert actual_num_nodes > expected_num_unique_nodes

            # Actual test #2: was the task solved? (it should be)
            assert output_graph.root_node.has_solution

    @pytest.mark.parametrize("set_has_solution", [True, False])
    def test_set_has_solution(
        self, set_has_solution: bool, retrosynthesis_task4: RetrosynthesisTask
    ) -> None:
        """
        Test the 'set_has_solution' argument, which toggles whether the 'has_solution'
        attribute is set during node updates.
        - If True, then has_solution should be set.
        - If False, then it should not be set, even when a solution has actually been found.

        The test is run on a small finite tree for simplicity.
        """

        # Run a search which should fully exhaust the tree
        alg = self.setup_algorithm(
            reaction_model=retrosynthesis_task4.reaction_model,
            mol_inventory=retrosynthesis_task4.inventory,
            limit_iterations=10_000,
            time_limit_s=0.1 * self.time_limit_multiplier,
            set_has_solution=set_has_solution,
        )
        output_graph, _ = alg.run_from_mol(retrosynthesis_task4.target_mol)

        if set_has_solution:
            assert output_graph.root_node.has_solution
        else:
            # The root node should not have `has_solution` set
            assert not output_graph.root_node.has_solution

            # However, a solution should actually have been found,
            # so setting has solution for all nodes should reveal a solution
            run_message_passing(
                graph=output_graph, nodes=output_graph.nodes(), update_fns=[has_solution_update]
            )
            assert output_graph.root_node.has_solution

    @pytest.mark.parametrize("set_depth", [True, False])
    def test_set_depth(self, set_depth: bool, retrosynthesis_task4: RetrosynthesisTask) -> None:
        """
        Test the 'set_depth' argument, which toggles whether the 'depth'
        attribute is set during node updates.

        The test is run on a small finite tree for simplicity.
        """

        # Run a search which should fully exhaust the tree
        alg = self.setup_algorithm(
            reaction_model=retrosynthesis_task4.reaction_model,
            mol_inventory=retrosynthesis_task4.inventory,
            limit_iterations=10_000,
            time_limit_s=0.1 * self.time_limit_multiplier,
            set_depth=set_depth,
        )
        output_graph, _ = alg.run_from_mol(retrosynthesis_task4.target_mol)

        # Test on depths
        node_depths = [node.depth for node in output_graph.nodes()]
        if set_depth:
            assert (
                math.inf not in node_depths
            )  # this is the default value which should be overridden
            assert 0 in node_depths  # value for root node
            assert 1 in node_depths
        else:
            assert math.inf in node_depths  # default value should be present
            assert 1 not in node_depths

    @pytest.mark.parametrize("prevent", [False, True])
    def test_prevent_repeat_mol_in_trees(
        self, prevent: bool, retrosynthesis_task5: RetrosynthesisTask
    ) -> None:
        r"""
        Test that if the prevent_repeat_mol_in_trees argument is True then
        there are no nodes with the same ancestor mol or molset.

        Test case is to synthesize "C" with the LinearMolecules model without purchasable molecules.
        If prevent_repeat_mol_in_trees=True then the tree should be *finite*:

                           C
                         /   \
                        O     S
                        |     |
                        S     O

        Otherwise it should be able to repeat forever:

                           C
                         /   \
                        O     S
                     /   \    |  \
                    C     S   C   O
                    [...]
        """
        alg = self.setup_algorithm(
            reaction_model=retrosynthesis_task5.reaction_model,
            mol_inventory=SmilesListInventory([]),
            time_limit_s=10.0 * self.time_limit_multiplier,
            limit_iterations=1_000,
            limit_reaction_model_calls=100,
            prevent_repeat_mol_in_trees=prevent,
        )
        output_graph, _ = alg.run_from_mol(Molecule("C"))

        # Test 1: should not be solved
        assert not output_graph.root_node.has_solution

        # Test 2: max depth should low if prevent=True, otherwise high
        max_depth = max(n.depth for n in output_graph.nodes())
        if prevent:
            if isinstance(output_graph, AndOrGraph):
                assert max_depth == 4
            elif isinstance(output_graph, MolSetGraph):
                assert max_depth == 2
            else:
                raise RuntimeError()
        else:
            assert max_depth > 4  # should definitely be satisfied

    def _run_alg_and_extract_routes(
        self,
        task: RetrosynthesisTask,
        time_limit_s: float,
        limit_iterations: int = 10_000,
        max_routes: int = 100,
        **kwargs,
    ) -> list[SynthesisGraph]:
        """Utility function to run an algorithm and extract routes."""

        # Set up and run algorithm
        alg = self.setup_algorithm(
            reaction_model=task.reaction_model,
            mol_inventory=task.inventory,
            limit_iterations=limit_iterations,
            time_limit_s=time_limit_s,
            **kwargs,
        )
        output_graph, _ = alg.run_from_mol(task.target_mol)

        # Extract routes by minimum length
        for node in output_graph.nodes():
            if isinstance(node, (AndNode, MolSetNode)):
                node.data["route_cost"] = 1.0
            else:
                node.data["route_cost"] = 0.0
        routes = list(iter_routes_cost_order(output_graph, max_routes=max_routes))
        route_objs = [output_graph.to_synthesis_graph(route) for route in routes]
        return route_objs

    def test_found_routes1(self, retrosynthesis_task1: RetrosynthesisTask) -> None:
        """Test that the correct routes are found for a simple example."""

        route_objs = self._run_alg_and_extract_routes(
            retrosynthesis_task1,
            time_limit_s=0.1 * self.time_limit_multiplier,
            limit_iterations=10_000,
        )
        assert len(route_objs) > 2  # should find AT LEAST this many routes

        # Check that all expected routes are present
        for expected_route in retrosynthesis_task1.known_routes.values():
            route_matches = [expected_route == r for r in route_objs]
            assert sum(route_matches) == 1  # should be exactly one match

    def test_found_routes2(self, retrosynthesis_task2: RetrosynthesisTask) -> None:
        """Test that correct routes are found for a more complex example."""

        route_objs = self._run_alg_and_extract_routes(
            retrosynthesis_task2,
            time_limit_s=0.2 * self.time_limit_multiplier,
            limit_iterations=10_000,
        )
        assert len(route_objs) > 5  # should find AT LEAST this many routes

        # Check that all expected routes are present
        for expected_route in retrosynthesis_task2.known_routes.values():
            route_matches = [expected_route == r for r in route_objs]
            assert sum(route_matches) == 1  # should be exactly one match

        # Check that incorrect routes are not present
        for incorrect_route in retrosynthesis_task2.incorrect_routes.values():
            route_matches = [incorrect_route == r for r in route_objs]
            assert not any(route_matches)

    def test_stop_on_first_solution(self, retrosynthesis_task1: RetrosynthesisTask) -> None:
        """
        Test that `stop_on_first_solution` really does stop the algorithm once a solution is found.

        The test for this is to run the same search as in `test_found_routes1` but with
        `stop_on_first_solution=True`. This should find exactly one route for this problem.

        Note however that `stop_on_first_solution=True` does not guarantee finding at most one route
        because several routes could possibly be found at the same time. The test works for this specific
        problem because there is only one route found in the first iteration.
        """

        route_objs = self._run_alg_and_extract_routes(
            retrosynthesis_task1,
            time_limit_s=0.1 * self.time_limit_multiplier,
            limit_iterations=10_000,
            stop_on_first_solution=True,
        )
        assert len(route_objs) == 1
