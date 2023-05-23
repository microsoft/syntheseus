"""
Message passing is used in other algorithms and is tested implicitly in the algorithm tests.

Therefore this file just contains minimal tests of correctness and edge cases.
"""

from __future__ import annotations

import pytest

from syntheseus.search.graph.and_or import AndOrGraph
from syntheseus.search.graph.message_passing import (
    depth_update,
    has_solution_update,
    run_message_passing,
)


def test_no_update_functions(andor_tree_non_minimal: AndOrGraph) -> None:
    """
    Test that if no update functions are provided, the message passing algorithm
    doesn't actually run.
    """
    g = andor_tree_non_minimal  # rename for brevity
    output = run_message_passing(g, g.nodes(), update_fns=[], max_iterations=1)
    assert len(output) == 0


def test_no_input_nodes(andor_tree_non_minimal: AndOrGraph) -> None:
    """
    If no input nodes are provided, the message passing algorithm should
    terminate without running.
    """
    g = andor_tree_non_minimal  # rename for brevity
    output = run_message_passing(g, [], update_fns=[has_solution_update], max_iterations=1)
    assert len(output) == 0


@pytest.mark.parametrize("update_successors", [True, False])
def test_update_successors(andor_tree_non_minimal: AndOrGraph, update_successors: bool) -> None:
    """
    Test that the "update successors" function works as expected by
    setting the "has_solution" attribute of the root node to False.

    If update_successors=False then message passing should terminate after 1 iteration.
    However, if update_successors=True then message passing should terminate visiting
    the root node and its children (3 iterations). In both cases only 1 node should be updated.
    """
    g = andor_tree_non_minimal  # rename for brevity
    if update_successors:
        enough_iterations = 3
    else:
        enough_iterations = 1
    too_few_iterations = enough_iterations - 1

    # Test 1: in both cases, root node should be updated to has_solution=True
    # and that should be the only node updated
    g.root_node.has_solution = False
    output = run_message_passing(
        g,
        [g.root_node],
        update_fns=[has_solution_update],
        update_successors=update_successors,
        max_iterations=enough_iterations,
    )  # should run without error
    assert g.root_node.has_solution
    assert len(output) == 1

    # Test 2: should raise error if too few iterations
    g.root_node.has_solution = False
    with pytest.raises(RuntimeError):
        run_message_passing(
            g,
            [g.root_node],
            update_fns=[has_solution_update],
            update_successors=update_successors,
            max_iterations=too_few_iterations,
        )


@pytest.mark.parametrize("update_predecessors", [True, False])
def test_update_predecessors(andor_tree_non_minimal: AndOrGraph, update_predecessors: bool) -> None:
    """Similar to above but for predecessors, updating the depth of a node."""

    g = andor_tree_non_minimal  # rename for brevity
    node_to_perturb = list(g.successors(g.root_node))[0]
    if update_predecessors:  # NOTE: only enough if not updating successors
        too_few_iterations = 1
        enough_iterations = 2
    else:
        too_few_iterations = 0
        enough_iterations = 1

    # Test 1: set depth of child node incorrectly
    # and check that it is updated correctly.
    node_to_perturb.depth = 500
    output = run_message_passing(
        g,
        [node_to_perturb],
        update_fns=[depth_update],
        update_predecessors=update_predecessors,
        max_iterations=enough_iterations,
        update_successors=False,
    )  # should run without error
    assert node_to_perturb.depth == 1
    assert len(output) == 1

    # Test 2: should raise error if too few iterations
    node_to_perturb.depth = 500
    with pytest.raises(RuntimeError):
        run_message_passing(
            g,
            [node_to_perturb],
            update_fns=[depth_update],
            update_predecessors=update_predecessors,
            max_iterations=too_few_iterations,
            update_successors=False,
        )


def update_fn_which_can_never_converge(*args, **kwargs) -> bool:
    return True


def test_no_convergence(andor_tree_non_minimal: AndOrGraph) -> None:
    """Test that if no convergence is reached, an error is raised."""
    g = andor_tree_non_minimal  # rename for brevity
    with pytest.raises(RuntimeError):
        run_message_passing(
            g, g.nodes(), update_fns=[update_fn_which_can_never_converge], max_iterations=1_000
        )
