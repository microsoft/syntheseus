"""
Test route extraction. This is tested indirectly in the algorithm tests,
so tests here are not comprehensive and mostly focus on edge cases.

NOTE: the tests here cover "min_cost_routes" method, but not more general
"_iter_top_routes" method. This could be improved in the future.
"""
from __future__ import annotations

import numpy as np
import pytest

from syntheseus.search.analysis.route_extraction import (
    iter_routes_cost_order,
    iter_routes_time_order,
)
from syntheseus.search.graph.and_or import AndNode, AndOrGraph
from syntheseus.search.graph.molset import MolSetGraph
from syntheseus.search.graph.route import SynthesisGraph
from syntheseus.tests.search.analysis.conftest import set_uniform_costs


def test_correct_min_cost_routes_andor(
    andor_graph_non_minimal: AndOrGraph,
    andor_tree_non_minimal: AndOrGraph,
    minimal_synthesis_graph: SynthesisGraph,
) -> None:
    """Test that the correct routes are extracted from an AND/OR graph."""

    # Do same tests for both versions of the graph
    # Cost is different because some reactions are duplicated
    for g, expected_costs in [
        (andor_graph_non_minimal, [3.0, 3.0]),
        (andor_tree_non_minimal, [3.0, 4.0]),
    ]:
        set_uniform_costs(g)
        all_routes = list(iter_routes_cost_order(g, max_routes=10_000))

        # Test 1: should be 2 routes
        assert len(all_routes) == 2

        # Test 2: reaction costs should be correct
        route_costs = [sum(node.data["route_cost"] for node in route) for route in all_routes]
        assert route_costs == expected_costs

        # Test 3: exactly one route should be the same as ground-truth synthesis graph
        routes_match_reference = [
            g.to_synthesis_graph(route) == minimal_synthesis_graph  # type: ignore  # node type unclear
            for route in all_routes
        ]
        assert sum(routes_match_reference) == 1


def test_correct_min_cost_routes_molset(
    molset_tree_non_minimal: MolSetGraph, minimal_synthesis_graph: SynthesisGraph
) -> None:
    g = molset_tree_non_minimal  # short name for variable
    set_uniform_costs(g)
    all_routes = list(iter_routes_cost_order(g, max_routes=10_000))

    # Test 1: should be 2 "proper" routes, but due to permutations there are 3 total
    assert len(all_routes) == 3

    # Test 2: reaction costs should be correct
    route_costs = [sum(node.data["route_cost"] for node in route) for route in all_routes]
    assert route_costs == [4.0, 4.0, 5.0]  # higher than above because root node has cost of 1

    # Test 3: 2 routes should match reference synthesis graph (due to different order of reactions)
    routes_match_reference = [
        g.to_synthesis_graph(route) == minimal_synthesis_graph  # type: ignore  # node type unclear
        for route in all_routes
    ]
    assert sum(routes_match_reference) == 2


def test_correct_routes_andor_time_order(
    andor_graph_non_minimal: AndOrGraph,
    andor_tree_non_minimal: AndOrGraph,
    minimal_synthesis_graph: SynthesisGraph,
) -> None:
    """
    Test that the correct routes are extracted from an AND/OR graph
    in the order that they should be found.

    To do this, we check that the number of routes found is correct
    and that the reference route is found at the expected time,
    which is the first route for both the graph and the tree.
    """

    # Do same tests for both versions of the graph
    # Cost is different because some reactions are duplicated
    for g in [
        (andor_graph_non_minimal),
        (andor_tree_non_minimal),
    ]:
        all_routes = list(iter_routes_time_order(g, max_routes=10_000))

        # Test 1: should be 2 routes
        assert len(all_routes) == 2

        # Test 2: only the route at the specified index should match reference synthesis graph
        routes_match_reference = [
            g.to_synthesis_graph(route) == minimal_synthesis_graph  # type: ignore  # node type unclear
            for route in all_routes
        ]
        assert routes_match_reference == [True, False]


def test_correct_routes_molset_time_order(
    molset_tree_non_minimal: MolSetGraph, minimal_synthesis_graph: SynthesisGraph
) -> None:
    g = molset_tree_non_minimal  # short name for variable
    all_routes = list(iter_routes_time_order(g, max_routes=10_000))

    # Test 1: should be 2 "proper" routes, but due to permutations there are 3 total
    assert len(all_routes) == 3

    # Test 2: the first and third routes should match the reference route
    routes_match_reference = [
        g.to_synthesis_graph(route) == minimal_synthesis_graph  # type: ignore  # node type unclear
        for route in all_routes
    ]
    assert routes_match_reference == [True, False, True]


def _check_routes_unique(graph, routes) -> None:
    synthesis_graphs = [graph.to_synthesis_graph(route) for route in routes]
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            assert synthesis_graphs[i] != synthesis_graphs[j]


@pytest.mark.parametrize("max_routes", [-1, 0, 1, 2, 3, 100])
def test_max_routes(andor_graph_with_many_routes: AndOrGraph, max_routes: int) -> None:
    """
    Test that max_routes is respected (and that routes are unique)
    for all top-level route extraction functions.
    """
    g = andor_graph_with_many_routes
    for route_extraction_function in [iter_routes_cost_order, iter_routes_time_order]:
        # NOTE: type ignore is because mypy confused about type of route extraction function
        all_routes = list(route_extraction_function(g, max_routes=max_routes))  # type: ignore[operator]

        # Test 1: check that max number of routes is reached
        if max_routes <= 0:
            assert len(all_routes) == 0
        else:
            assert len(all_routes) == max_routes

        # Test 2: check that routes are unique
        _check_routes_unique(g, all_routes)


@pytest.mark.parametrize("stop_cost", [0.0, 0.5, 0.999, 1.0, 2.0, 2.5, 3.0, 4.0])
def test_stop_cost(andor_graph_with_many_routes: AndOrGraph, stop_cost: float) -> None:
    """Test that stop_cost argument is respected (should yield routes until stop_cost is reached)."""

    # Extract all routes
    g = andor_graph_with_many_routes
    all_routes = list(iter_routes_cost_order(g, max_routes=1_000_000, stop_cost=stop_cost))
    route_costs = [sum(node.data["route_cost"] for node in route) for route in all_routes]

    # Test #1: no route should have cost >= stop cost
    assert all(cost < stop_cost for cost in route_costs)

    # Test #2: expected number of routes should be extracted
    if stop_cost <= 1.0:
        assert len(all_routes) == 0  # lowest route cost is 1.0
    elif stop_cost <= 2.0:
        assert len(all_routes) == 1  # lowest route cost is 1.0; nothing in between
    elif stop_cost <= 3.0:
        assert len(all_routes) == 3
    elif stop_cost <= 4.0:
        assert len(all_routes) == 11
    else:
        raise NotImplementedError(f"Test not implemented for stop_cost={stop_cost}")

    # Test #3: all routes should be unique
    _check_routes_unique(g, all_routes)


@pytest.mark.parametrize(
    "max_routes,stop_cost,expected_num_routes",
    [
        (2, 10.0, 2),  # > 2 routes of cost < 10 but max routes is 2 so only 2 routes returned
        (1_000, 4.0, 11),  # only 11 routes of cost < 4.0 so 11 routes returned (max routes ignored)
        (1, 1.01, 1),  # 1 route of cost < 1.01 so 1 route returned (stop cost matches max routes)
    ],
)
def test_stop_cost_max_routes_together(
    andor_graph_with_many_routes: AndOrGraph,
    max_routes: int,
    stop_cost: float,
    expected_num_routes: int,
) -> None:
    """
    Test that stop_cost and max_routes can be used together
    (route extraction terminates when EITHER condition is reached).
    """
    g = andor_graph_with_many_routes
    all_routes = list(iter_routes_cost_order(g, max_routes=max_routes, stop_cost=stop_cost))
    assert len(all_routes) == expected_num_routes


def test_has_solution_false(andor_graph_with_many_routes: AndOrGraph) -> None:
    """
    Test that routes are not extracted if has_solution=False,
    even if the underlying graph *does* contain solutions.

    Should be true for all top-level extraction functions.
    """
    g = andor_graph_with_many_routes
    for node in g.nodes():
        node.has_solution = False
    for route_extraction_function in [iter_routes_cost_order, iter_routes_time_order]:
        # NOTE: type ignore is because mypy confused about type of route extraction function
        all_routes = list(route_extraction_function(g, max_routes=1_000_000))  # type: ignore[operator]
        assert len(all_routes) == 0


def test_alternative_cost_function(andor_graph_with_many_routes: AndOrGraph) -> None:
    """
    Test that alternative cost functions can be used.
    """

    # Set alternative costs so that 2 routes of length 3 are optimal
    g = andor_graph_with_many_routes
    for node in g.nodes():
        if isinstance(node, AndNode):
            # These 3 for the first route
            if node.reaction.reaction_smiles == "CCCOS>>CCCOC":
                node.data["route_cost"] = 0.3
            elif node.reaction.reaction_smiles == "CC.COS>>CCCOS":
                node.data["route_cost"] = 0.3
            elif node.reaction.reaction_smiles == "COC>>COS":
                node.data["route_cost"] = 0.3

            # These 3 for the second route
            if node.reaction.reaction_smiles == "CCCOO>>CCCOC":
                node.data["route_cost"] = 0.31
            elif node.reaction.reaction_smiles == "CC.COO>>CCCOO":
                node.data["route_cost"] = 0.31
            elif node.reaction.reaction_smiles == "COC>>COO":
                node.data["route_cost"] = 0.31

    all_routes = list(iter_routes_cost_order(g, max_routes=3))
    route_costs = [sum(node.data["route_cost"] for node in route) for route in all_routes]
    assert np.allclose(route_costs, [0.9, 0.93, 1.0], atol=1e-4)
