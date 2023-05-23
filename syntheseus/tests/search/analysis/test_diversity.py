"""Test diversity analysis."""
from __future__ import annotations

import random

import pytest

from syntheseus.search.analysis.diversity import (
    estimate_packing_number,
    molecule_jaccard_distance,
    molecule_symmetric_difference_distance,
    reaction_jaccard_distance,
    reaction_symmetric_difference_distance,
)
from syntheseus.search.graph.route import SynthesisGraph


def test_empty_input() -> None:
    """Test there is no crash when an empty list of routes is input."""
    distinct_routes = estimate_packing_number(
        routes=[], radius=0.5, distance_metric=reaction_jaccard_distance, num_tries=100
    )
    assert len(distinct_routes) == 0


@pytest.mark.parametrize(
    "metric, threshold, expected_packing_number",
    [
        # Easy case 1: distance threshold is very large so only 1 route can be returned.
        # For Jaccard distance choose a value of 1 (the max value dist distance can take)
        # For symmetric difference distance choose a value of 20 (more than the max number of mols/rxns in a route)
        (molecule_jaccard_distance, 1.0, 1),
        (reaction_jaccard_distance, 1.0, 1),
        (molecule_symmetric_difference_distance, 20, 1),
        (reaction_symmetric_difference_distance, 20, 1),
        #
        # Easy case 2: distance threshold = 0 so it should just count number of distinct routes (11)
        (molecule_jaccard_distance, 0, 11),
        (reaction_jaccard_distance, 0, 11),
        (molecule_symmetric_difference_distance, 0, 11),
        (reaction_symmetric_difference_distance, 0, 11),
        #
        # Individual harder cases
        (molecule_jaccard_distance, 0.8, 2),
        (molecule_jaccard_distance, 0.5, 7),
        (molecule_jaccard_distance, 0.2, 9),
        (
            reaction_jaccard_distance,
            0.99,
            7,
        ),  # routes with completely non-overlapping reaction sets
        (reaction_symmetric_difference_distance, 4, 6),
        (molecule_symmetric_difference_distance, 2, 7),  # differ in at least 2 molecules
    ],
)
def test_estimate_packing_number(
    sample_synthesis_routes: list[SynthesisGraph],
    metric,
    threshold: float,
    expected_packing_number: int,
) -> None:
    """
    Check that after a large number of trials, the correct packing number is found
    for a set of routes.
    """

    # Run the packing number estimation
    distinct_routes = estimate_packing_number(
        routes=sample_synthesis_routes,
        radius=threshold,
        distance_metric=metric,
        num_tries=1000,
        random_state=random.Random(100),
    )

    # Check that routes returned are all a distance > threshold from each other
    for i, route1 in enumerate(distinct_routes):
        for route2 in distinct_routes[i + 1 :]:
            assert metric(route1, route2) > threshold

    # Check that the correct packing number is found
    assert len(distinct_routes) == expected_packing_number


@pytest.mark.parametrize(
    "metric, threshold, max_packing_number, expected_packing_number",
    [
        # Easy case 1: distance threshold is very large so only 1 route can be returned
        (reaction_jaccard_distance, 1.0, 0, 0),  # limit is 0 so no routes can be returned
        (reaction_jaccard_distance, 1.0, 10, 1),  # limit is higher than actual number
        (reaction_jaccard_distance, 1e-3, 5, 5),  # are 11 such routes but limit is 5
    ],
)
def test_max_packing_number(
    sample_synthesis_routes: list[SynthesisGraph],
    metric,
    threshold: float,
    max_packing_number: int,
    expected_packing_number: int,
) -> None:
    """Test that max packing number actually limits the packing number."""

    # Run the packing number estimation
    distinct_routes = estimate_packing_number(
        routes=sample_synthesis_routes,
        radius=threshold,
        distance_metric=metric,
        max_packing_number=max_packing_number,
        num_tries=100,
        random_state=random.Random(100),
    )
    assert len(distinct_routes) == expected_packing_number
