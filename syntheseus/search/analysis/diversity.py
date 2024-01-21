from __future__ import annotations

import logging
import random
from collections.abc import Collection
from typing import Callable, Optional

import networkx as nx

from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.search.graph.route import SynthesisGraph

ROUTE_DISTANCE_METRIC = Callable[[SynthesisGraph, SynthesisGraph], float]

logger = logging.getLogger(__name__)


def estimate_packing_number(
    routes: Collection[SynthesisGraph],
    radius: float,
    distance_metric: ROUTE_DISTANCE_METRIC,
    max_packing_number: Optional[int] = None,
    num_tries: int = 10,
    random_state: Optional[random.Random] = None,
) -> list[SynthesisGraph]:
    """
    Estimate packing number of a set of routes,
    defined as the size of the largest subset
    where d(x, y) > radius for all distinct x, y in the subset.
    This function estimates the packing number by trying to construct
    a subset of routes that satisfy the definition above.
    Because this is an NP-hard problem, we use a greedy heuristic algorithm.
    This algorithm is run several times and the best result is returned.

    Because this algorithm is constructive, we return the set of routes
    rather than simply the packing number. The size of the returned set
    is a lower bound to the true packing number.

    Args:
        routes: set of routes to estimate the packing number of.
        radius: distance threshold for the definition of packing number.
        distance_metric: distance between two routes.
        max_packing_number: to avoid expensive computations,
            the algorithm is stopped if a packing number
            larger or equal to this value is found.
            If `None`, the algorithm will run until completion.
        num_tries: the number of random restarts to perform.
        random_state: random state to use for shuffling routes.

    Returns:
        A set of routes with the largest packing number found.
    """

    # Cleanly handle edge case of no routes
    if len(routes) == 0:
        return list()

    # Check argument type (leads to cryptic error message if not checked)
    assert all(
        isinstance(route, SynthesisGraph) for route in routes
    ), "Routes must be of type SynthesisGraph."

    # Initialize random state
    random_state = random_state or random.Random()

    # Try to get best packing set
    best_packing_set: list[SynthesisGraph] = list()
    route_list = list(routes)
    for try_idx in range(num_tries):
        if max_packing_number is not None and len(best_packing_set) >= max_packing_number:
            logger.debug("Stopping early because max packing number has been reached.")
            break  # no point trying further since the max packing number has been reached

        # Shuffle routes (gives a random restart to greedy algorithm)
        random_state.shuffle(route_list)

        # Construct a packing set and check whether it is better than the previous one
        packing_set = _recursive_construct_packing_set(
            0,
            len(route_list),
            route_list,
            radius,
            distance_metric,
            max_packing_number,
        )
        logger.debug(
            f"Run #{try_idx+1}/{num_tries}:"
            f" Found packing set of size {len(packing_set)}."
            f" (previous best size={len(best_packing_set)})."
        )
        if len(packing_set) > len(best_packing_set):
            logger.debug("This is the new best.")
            best_packing_set = packing_set

    return best_packing_set


def _recursive_construct_packing_set(
    idx_start: int,
    idx_end: int,
    routes: list[SynthesisGraph],
    radius: float,
    distance_metric: ROUTE_DISTANCE_METRIC,
    max_packing_number: Optional[int] = None,
) -> list[SynthesisGraph]:
    """
    Recursive helper function for estimate_packing_number which finds a packing set.

    The base case is when <= 1 route is provided: here the packing set is just the set of routes.
    If >= 2 routes are provided, then the list is divided into two subsets,
    and a recursive call is made to find a packing set for each subset.
    These packing sets are then optimally merged, requiring at most (N/2)^2 distance computations.

    The details of the optimal merging are as follows:

    1. A graph is constructed between all routes in both packing sets A and B,
       with an edge between two nodes if they correspond to routes with distance at most `radius`.
       Because A and B are individually packing sets, this graph is *bipartite*,
       i.e. edges will only exist between nodes in A and nodes in B.
    2. A *minimum vertex cover* (a set of vertices where each edge in the graph
        links to at least one vertex in the set) is found by first finding a *maximum
        matching* (a set of edges in a graph such that no two edges share a vertex),
        then applying Konig's theorem which guarantees a correspondence between
        a maximum matching and a minimum vertex cover for bipartite graphs.
        This is all implemented in `networkx`, which provides efficient algorithms.
    3. The nodes from the minimum vertex cover are removed from the graph.
        From the definition of vertex cover, this means that no edges will remain in the
        graph, and because a *minimum* vertex cover was found,
        this means that the smallest possible number of nodes was deleted.
        The remaining nodes now form a packing set.
    """

    assert (
        max_packing_number is None or max_packing_number > 0
    ), "Max packing number must be positive."

    # Base cases: simple greedy algorithm is optimal if there are no more than two routes
    if idx_end - idx_start <= 2:
        best_set: list[SynthesisGraph] = []
        for idx in range(idx_start, idx_end):
            if all(distance_metric(routes[idx], route) > radius for route in best_set):
                best_set.append(routes[idx])
                if len(best_set) == max_packing_number:
                    break

        return best_set

    # Recursive case:
    # First calculate packing set for both halves
    cutoff_idx = (idx_start + idx_end) // 2

    route_set1 = _recursive_construct_packing_set(
        idx_start,
        cutoff_idx,
        routes,
        radius,
        distance_metric,
        max_packing_number,
    )
    if len(route_set1) == max_packing_number:  # return directly if max packing number is reached
        return route_set1

    route_set2 = _recursive_construct_packing_set(
        cutoff_idx,
        idx_end,
        routes,
        radius,
        distance_metric,
        max_packing_number,
    )
    if len(route_set2) == max_packing_number:  # return directly if max packing number is reached
        return route_set2

    assert (
        max_packing_number is None or max(len(route_set1), len(route_set2)) <= max_packing_number
    ), "Max packing number exceeded in recursive call."

    compatibility_graph = nx.Graph()

    top_nodes = [(0, idx) for idx in range(len(route_set1))]
    bottom_nodes = [(1, idx) for idx in range(len(route_set2))]
    compatibility_graph.add_nodes_from(top_nodes + bottom_nodes)

    for idx1, route1 in enumerate(route_set1):
        for idx2, route2 in enumerate(route_set2):
            if distance_metric(route1, route2) <= radius:
                compatibility_graph.add_edge((0, idx1), (1, idx2))

    # Compute the minimum vertex cover in `compatibility_graph`.
    matching = nx.bipartite.maximum_matching(compatibility_graph, top_nodes=top_nodes)
    vertex_cover = nx.bipartite.to_vertex_cover(compatibility_graph, matching, top_nodes=top_nodes)

    # The maximum independent set is the complement of the vertex cover.
    independent_set = list(set(compatibility_graph) - vertex_cover)

    if max_packing_number is not None:
        # Truncate the solution if it is too big (avoids asserts failing downstream).
        independent_set = independent_set[:max_packing_number]

    return [[route_set1, route_set2][side][idx] for side, idx in independent_set]


def _jaccard_distance(
    set1: set,
    set2: set,
) -> float:
    intersection_size = len(set1 & set2)
    union_size = len(set1) + len(set2) - intersection_size

    if union_size == 0:
        return 0.0  # both sets are empty so distance is 0
    else:
        return 1.0 - intersection_size / union_size


def _get_reactions(route: SynthesisGraph) -> set[SingleProductReaction]:
    return set(route._graph.nodes)


def _get_molecules(route: SynthesisGraph) -> set[Molecule]:
    all_mols: set[Molecule] = set()
    for rxn in route._graph.nodes:
        all_mols.add(rxn.product)
        all_mols.update(rxn.reactants)
    return all_mols


def reaction_jaccard_distance(
    route1: SynthesisGraph,
    route2: SynthesisGraph,
) -> float:
    """
    Calculate the Jaccard distance between the sets of reactions in 2 routes.
    """

    # Get sets of reactions
    reactions1 = _get_reactions(route1)
    reactions2 = _get_reactions(route2)
    return _jaccard_distance(reactions1, reactions2)


def molecule_jaccard_distance(
    route1: SynthesisGraph,
    route2: SynthesisGraph,
) -> float:
    """
    Calculate the Jaccard distance between the sets of molecules in 2 routes.
    """

    # Get sets of molecules
    molecules1 = _get_molecules(route1)
    molecules2 = _get_molecules(route2)
    return _jaccard_distance(molecules1, molecules2)


def reaction_symmetric_difference_distance(
    route1: SynthesisGraph,
    route2: SynthesisGraph,
) -> float:
    """
    Calculate the symmetric difference distance between the sets of reactions in 2 routes.
    """

    # Get sets of reactions
    reactions1 = _get_reactions(route1)
    reactions2 = _get_reactions(route2)
    return len(reactions1 ^ reactions2)


def molecule_symmetric_difference_distance(
    route1: SynthesisGraph,
    route2: SynthesisGraph,
) -> float:
    """
    Calculate the symmetric difference distance between the sets of molecules in 2 routes.
    """

    # Get sets of molecules
    molecules1 = _get_molecules(route1)
    molecules2 = _get_molecules(route2)
    return len(molecules1 ^ molecules2)
