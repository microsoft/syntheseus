from __future__ import annotations

import heapq
import math
from collections.abc import Collection, Iterator
from datetime import datetime
from typing import Callable, Optional, TypeVar

from syntheseus.search.graph.and_or import AndNode, OrNode
from syntheseus.search.graph.base_graph import RetrosynthesisSearchGraph
from syntheseus.search.graph.molset import MolSetNode
from syntheseus.search.graph.node import BaseGraphNode

NodeType = TypeVar("NodeType", bound=BaseGraphNode)


def _iter_top_routes(
    graph: RetrosynthesisSearchGraph,
    cost_fn: Callable[[Collection[NodeType], RetrosynthesisSearchGraph[NodeType]], float],
    cost_lower_bound: Callable[[Collection[NodeType], RetrosynthesisSearchGraph[NodeType]], float],
    max_routes: int,
    max_time_s: float = math.inf,
    yield_partial_routes: bool = False,
) -> Iterator[tuple[float, Collection[NodeType]]]:
    """
    Iterator over the minimal trees (routes) with lowest cost.
    This can be done efficiently given a lower bound on the cost.

    NOTE: it is not clear whether this function is the best way to extract routes,
    and if in general it is guaranteed to not return the same route twice. We think
    this is the case but are not sure in general.

    Args:
        graph: Graph to iterate over. Could be tree, but does not need to be.
        cost_fn: Gives the cost of a route (specified by the set of nodes).
            A cost of inf means the route will not be returned.
        cost_lower_bound: A lower bound of the cost. The lower bound means that
            if the function is evaluated on a set A, the cost of a set B >= A
            will always exceed this lower bound.
            This function will always be evaluated on partial routes.
        max_routes: Maximum number of routes to return.
        max_time_s: Maximum total number of seconds to spend extracting routes.
        yield_partial_routes: if True, will yield routes whose leaves
            have children in the full graph. This could be useful if, for example,
            there are purchasable molecules which have children.
            Typically this will be undesirable though.

    Yields:
        Tuples of cost, route nodes.
    """

    # Initialize priority queue
    # items are: cost, whether the cost is the true cost or a lower bound,
    # tie-breaking integer (since sets cannot be ordered),
    # set of nodes in partial route, list of nodes on the route's frontier
    queue: list[tuple[float, bool, int, set[NodeType], list[NodeType]]] = [
        (-math.inf, False, 0, {graph.root_node}, [graph.root_node])
    ]
    tie_breaker = 1
    start_time = datetime.now()

    # Do best-first search
    num_routes_yielded = 0
    while (
        len(queue) > 0
        and num_routes_yielded < max_routes
        and (datetime.now() - start_time).total_seconds() < max_time_s
    ):
        # Pop route
        cost, is_true_cost, _, partial_route, route_frontier = heapq.heappop(queue)
        assert cost < math.inf, "Infinite cost routes should not be in the queue."

        # Scenario 1: if it is a full route, then yield it,
        # because its cost must be lower than the partial cost of all other routes.
        if is_true_cost:
            assert len(route_frontier) == 0
            yield (cost, partial_route)
            num_routes_yielded += 1
        else:
            # Choose the first node in the frontier to be "expanded"
            # and re-add to the queue
            assert len(route_frontier) > 0
            node_to_expand = route_frontier[0]
            remaining_frontier = route_frontier[1:]
            possible_new_routes: list[tuple[set[NodeType], list[NodeType]]] = []

            # Potentially add this node without any of its children
            if len(list(graph.successors(node_to_expand))) == 0 or yield_partial_routes:
                possible_new_routes.append((partial_route, remaining_frontier))

            # Add all children routes, 1 at a time
            if isinstance(node_to_expand, OrNode):
                # For AND/OR trees, add each And Child and all of its children
                for and_child in graph.successors(node_to_expand):
                    and_child_children = list(graph.successors(and_child))
                    new_partial_route = partial_route | {and_child} | set(and_child_children)
                    # New frontier excludes nodes already in partial route which would either already be expanded
                    # or be in the frontier already
                    new_frontier = remaining_frontier + [
                        n for n in and_child_children if n not in partial_route
                    ]
                    possible_new_routes.append((new_partial_route, new_frontier))
            elif isinstance(node_to_expand, MolSetNode):
                # For MolSet graphs, add each child individually
                for child in graph.successors(node_to_expand):
                    new_partial_route = partial_route | {child}
                    new_frontier = list(remaining_frontier)
                    if child not in partial_route:
                        new_frontier.append(child)
                    possible_new_routes.append((new_partial_route, new_frontier))
            else:
                raise TypeError(f"Unknown node type {type(node_to_expand)}.")

            # Add all possible routes onto the queue
            for new_partial_route, new_frontier in possible_new_routes:
                if len(new_frontier) == 0:
                    new_cost = cost_fn(new_partial_route, graph)
                    assert new_cost >= cost, "lower bound not satisfied"
                    new_cost_is_full = True
                else:
                    new_cost = cost_lower_bound(new_partial_route, graph)
                    new_cost_is_full = False

                if new_cost < math.inf:
                    heapq.heappush(
                        queue,
                        (new_cost, new_cost_is_full, tie_breaker, new_partial_route, new_frontier),
                    )
                    tie_breaker += 1


def _route_has_solution(nodes: Collection[BaseGraphNode], graph: RetrosynthesisSearchGraph) -> bool:
    """Whether a route is solved, calculated without in-place modification of the nodes."""
    subgraph = graph._graph.subgraph(nodes)
    node_to_soln = {node: node._has_intrinsic_solution() for node in nodes}

    # Iterate to update unsolved nodes
    was_update = True
    while was_update:
        was_update = False
        for node in list(node_to_soln):
            if not node_to_soln[node]:  # only update unsolved nodes
                if isinstance(node, (OrNode, MolSetNode)):
                    node_to_soln[node] = any(node_to_soln[c] for c in subgraph.successors(node))
                elif isinstance(node, AndNode):
                    node_to_soln[node] = all(node_to_soln[c] for c in subgraph.successors(node))
                else:
                    raise TypeError
                was_update = was_update or node_to_soln[node]

    return all(node_to_soln.values())  # Has solution if all nodes are solved now


def _min_route_cost(nodes: Collection[BaseGraphNode], graph: RetrosynthesisSearchGraph) -> float:
    if _route_has_solution(nodes, graph):
        return sum(n.data.get("route_cost", 0.0) for n in nodes)
    else:
        return math.inf


def _min_route_partial_cost(
    nodes: Collection[BaseGraphNode], _: RetrosynthesisSearchGraph
) -> float:
    if all(n.has_solution for n in nodes):
        return sum(n.data.get("route_cost", 0.0) for n in nodes)
    else:
        return math.inf


def iter_routes_cost_order(
    graph: RetrosynthesisSearchGraph,
    max_routes: int,
    max_time_s: float = math.inf,
    stop_cost: Optional[float] = None,
) -> Iterator[Collection[BaseGraphNode]]:
    """
    Iterate over all solved routes from `graph` in order of increasing cost.
    `graph` can be an AND/OR or MolSet graph.
    The cost of each route is the sum of `node.data["route_cost"]` for each
    node in the route. It is assumed that this is set beforehand.
    It is also assumed that `node.has_solution` is set beforehand.

    Args:
        graph: Graph to extract routes from.
        max_routes: Maximum number of routes to yield.
        max_time_s: Maximum total number of seconds to spend extracting routes.
        stop_cost: If provided, iterator will terminate once a route of cost
            larger than `stop_cost` is encountered.
    """

    for cost, route in _iter_top_routes(
        graph=graph,
        cost_fn=_min_route_cost,
        cost_lower_bound=_min_route_partial_cost,
        max_routes=max_routes,
        max_time_s=max_time_s,
        yield_partial_routes=False,
    ):
        if stop_cost is not None and cost >= stop_cost:
            break
        else:
            yield route


def _route_time_cost(nodes, graph) -> float:
    """
    Cost function for routes that is the maximum timestamp of any node in the route
    (or inf if it is not solved).
    """
    if _route_has_solution(nodes, graph):
        return max(n.creation_time.timestamp() for n in nodes)
    else:
        return math.inf


def _route_time_partial_cost(nodes, _) -> float:
    """Partial cost version of above. It is a lower bound to the true cost."""
    if all(n.has_solution for n in nodes):
        return max(n.creation_time.timestamp() for n in nodes)
    else:
        return math.inf


def iter_routes_time_order(
    graph: RetrosynthesisSearchGraph, max_routes: int, max_time_s: float = math.inf
) -> Iterator[Collection[BaseGraphNode]]:
    """
    Iterate over all solved routes from `graph` in the order they were created
    (a route is considered created when the last node in the route is created).

    `graph` can be an AND/OR or MolSet graph.
    Creation time is measured by `node.creation_time`.

    Args:
        graph: Graph to extract routes from.
        max_routes: Maximum number of routes to yield.
        max_time_s: Maximum total number of seconds to spend extracting routes.
    """

    for _, r in _iter_top_routes(
        graph=graph,
        cost_fn=_route_time_cost,
        cost_lower_bound=_route_time_partial_cost,
        max_routes=max_routes,
        max_time_s=max_time_s,
        yield_partial_routes=False,
    ):
        yield r
