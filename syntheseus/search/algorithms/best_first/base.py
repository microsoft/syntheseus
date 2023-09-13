from __future__ import annotations

import abc
import heapq
import logging
import math
from collections.abc import Sequence
from typing import Generic

from syntheseus.search.algorithms.base import GraphType, SearchAlgorithm
from syntheseus.search.graph.node import BaseGraphNode

logger = logging.getLogger(__name__)


class GeneralBestFirstSearch(SearchAlgorithm[GraphType, int], Generic[GraphType]):
    """
    Base class for 'best-first search' algorithms which hold a queue with the 'best'
    nodes, and at each step pop the best node and (probably) expand it.
    """

    @property
    def requires_tree(self) -> bool:
        return True  # NOTE: this restriction could be loosened later

    @abc.abstractmethod
    def priority_function(self, node: BaseGraphNode, graph: GraphType) -> float:
        """
        Defines a node's priority in the queue.
        Lower value means higher priority.
        Priority of inf generally won't be added to the queue.
        """
        raise NotImplementedError

    def node_eligible_for_queue(self, node: BaseGraphNode, graph: GraphType) -> bool:
        """
        Whether a node is eligible to be added to the queue.
        Default implementation is to check whether the node is expanded.
        """
        return self.can_expand_node(node, graph)

    def visit_node(self, node: BaseGraphNode, graph: GraphType) -> Sequence[BaseGraphNode]:
        """
        Visit a node (whatever that means for the algorithm).
        Default implementation is to expand the node,
        which is what most algorithms do.

        Returns a collection of new nodes created by the visit.
        """
        return self.expand_node(node, graph)

    def _run_from_graph_after_setup(self, graph: GraphType) -> int:
        # Logging setup
        log_level = logging.DEBUG - 1
        logger_active = logger.isEnabledFor(log_level)

        # Initialize queue
        queue: list[tuple[float, int, BaseGraphNode]] = []
        tie_breaker = 0  # to break ties in priority
        for node in graph._graph.nodes():
            if self.node_eligible_for_queue(node, graph):
                priority = self.priority_function(node, graph)
                if priority < math.inf:
                    heapq.heappush(queue, (priority, tie_breaker, node))
                    tie_breaker += 1
        if logger_active:
            logger.log(log_level, f"Initial queue has {len(queue)} nodes")

        # Run search until time limit or queue is empty
        step = 0
        for step in range(self.limit_iterations):
            if self.should_stop_search(graph) or len(queue) == 0:
                break

            # Pop node and potentially expand it
            priority, _, node = heapq.heappop(queue)
            assert priority < math.inf, "inf priority should not be in the queue"

            # Re-calculate priority in case it changed since it was added to the queue
            latest_priority = self.priority_function(node, graph)

            # Decide between 3 options: discarding the node, re-inserting it,
            # or visiting it (which most likely means expanding it)
            if not self.node_eligible_for_queue(node, graph):
                action = "discarded (already expanded/not eligible for queue)"
            elif not math.isclose(priority, latest_priority):
                # Re-insert the node with the correct priority,
                # unless the new priority is inf
                priority_change_str = f"(priority changed from {priority} to {latest_priority})"
                if latest_priority < math.inf:
                    action = f"re-inserted {priority_change_str}"
                    heapq.heappush(queue, (latest_priority, tie_breaker, node))
                    tie_breaker += 1
                else:
                    action = f"discarded {priority_change_str}"
            else:
                # Visit node
                new_nodes = list(self.visit_node(node, graph))

                # Update node values
                nodes_updated = self.set_node_values(new_nodes + [node], graph)

                # Add new eligible nodes to the queue, since
                # their priority may have changed.
                # dict.fromkeys is to preserve order and uniqueness
                for updated_node in dict.fromkeys(new_nodes + list(nodes_updated)):
                    if self.node_eligible_for_queue(updated_node, graph):
                        updated_node_priority = self.priority_function(updated_node, graph)
                        if updated_node_priority < math.inf:
                            heapq.heappush(
                                queue, (updated_node_priority, tie_breaker, updated_node)
                            )
                            tie_breaker += 1

                # Log str
                action = f"visited, {len(new_nodes)} new nodes created, {len(nodes_updated)} nodes updated)"

            # Log
            if logger_active:
                logger.log(
                    log_level,
                    f"Step {step}:\tnode={node}, action={action}, queue size={len(queue)}",
                )

        return step
