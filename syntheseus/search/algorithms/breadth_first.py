from __future__ import annotations

import collections
import logging
from typing import Generic

from syntheseus.search.algorithms.base import (
    AndOrSearchAlgorithm,
    GraphType,
    MolSetSearchAlgorithm,
    SearchAlgorithm,
)

logger = logging.getLogger(__name__)


class GeneralBreadthFirstSearch(SearchAlgorithm[GraphType, int], Generic[GraphType]):
    """Base class for breadth first search algorithms (pseudo-code is the same for all data structures)."""

    @property
    def requires_tree(self) -> bool:
        return False  # can work on any graph

    def _run_from_graph_after_setup(self, graph: GraphType) -> int:
        log_level = logging.DEBUG - 1
        logger_active = logger.isEnabledFor(log_level)

        queue = collections.deque([node for node in graph._graph.nodes() if not node.is_expanded])
        step = 0  # initialize this variable in case loop is not entered
        for step in range(self.limit_iterations):
            if self.should_stop_search(graph) or len(queue) == 0:
                break

            # Pop node and potentially expand it
            node = queue.popleft()
            if node.is_expanded:
                outcome = "already expanded, do nothing"
            else:
                new_nodes = self.expand_node(node, graph)
                self.set_node_values(set(new_nodes) | {node}, graph)
                queue.extend([n for n in new_nodes if self.can_expand_node(n, graph)])
                outcome = f"expanded, created {len(new_nodes)} new nodes"

            if logger_active:
                logger.log(
                    log_level, f"Step {step}: node {node} {outcome}. Queue size: {len(queue)}"
                )

        return step


class MolSet_BreadthFirstSearch(GeneralBreadthFirstSearch, MolSetSearchAlgorithm):
    @property
    def requires_tree(self) -> bool:
        # Even though it "could" work, molset graphs with unique nodes are not
        # well-supported so we don't allow it at this time.
        return True


class AndOr_BreadthFirstSearch(GeneralBreadthFirstSearch, AndOrSearchAlgorithm):
    pass
