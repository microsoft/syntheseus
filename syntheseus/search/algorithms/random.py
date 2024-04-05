from __future__ import annotations

import logging
from typing import Generic

from syntheseus.search.algorithms.base import (
    AndOrSearchAlgorithm,
    GraphType,
    MolSetSearchAlgorithm,
    SearchAlgorithm,
)

logger = logging.getLogger(__name__)


class BaseRandomSearch(SearchAlgorithm[GraphType, int], Generic[GraphType]):
    """Base class for both AND/OR and MolSet random search algorithms."""

    @property
    def requires_tree(self) -> bool:
        return False  # can work on any graph

    def _run_from_graph_after_setup(self, graph: GraphType) -> int:
        log_level = logging.DEBUG - 1
        logger_active = logger.isEnabledFor(log_level)

        # Initialize a set of nodes which can be expanded
        expandable_nodes = {node for node in graph.nodes() if self.can_expand_node(node, graph)}

        step = 0  # initialize this variable in case loop is not entered
        for step in range(self.limit_iterations):
            if self.should_stop_search(graph) or len(expandable_nodes) == 0:
                break

            # Choose a random node to expand
            node = self.random_state.choice(list(expandable_nodes))
            expandable_nodes.remove(node)

            # Expand the node
            new_nodes = self.expand_node(node, graph)
            self.set_node_values(set(new_nodes) | {node}, graph)
            for n in new_nodes:
                if self.can_expand_node(n, graph):
                    expandable_nodes.add(n)

            if logger_active:
                logger.log(
                    log_level,
                    f"Step {step}: node {node} expanded, created {len(new_nodes)} new nodes. "
                    f"Num expandable nodes: {len(expandable_nodes)}.",
                )

        return step


class MolSet_RandomSearch(BaseRandomSearch, MolSetSearchAlgorithm):
    @property
    def requires_tree(self) -> bool:
        # Even though it "could" work, molset graphs with unique nodes are not
        # well-supported so we don't allow it at this time.
        return True


class AndOr_RandomSearch(BaseRandomSearch, AndOrSearchAlgorithm):
    pass
