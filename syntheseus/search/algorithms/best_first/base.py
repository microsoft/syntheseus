from __future__ import annotations

import abc
import heapq
import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Generic, Sequence

from syntheseus.search.algorithms.base import GraphType, SearchAlgorithm
from syntheseus.search.graph.node import BaseGraphNode

logger = logging.getLogger(__name__)


@dataclass(order=True)
class PriorityQueueItem:
    priority: Any  # usually float, but just needs to be comparable
    tie_breaker: int
    item: Any = field(compare=False)


class PriorityQueue:
    """
    Simple priority queue implementation which supports removing elements
    and changing their priority.

    Implementation based on heapq and guidance from:
    https://docs.python.org/3/library/heapq.html#priority-queue-implementation-notes

    In particular, this means items are never really removed from the queue:
    they are just marked as invalid and set to `None`. This is why the queue
    length is not necessarily the same as the number of unique items in the
    queue. It also assumes that items are unique and hashable.
    """

    def __init__(self):
        self._queue = []
        self._entry_finder = {}
        self._counter = itertools.count()  # to break ties in priority

    def remove_item(self, item):
        """Removes an item if it is present."""
        if item in self._entry_finder:
            entry = self._entry_finder.pop(item)
            entry.item = None

    def push_item(self, item, priority):
        """
        Pushes an item with a given priority.

        If the item is already present, it is removed and re-inserted.
        """
        if item in self._entry_finder:
            self.remove_item(item)
        entry = PriorityQueueItem(priority, next(self._counter), item)
        self._entry_finder[item] = entry
        heapq.heappush(self._queue, entry)

    def pop_item(self):
        """Removes an item with the lowest priority and returns it."""
        while self._queue:
            entry = heapq.heappop(self._queue)
            if self._entry_finder.get(entry.item) == entry:
                # ^^ ensures 1) item in entry finder and
                # 2) item in entry finder has the same priority as the popped item
                # (if `None` is added to the queue there may be multiple `None` items)
                del self._entry_finder[entry.item]
                return entry
        raise IndexError("pop from an empty priority queue")

    def __len__(self) -> int:
        """Length is number of unique items in the queue."""
        return len(self._entry_finder)

    def __contains__(self, item) -> bool:
        return item in self._entry_finder

    def get_priority(self, item) -> Any:
        """Returns the priority of an item in the queue, raising KeyError if not found."""
        return self._entry_finder[item].priority

    def raw_len(self) -> int:
        return len(self._queue)


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

        # Initialize queue. Queue should never contain nodes with priority of inf.
        queue = PriorityQueue()
        for node in graph._graph.nodes():
            if self.node_eligible_for_queue(node, graph):
                priority = self.priority_function(node, graph)
                if priority < math.inf:
                    queue.push_item(node, priority)
        if logger_active:
            logger.log(log_level, f"Initial queue has {len(queue)} nodes")

        # Run search until time limit or queue is empty
        step = 0
        for step in range(self.limit_iterations):
            if self.should_stop_search(graph):
                break

            # Take nodes from the priority queue until a node eligible for expansion is found
            num_popped = 0
            while len(queue) > 0:
                pq_item = queue.pop_item()
                num_popped += 1

                # Do a few checks
                assert pq_item.priority < math.inf, "Inf priority should not be in the queue"
                assert math.isclose(
                    pq_item.priority, self.priority_function(pq_item.item, graph)
                ), "Priority in the queue should always be up-to-date"

                if self.node_eligible_for_queue(pq_item.item, graph):
                    node = pq_item.item
                    break
            else:
                logger.log(log_level, "No eligible node found. Stopping search.")
                break

            # Visit node
            new_nodes = list(self.visit_node(node, graph))

            # Update node values
            nodes_updated = self.set_node_values(new_nodes + [node], graph)

            # Add new nodes and updated nodes to the queue, since their priority may have changed.
            # dict.fromkeys is to preserve order and uniqueness
            for updated_node in dict.fromkeys(new_nodes + list(nodes_updated)):
                if self.node_eligible_for_queue(updated_node, graph):
                    updated_node_priority = self.priority_function(updated_node, graph)
                    already_in_queue_at_correct_priority = updated_node in queue and math.isclose(
                        queue.get_priority(updated_node), updated_node_priority
                    )

                    if already_in_queue_at_correct_priority:
                        # In this case re-inserting the node is redundant so we do nothing
                        pass
                    elif updated_node_priority < math.inf:
                        # Main case: insert (or re-insert) node into queue
                        queue.push_item(updated_node, updated_node_priority)
                    else:
                        # Edge case: reached if new priority is inf.
                        # In this case, remove the node from the queue if it was there.
                        queue.remove_item(updated_node)

            # Log
            if logger_active:
                logging_str = (
                    f"Step: {step}, "
                    f"Nodes affected during visit: {len(new_nodes)}, "
                    f"Nodes updated: {len(nodes_updated)}, "
                    f"Graph size: {len(graph)}, "
                    f"Queue size: {len(queue)} (raw length: {queue.raw_len()}), "
                    f"Num popped: {num_popped}, "
                    f"Reaction model calls: {self.reaction_model.num_calls()}, "
                    f"Node visited: {node}"
                )
                logger.log(
                    log_level,
                    logging_str,
                )

        return step
