from __future__ import annotations

import logging
from collections import deque
from collections.abc import Collection, Iterable
from typing import Callable, Sequence, TypeVar

from syntheseus.search import INT_INF
from syntheseus.search.graph.base_graph import RetrosynthesisSearchGraph
from syntheseus.search.graph.node import BaseGraphNode

NodeType = TypeVar("NodeType", bound=BaseGraphNode)

logger = logging.getLogger(__name__)


def run_message_passing(
    graph: RetrosynthesisSearchGraph,
    nodes: Iterable[NodeType],
    update_fns: Sequence[Callable[[NodeType, RetrosynthesisSearchGraph], bool]],
    update_predecessors: bool = True,
    update_successors: bool = True,
    max_iterations: int = INT_INF,
) -> Collection[BaseGraphNode]:
    """
    Runs a general message passing algorithm until convergence,
    returning a set of nodes which were updated. This function can be used
    as a useful sub-routine for many algorithms which assign values to nodes
    defined recursively with respect to the values of other nodes. Examples include:
    - the "depth" of a node (minimum distance to root node)
    - whether a node is "solved"
    - Number of distinct solutions for a node

    More precisely, the algorithm keeps a FIFO queue of nodes, initialized by `nodes`.
    At each iteration, a node is popped from the queue and:
    1) all functions in `update_fns` are called on the node
    2) if any of these functions return True, all the node's predecessors and successors are added to the queue,
        since they may need to be updated. This can be optionally modified using the
        `update_predecessors` and `update_successors` arguments (more on this below).

    The actual message passing is controlled by a list of `update_fns`.
    These functions are expected to perform a message passing update
    on a node based on the values of its predecessors and successors. It is assumed that:
    1. These functions *only* change the node they are called on (not its predecessors/successors nodes)
    2. Depend only on the node itself and the attributes of predecessor/successor nodes
    3. Return "True" if a node attribute was changed due to an update, and "False" if it remained the same.

    If these assumptions are met,
    this procedure will only terminate if message passing has *converged*
    (meaning that running `update_fn` won't change any nodes).
    In other cases, it may never terminate / may terminate without converging.
    Conversely, meeting these assumptions does not imply that the procedure will converge,
    since for some update functions it may not be possible to reach a "steady state".
    An example of this is an update function which increments an internal node counter:
    the value of the counter will always change and therefore this procedure will never converge.

    For update functions which depend only on a node's predecessors or on its successors but not both,
    it may be more efficient to set either `update_predecessors` or `update_successors`
    to False to avoid performing unnecessary updates. For example, consider a node's
    depth, which is defined as the minimum predecessor depth + 1 (for a general graph).
    If a node's depth is updated, we know that the depth of the predecessors will not change
    because of this, so it would be redundant to add the predecessor nodes to the update queue.
    In this case, setting `update_predecessors=False` would result in higher efficiency.

    Returns a set of nodes where at least one update function caused a change.

    Args:
        nodes: nodes to start update propagation with (used to seed the queue)
        update_fns: sequence of update functions. Each function is expected to set some
            attribute of a node based on its predecessors/successors,
            and return True if the attribute has changed, or False if it did not change.
        update_predecessors: if True, a node's predecessors will be added to the update queue
            if one of its values changes. Set to False if all functions are completely
            determined by a node's predecessors for improved efficiency.
        update_successors: if True, a node's successors will be added to the update queue
            if one of its values changes. Set to False if all functions are completely
            determined by a node's successors for improved efficiency.
        max_iterations: maximum number of iterations to run before terminating.

    Returns:
        Set of all nodes where one of the `update_fns` returned true at least one time.
    """
    message_passing_log_level = logging.DEBUG - 2  # message passing is very low level
    logger_active = logger.isEnabledFor(message_passing_log_level)

    # Special case: no update functions = no nodes are updated
    if len(update_fns) == 0:
        logger.log(
            message_passing_log_level,
            "No update functions, so message passing will terminate.",
        )
        return set()

    # Initialize a FIFO queue to track nodes that may need updating
    queue = deque(nodes)

    # Iterate through the queue, checking all nodes and adding their predecessors/successors
    changed_set: set[NodeType] = set()
    n_iter = 0
    while len(queue) > 0:
        n_iter += 1
        if n_iter > max_iterations:
            raise RuntimeError(
                f"Message passing did not converge after {max_iterations} iterations."
            )
        node = queue.popleft()

        # Update its value
        something_changed = [update_fn(node, graph) for update_fn in update_fns]

        # If it changed, then potentially add adjacent nodes to queue
        if any(something_changed):
            changed_set.add(node)
            if update_predecessors:
                for parent in graph.predecessors(node):
                    queue.append(parent)

            if update_successors:
                for child in graph.successors(node):
                    queue.append(child)

        # Potentially log the update
        if logger_active:
            logger.log(
                message_passing_log_level,
                f"Iteration {n_iter}. Node {node} updates: {something_changed}. Queue size: {len(queue)}.",
            )

    logger.log(
        message_passing_log_level,
        f"End of message passing. Number of nodes updated: {len(changed_set)}.",
    )
    return changed_set
