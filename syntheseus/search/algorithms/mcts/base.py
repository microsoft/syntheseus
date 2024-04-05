from __future__ import annotations

import abc
import logging
import math
import random
import warnings
from dataclasses import dataclass
from typing import Callable, Generic, Optional, Sequence, TypeVar, cast

import numpy as np

from syntheseus.search import INT_INF
from syntheseus.search.algorithms.base import GraphType, SearchAlgorithm
from syntheseus.search.algorithms.mixins import ValueFunctionMixin
from syntheseus.search.graph.base_graph import RetrosynthesisSearchGraph
from syntheseus.search.graph.node import BaseGraphNode
from syntheseus.search.node_evaluation import BaseNodeEvaluator

RewardNodeType = TypeVar("RewardNodeType", bound=BaseGraphNode)
PolicyNodeType = TypeVar("PolicyNodeType", bound=BaseGraphNode)


logger = logging.getLogger(__name__)


def random_argmin(arr: list[float], random_state: Optional[random.Random] = None) -> int:
    """
    Returns a random minimum index of a list of numbers.

    If a list has only one minimum element, it returns its index:
    e.g. for [2, 1, 3] this will return index 1

    If a list has multiple minimum elements, it will sample one uniformly
    at random.
    e.g. for [2, 1, 3, 1, 1] it will return either 1, 3, or 4 with probability
    1/3 each.

    Args:
        arr: The list of floats whose random argmin to find.

    Raises:
        ValueError: if list is empty
        ValueError: if any elements are NaN (behaviour not well defined here)

    Returns:
        an integer i such that arr[i] is a randomly-chosen minimum element of arr.
    """

    # Cast to numpy array
    if len(arr) == 0:
        raise ValueError("List must not be empty!")
    arr_np = np.asarray(arr)
    if np.any(np.isnan(arr_np)):
        raise ValueError("No NaNs allowed!")

    # Initialize random state
    random_state = random_state or random.Random()

    # Argsort
    argsort = list(np.argsort(arr_np))

    # Choose indices
    min_indices = []
    for idx in argsort:
        if math.isclose(arr[argsort[0]], arr[idx]):
            min_indices.append(idx)
        else:
            break

    return random_state.choice(min_indices)


@dataclass
class MCTS_IterResult:
    nodes_visited: list
    nodes_created: list
    nodes_updated: set


def uct_bound(node: BaseGraphNode, graph: RetrosynthesisSearchGraph) -> float:
    parents = list(graph.predecessors(node))
    if node.num_visit == 0:
        return math.inf
    elif len(parents) == 0:
        return 1.0  # arbitrary value
    elif len(parents) == 1:
        return math.sqrt(math.log(parents[0].num_visit) / node.num_visit)
    else:
        raise ValueError(f"Nodes should have 1 parent in MCTS, this node has {len(parents)}")


def pucb_bound(node: BaseGraphNode, graph: RetrosynthesisSearchGraph) -> float:
    parents = list(graph.predecessors(node))
    if len(parents) == 0:
        return 1.0  # arbitrary value

    if "policy_score" not in node.data:
        raise RuntimeError("P-UCB requires a policy score to be set on each node.")

    if len(parents) != 1:
        raise ValueError(f"Nodes should have 1 parent in MCTS, this node has {len(parents)}")

    policy_score = node.data["policy_score"]
    assert policy_score >= 0, "Policies must be non-negative."
    return policy_score * math.sqrt(parents[0].num_visit) / (1 + node.num_visit)


class BaseMCTS(
    ValueFunctionMixin[RewardNodeType],
    SearchAlgorithm[GraphType, int],
    Generic[GraphType, RewardNodeType, PolicyNodeType],
    abc.ABC,
):
    def __init__(
        self,
        reward_function: BaseNodeEvaluator[RewardNodeType],
        min_num_visit_to_expand: int = 1,
        bound_function: Callable[[BaseGraphNode, GraphType], float] = uct_bound,
        bound_constant: float = 1.0,
        policy: Optional[BaseNodeEvaluator[PolicyNodeType]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reward_function = reward_function
        self.policy = policy
        self.min_num_visit_to_expand = min_num_visit_to_expand
        self.bound_function = bound_function
        self.bound_constant = bound_constant
        self.init_mcts_value = 0.0

        # Runtime warning
        self._check_infinite_runtime()

    def _check_infinite_runtime(self) -> None:
        """Perform a check to see if the algorithm could run forever and warn the user."""
        no_limit_iter = self.limit_iterations >= INT_INF
        no_limit_rxn = self.limit_reaction_model_calls >= INT_INF
        no_limit_time = self.time_limit_s >= math.inf
        no_limit_nodes = self.limit_graph_nodes >= INT_INF

        if no_limit_iter and no_limit_rxn and no_limit_time and no_limit_nodes:
            warnings.warn(
                "No kind of run limit set. This algorithm will almost certainty run forever."
            )
        elif no_limit_iter and no_limit_time:
            warnings.warn(
                "No iteration or time limit set (although a reaction model call and/or graph node limit was set)."
                " Under these conditions, it is possible (but not certain) that MCTS "
                "will run forever (for example if there are no leaf nodes eligible for expansion in the graph)."
                " At the very least, it could run for an unexpectedly long time."
                " It is recommended to set either an iteration limit or a time limit."
            )

    def setup(self, graph) -> None:
        # If there is only one node in the initial graph (the root node),
        # then the first visit will call the value function unnecessarily.
        # Save one call by doing a "pseudo-visit" and setting the value arbitrarily.
        if len(graph) == 1 and graph.root_node.num_visit == 0:
            graph.root_node.data.setdefault("mcts_value", self.init_mcts_value)
            graph.root_node.num_visit += 1

        return super().setup(graph)

    def _run_from_graph_after_setup(self, graph: GraphType) -> int:
        # Logging setup
        log_level = logging.DEBUG - 1
        logger_active = logger.isEnabledFor(log_level)

        # Run search until time limit or queue is empty
        step = 0  # define explicitly to handle 0 iteration edge case
        for step in range(self.limit_iterations):
            if self.should_stop_search(graph):
                break

            # Visit root node
            result = self.mcts_visit(graph.root_node, graph)

            # Set all node values just to be safe
            # (MCTS subclasses may have some values that need to be set after the visit is complete)
            all_nodes_to_update = {graph.root_node}
            all_nodes_to_update.update(
                result.nodes_visited, result.nodes_created, result.nodes_updated
            )
            self.set_node_values(all_nodes_to_update, graph)

            # Potential logging
            if logger_active:
                logger.log(
                    log_level,
                    f"Step {step}: reward={graph.root_node.data.get('mcts_prev_reward')},"
                    f" num_visit={graph.root_node.num_visit},",
                )

        return step

    def mcts_visit(self, node: BaseGraphNode, graph: GraphType) -> MCTS_IterResult:
        """Recursive function to visit a node."""

        # Logging setup
        log_level = logging.DEBUG - 2
        logger_active = logger.isEnabledFor(log_level)
        if logger_active:
            logger.log(log_level, f"Visiting node {node} (num_visit={node.num_visit})")

        # Step 0: initialize return value
        output = MCTS_IterResult(
            nodes_visited=[node],
            nodes_created=[],
            nodes_updated=set(),
        )

        # Step 1: expand the node if it can be expanded and has been visited enough times
        if self.can_expand_node(node, graph) and node.num_visit >= self.min_num_visit_to_expand:
            logger.log(log_level, "Expanding node")

            # Do expansion
            new_nodes = list(self.expand_node(node, graph))

            # Assign policy values
            children = list(graph.successors(node))
            if self.policy is not None and len(children) > 0:
                policy_values = self.policy(children, graph)
                for child, policy_value in zip(children, policy_values):
                    child.data["policy_score"] = policy_value
            del children

            # Immediately set node values in case rewards depend on them
            nodes_updated = self.set_node_values(new_nodes, graph)

            # Track results
            output.nodes_created.extend(new_nodes)
            output.nodes_updated.update(nodes_updated)
        else:
            logger.log(log_level, "Not expanding node")

        # Step 2: obtain reward from node
        if len(list(graph.successors(node))) == 0:
            # case A: no children. Reward comes from the node itself
            reward = self._get_leaf_node_reward(cast(RewardNodeType, node), graph)
            logger.log(log_level, f"Reward from leaf node: {reward}")
        else:
            # case b: has children. Reward comes from child nodes

            # Choose child(ren) to visit and visit them
            children_to_visit = self.choose_successors_to_visit(node, graph)
            child_visit_outputs = [self.mcts_visit(child, graph) for child in children_to_visit]

            # Produce a reward from these children
            reward = self._get_reward_from_successors(node, graph, children_to_visit)
            logger.log(log_level, f"Reward from children: {reward}")

            # Track results
            for child_output in child_visit_outputs:
                output.nodes_visited.extend(child_output.nodes_visited)
                output.nodes_created.extend(child_output.nodes_created)
                output.nodes_updated.update(child_output.nodes_updated)

        # Step 3: update node values
        self._update_value_from_reward(node, graph, reward)

        # Step 4: return visit output
        return output

    def choose_successors_to_visit(
        self, node: BaseGraphNode, graph: GraphType
    ) -> Sequence[BaseGraphNode]:
        children = list(graph.successors(node))
        assert len(children) > 0, "Cannot choose successors to visit if there are no successors."

        # Find bound function values for each child
        bound_values = [self.bound_function(child, graph) for child in children]

        # Find lower confidence bound values: -(value + c*bound)
        lower_bound_values = [
            -child.data["mcts_value"] - self.bound_constant * bound
            for child, bound in zip(children, bound_values)
        ]

        # Choose child with lowest lower bound value
        idx = random_argmin(lower_bound_values, random_state=self.random_state)
        return [children[idx]]

    def _get_reward_from_successors(
        self, node: BaseGraphNode, graph: GraphType, children_visited: Sequence[BaseGraphNode]
    ) -> float:
        """After a series of children have been visited, produce a reward for this node from the children."""

        # Standard case is that 1 child is visited, and the reward is the reward received by that child
        assert len(children_visited) == 1
        return children_visited[0].data["mcts_prev_reward"]

    def _get_leaf_node_reward(self, node: RewardNodeType, graph: GraphType) -> float:
        """
        Get a reward for visiting a node.
        This is either the value of the reward function (for terminal nodes),
        or the value of the value function (for non-terminal nodes).
        """
        assert len(list(graph.successors(node))) == 0, "This is not a leaf node."

        # Use the REWARD function if:
        #   - the node is expanded but has no children
        #   - the node is not expanded, but will not be expanded by the algorithm
        #     (e.g. because mols are purchasable or it is too deep)
        if node.is_expanded or not self.can_expand_node(node, graph):
            return self.reward_function([node], graph)[0]
        else:
            return self.value_function([node], graph)[0]

    def _update_value_from_reward(self, node: BaseGraphNode, _: GraphType, reward: float) -> None:
        if node.num_visit == 0:
            node.data["mcts_value"] = reward  # overwrite any placeholder value
        else:
            total_reward = reward + node.data["mcts_value"] * node.num_visit
            node.data["mcts_value"] = total_reward / (node.num_visit + 1)

        node.data["mcts_prev_reward"] = reward
        node.num_visit += 1

    def set_node_values(self, nodes, graph):
        updated_nodes = super().set_node_values(nodes, graph)
        for node in updated_nodes:
            node.data.setdefault("mcts_value", self.init_mcts_value)
        return updated_nodes
