"""
Contains code related to the "Retrosynthetic Planning with Dual Value Networks" algorithm.

See: https://arxiv.org/abs/2301.13755
"""

from __future__ import annotations

import abc
import logging
import math
import random
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar, cast

import numpy as np

from syntheseus.search import INT_INF
from syntheseus.search.algorithms.base import AndOrSearchAlgorithm
from syntheseus.search.graph.and_or import ANDOR_NODE, AndNode, AndOrGraph, OrNode
from syntheseus.search.algorithms.mcts.base import BaseMCTS, pucb_bound, random_argmin
from syntheseus.search.algorithms.mixins import ValueFunctionMixin
from syntheseus.search.graph.base_graph import RetrosynthesisSearchGraph
from syntheseus.search.graph.node import BaseGraphNode
from syntheseus.search.node_evaluation import BaseNodeEvaluator


logger = logging.getLogger(__name__)  # TODO: is this used??

class PDVN_MCTS(BaseMCTS[AndOrGraph, OrNode, AndNode], AndOrSearchAlgorithm[int]):
    """
    Code for the MCTS algorithm used to train PDVN.
    
    TODO: explain how it is just a modified version of MCTS.

    TODO: maybe change the reward type of base MCTS?

    NOTE: value function is set to value function cost

    TODO variables:
    - pdvn_mcts_v_syn: accumulated "synthesizability" value (equation 4/7 of Liu et al 2023)
    - pdvn_mcts_v_cost: accumulated "cost" value (equation 4/7 of Liu et al 2023)
    - rewards: are tuple valued, with values (syn [binary], cost [float])
    """
    def __init__(self, 
        c_dead: float,  # cost of a dead end (equation 1 of Liu et al 2023)
        value_function_syn: BaseNodeEvaluator[OrNode],
        value_function_cost: BaseNodeEvaluator[OrNode],
        and_node_cost_fn: BaseNodeEvaluator[AndNode],
        bound_function: Callable[[AndNode], AndOrGraph] = pucb_bound,  # equation 5 of Liu et al 2023
        **kwargs
    ):
        super().__init__(bound_function=bound_function, value_function=value_function_cost, reward_function=None, **kwargs)
        self.c_dead = c_dead
        self.value_function_syn = value_function_syn
        self.value_function_cost = value_function_cost
        self.and_node_cost_fn = and_node_cost_fn
        assert self.policy is not None, "PDVN_MCTS requires a policy to be specified."

    def set_node_values(self, nodes, graph):
        updated_nodes = super().set_node_values(nodes, graph)
        
        # Ensure synthesis and cost values are initialized
        for node in updated_nodes:
            node.data.setdefault("pdvn_mcts_v_syn", self.init_mcts_value)
            node.data.setdefault("pdvn_mcts_v_cost", self.init_mcts_value)
        
        # Ensure reaction costs are set
        nodes_without_cost = [n for n in updated_nodes if n.data.get("pdvn_mcts_reaction_cost") is None and isinstance(n, AndNode)]
        self._set_and_node_costs(nodes_without_cost, graph)

        return updated_nodes

    def _set_and_node_costs(self, and_nodes: Sequence[AndNode], graph: AndOrGraph) -> None:
        costs = self.and_node_cost_fn(and_nodes, graph=graph)
        assert len(costs) == len(and_nodes)
        for node, cost in zip(and_nodes, costs):
            node.data["pdvn_mcts_reaction_cost"] = cost

    def choose_successors_to_visit(
        self, node: ANDOR_NODE, graph: AndOrGraph,
    ) -> Sequence[ANDOR_NODE]:

        if isinstance(node, OrNode):
            # This is equivalent to normal MCTS. Choose single child based on upper confidence bound
            output = super().choose_successors_to_visit(node, graph)
        elif isinstance(node, AndNode):

            # From text under equation 5, choose a child which is unsolved and has been visited
            # the fewest times.
            # if all children are solved, choose a random one
            children = list(graph.successors(node))
            unsolved_children = [c for c in children if not c.has_solution]
            if len(unsolved_children) == 0:
                # All children are solved, choose a random one
                output = [self.random_state.choice(children)]
            else:
                # Some children are unsolved, choose the one with the least visits,
                # breaking ties randomly
                output_idx = random_argmin([c.num_visit for c in unsolved_children], random_state=self.random_state)
                output = [unsolved_children[output_idx]]
        else:
            raise TypeError(f"Unexpected node type: {type(node)}")
        
        # Ensure only 1 node was chosen
        assert len(output) == 1, "PDVN_MCTS can only choose 1 successor to visit."
        return output

    def _get_leaf_node_reward(self, node: OrNode, graph: AndOrGraph) -> tuple[float, float]:
        """
        Reward for visiting a leaf node. Leaf nodes are always OrNodes.
        The reward is either:
        - (0, c_dead) if the node is a dead end
        - (1, 0) if the node is purchasable
        - (v_syn, v_cost) if the node is expandable (i.e. reward comes from the dual value functions)

        NOTE: in the future we may want to assign a non-zero cost to purchasable nodes.
        """
        assert len(list(graph.successors(node))) == 0, "This is not a leaf node."

        if node.is_expanded or not self.can_expand_node(node, graph):
            purchasable = node.mol.metadata["is_purchasable"]
            reward_syn = float(purchasable)
            reward_cost = 0.0 if purchasable else self.c_dead
        else:
            reward_syn = self.value_function_syn([node], graph)[0]
            reward_cost = self.value_function_cost([node], graph)[0]
        
        return reward_syn, reward_cost

    def _get_reward_from_successors(
        self, node: ANDOR_NODE, graph: AndOrGraph, children_visited: ANDOR_NODE,
    ) -> tuple[float, float]:
        """
        Get rewards from children. For an OrNode these are just the rewards from the AndNode visited.

        For an AndNode this reward comes from the last reward of all children visited.
        """
        assert len(children_visited) == 1
        if isinstance(node, OrNode):
            reward_syn = children_visited[0].data["pdvn_mcts_prev_reward_syn"]
            reward_cost = children_visited[0].data["pdvn_mcts_prev_reward_cost"]
        elif isinstance(node, AndNode):
            
            # Synthesis reward is product of reward from current child
            # and the running average from all other children.
            # Cost reward is reaction cost plus cost reward from visited child
            # plus running average of costs from all other children.
            reward_syn = children_visited[0].data["pdvn_mcts_prev_reward_syn"]
            reward_cost = (
                children_visited[0].data["pdvn_mcts_prev_reward_cost"] +
                node.data["pdvn_mcts_reaction_cost"]
            )
            for c in graph.successors(node):
                if c not in children_visited:
                    reward_syn *= c.data["pdvn_mcts_v_syn"]
                    reward_cost += c.data["pdvn_mcts_v_cost"]
            
        else:
            raise TypeError(f"Unexpected node type: {type(node)}")
            
        return reward_syn, reward_cost

    def _update_value_from_reward(self, node: ANDOR_NODE, _: AndOrGraph, reward: tuple[float, float]) -> None:
        
        # Set "last rewards"
        reward_syn, reward_cost = reward
        node.data["pdvn_mcts_prev_reward_syn"] = reward_syn
        node.data["pdvn_mcts_prev_reward_cost"] = reward_cost

        # Update running cost averages
        if node.num_visit == 0:
            node.data["pdvn_mcts_v_syn"] = reward_syn
            node.data["pdvn_mcts_v_cost"] = reward_cost
        else:
            total_reward_syn = reward_syn + node.num_visit * node.data["pdvn_mcts_v_syn"]
            total_reward_cost = reward_cost + node.num_visit * node.data["pdvn_mcts_v_cost"]
            node.data["pdvn_mcts_v_syn"] = total_reward_syn / (node.num_visit + 1)
            node.data["pdvn_mcts_v_cost"] = total_reward_cost / (node.num_visit + 1)
        
        # Update MCTS value (- p_syn * cost_syn - (1-p_syn) * cost_dead)
        node.data["mcts_value"] = - (
            node.data["pdvn_mcts_v_syn"] * node.data["pdvn_mcts_v_cost"] + 
            (1 - node.data["pdvn_mcts_v_syn"]) * self.c_dead
        )
        node.num_visit += 1
            
    

def pdvn_min_cost_update(*args):
    # minimum cost update for PDVN
    # TODO
    pass

def pdvn_extract_training_data():
    # given a graph, extracts all training data required for PDVN
    # TODO
    pass