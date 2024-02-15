"""
Contains code related to the "Retrosynthetic Planning with Dual Value Networks" algorithm.

See: https://arxiv.org/abs/2301.13755

In particular, this file contains code for the AND/OR-MCTS algorithm used to make training data for PDVN,
and the code to extract training data from a completed search graph.
"""

from __future__ import annotations

import enum
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Sequence

from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.search.algorithms.base import AndOrSearchAlgorithm
from syntheseus.search.algorithms.mcts.base import BaseMCTS, pucb_bound
from syntheseus.search.graph.and_or import ANDOR_NODE, AndNode, AndOrGraph, OrNode
from syntheseus.search.graph.message_passing import (
    depth_update,
    has_solution_update,
    run_message_passing,
)
from syntheseus.search.node_evaluation import BaseNodeEvaluator


class SynthesizabilityOutcome(enum.IntEnum):
    DEAD_END = -1
    NO_SOLN_FOUND = 0
    SOLN_FOUND = 1


@dataclass
class PDVNSearchData:
    """Container for all data extracted from a search graph used to train policies/value functions in PDVN."""

    mol_to_synthesizability: dict[Molecule, SynthesizabilityOutcome]
    mol_to_min_syn_cost: dict[Molecule, float]
    mol_to_reactions_for_min_syn: dict[Molecule, set[SingleProductReaction]]


class PDVN_MCTS(BaseMCTS[AndOrGraph, OrNode, AndNode], AndOrSearchAlgorithm[int]):
    """
    Code for the MCTS algorithm used to train PDVN. It is essentially a modified version of MCTS
    for AND/OR trees. At an OrNode, a child reaction is selection using P-UCB. At and AndNode,
    one unsolved child is chosen. Two separate rewards are received: one for synthesis success,
    and another for synthesis costs. Details can be found in the PDVN paper (Liu et al 2023).

    The key logic of this algorithm is inherited from `mcts/base.py`, where the `mcts_visit`
    function is called recursively to descend the graph and update values based on the reward
    received. Because PDVN-MCTS uses the same select, expand, and backup steps as normal MCTS,
    only the different functions are overridden. Each function is explained in its docstring.

    Key `node.data` variables used by this algorithm are:
    - pdvn_mcts_v_syn: accumulated "synthesizability" value (equation 4/7 of Liu et al 2023)
    - pdvn_mcts_v_cost: accumulated "cost" value (equation 4/7 of Liu et al 2023)
    - rewards: are tuple valued, with values (syn [binary], cost [float])
    """

    def __init__(
        self,
        c_dead: float,  # cost of a dead end (equation 1 of Liu et al 2023)
        value_function_syn: BaseNodeEvaluator[OrNode],
        value_function_cost: BaseNodeEvaluator[OrNode],
        and_node_cost_fn: BaseNodeEvaluator[AndNode],
        bound_function: Callable[[AndNode], AndOrGraph] = pucb_bound,  # type: ignore[assignment]  # sloppy typing for bounds
        **kwargs,
    ):
        super().__init__(
            bound_function=bound_function,  # type: ignore[arg-type]  # sloppy typing for bounds
            value_function=value_function_cost,
            reward_function=None,  # type: ignore[arg-type]  # reward function is not used for PDVN MCTS
            **kwargs,
        )
        self.c_dead = c_dead
        self.value_function_syn = value_function_syn
        self.value_function_cost = value_function_cost
        self.and_node_cost_fn = and_node_cost_fn
        assert self.policy is not None, "PDVN_MCTS requires a policy to be specified."

    def set_node_values(self, nodes, graph):
        """
        This function updates values for all nodes in the graph and ensures everything is
        properly initialized.

        In addition to the logic in the base class (which sets things like depth and has_solution),
        this function ensures that the synthesizability and cost values are initialized
        and that the cost of every reaction is set.
        """
        updated_nodes = super().set_node_values(nodes, graph)

        # Ensure synthesis and cost values are initialized
        for node in updated_nodes:
            node.data.setdefault("pdvn_mcts_v_syn", self.init_mcts_value)
            node.data.setdefault("pdvn_mcts_v_cost", self.init_mcts_value)

        # Ensure reaction costs are set
        nodes_without_cost = [
            n
            for n in updated_nodes
            if n.data.get("pdvn_reaction_cost") is None and isinstance(n, AndNode)
        ]
        self._set_and_node_costs(nodes_without_cost, graph)

        return updated_nodes

    def _set_and_node_costs(self, and_nodes: Sequence[AndNode], graph: AndOrGraph) -> None:
        """Helper function to compute the cost of each AndNode using the `and_node_cost_fn`."""
        costs = self.and_node_cost_fn(and_nodes, graph=graph)
        assert len(costs) == len(and_nodes)
        for node, cost in zip(and_nodes, costs):
            node.data["pdvn_reaction_cost"] = cost

    def choose_successors_to_visit(
        self,
        node: ANDOR_NODE,  # type: ignore[override]  # AND/OR vs any graph node
        graph: AndOrGraph,
    ) -> Sequence[ANDOR_NODE]:
        """
        This function performs the "select" step of MCTS. It chooses which child node(s) to visit.

        In normal MCTS a single child is always chosen using an upper confidence bound. In PDVN-MCTS,
        the same logic is used to select a child at an OrNode (selecting a single reaction). However,
        at an AndNode the logic is different: a single OrNode child is chosen using the logic in
        section 3.2 of Liu et al 2023, which states:

            To select a child molecule node, we prioritize molecules that have not been expanded;
            if none are available, we choose ones that have not been solved. If molecule nodes are
            either all expanded or all solved, we randomly select one of them.

        The logic below follows the paragraph above, disregarding that the final clause
        will only be reached if the children are all expanded *and* all solved, rather than *or*
        as it says in the text.
        """
        if isinstance(node, OrNode):
            # This is equivalent to normal MCTS. Choose single child based on upper confidence bound
            output = super().choose_successors_to_visit(node, graph)
        elif isinstance(node, AndNode):
            children = list(graph.successors(node))
            unsolved_children = [c for c in children if not c.has_solution]
            expandable_children = [c for c in children if self.can_expand_node(c, graph)]

            # Choose a child at random from either unexpanded, unsolved, or all children
            if len(expandable_children) > 0:
                # First priority: expand an unexpanded child, if it exists
                children_to_choose_from = expandable_children
            elif len(unsolved_children) > 0:
                # Second priority: expand an unsolved node, if it exists
                children_to_choose_from = unsolved_children
            else:
                # Third priority: choose a random node
                children_to_choose_from = children
            output = [self.random_state.choice(children_to_choose_from)]
        else:
            raise TypeError(f"Unexpected node type: {type(node)}")

        # Ensure only 1 node was chosen
        assert len(output) == 1, "PDVN_MCTS can only choose 1 successor to visit."
        return output  # type: ignore[return-value]  # mypy thinks it could be any graph node due to super() call

    def _get_leaf_node_reward(self, node: OrNode, graph: AndOrGraph) -> tuple[float, float]:  # type: ignore[override]
        """
        Reward for visiting a leaf node. Leaf nodes are always OrNodes.
        Unlike regular MCTS where the reward is a single value,
        in PDVN MCTS the reward is a pair of floats representing the synthesizability
        and cost rewards. In this implementation they take one of the following values:
        - (0, c_dead) if the node is a dead end
        - (1, 0) if the node is purchasable (NOTE: could generalize to non-zero cost for purchasable mols in future.)
        - (v_syn, v_cost) if the node is expandable (i.e. reward comes from the dual value functions)
        """
        assert len(list(graph.successors(node))) == 0, "This is not a leaf node."

        if not self.can_expand_node(node, graph):
            purchasable = node.mol.metadata["is_purchasable"]
            reward_syn = float(purchasable)
            reward_cost = 0.0 if purchasable else self.c_dead
        else:
            reward_syn = self.value_function_syn([node], graph)[0]
            reward_cost = self.value_function_cost([node], graph)[0]

        return reward_syn, reward_cost

    def _get_reward_from_successors(  # type: ignore[override]  # returning a tuple
        self,
        node: ANDOR_NODE,  # type: ignore[override]  # AND/OR vs any graph node
        graph: AndOrGraph,
        children_visited: Sequence[ANDOR_NODE],  # type: ignore[override]  # AND/OR vs any graph node
    ) -> tuple[float, float]:
        """
        Get the reward for this node from its children which were visited.
        This function is used in the "backup" step of MCTS.

        It sets the values for pdvn_mcts_prev_reward_syn / pdvn_mcts_prev_reward_cost:
        these are the rewards from the current trajectory, and are denoted by
        V_T^syn and V_T^cost in the paper. The updates here are an alternative
        version of equation 6 from Liu et al 2023. Instead of being defined
        directly from the rewards of the children of the reaction visited,
        we define V_T^syn/cost for and AndNode as the product/sum of the rewards
        for its children, then define V_T^syn/cost for an OrNode as the reward
        from the AndNode that was visited.
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
                children_visited[0].data["pdvn_mcts_prev_reward_cost"]
                + node.data["pdvn_reaction_cost"]
            )
            for c in graph.successors(node):
                if c not in children_visited:
                    reward_syn *= c.data["pdvn_mcts_v_syn"]
                    reward_cost += c.data["pdvn_mcts_v_cost"]

        else:
            raise TypeError(f"Unexpected node type: {type(node)}")

        return reward_syn, reward_cost

    def _update_value_from_reward(
        self, node: ANDOR_NODE, _: AndOrGraph, reward: tuple[float, float]  # type: ignore[override]
    ) -> None:
        """
        This is essentially the "backup" step of MCTS, which updates the running reward
        trackers with the reward from the last trajectory.

        For all nodes, it sets the values for pdvn_mcts_v_syn / pdvn_mcts_v_cost
        to be the running averages of the rewards from all trajectories.
        For OrNodes this is exactly what is written between equations 6-7 in Liu et al 2023.
        For AndNodes, this formula is equivalent for the cost reward,
        but very slightly different for the synthesis reward: it ends up being the average
        of products rather than the product of averages. This form is slightly cleaner and
        should make no difference in practice.
        """
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
        node.data["mcts_value"] = -(
            node.data["pdvn_mcts_v_syn"] * node.data["pdvn_mcts_v_cost"]
            + (1 - node.data["pdvn_mcts_v_syn"]) * self.c_dead
        )
        node.num_visit += 1


def pdvn_min_cost_update(node: ANDOR_NODE, graph: AndOrGraph):
    """
    Update function to compute "pdvn_min_syn_cost" for a given node.
    The pdvn_min_syn_cost is the cost of the minimum synthesis route to synthesize
    a molecule, where the cost of a synthesis route is the sum
    of "pdvn_reaction_cost" for each reaction in the route.
    """
    if isinstance(node, AndNode):
        new_value = node.data["pdvn_reaction_cost"] + sum(
            c.data["pdvn_min_syn_cost"] for c in graph.successors(node)
        )
    elif isinstance(node, OrNode):
        # Cost is the minimum "pdvn_min_syn_cost" of all children
        # or 0 if the molecule is purchasable, otherwise infinity.
        # NOTE: could later generalize to non-zero purchasable molecule cost.
        possible_costs = [0.0 if node.mol.metadata["is_purchasable"] else math.inf]
        possible_costs.extend([c.data["pdvn_min_syn_cost"] for c in graph.successors(node)])
        new_value = min(possible_costs)
    else:
        raise TypeError(f"Unexpected node type: {type(node)}")

    # Do update and return whether the value changed
    old_value = node.data.get("pdvn_min_syn_cost")
    node.data["pdvn_min_syn_cost"] = new_value
    return old_value is None or not math.isclose(old_value, new_value)


def pdvn_extract_training_data(graph: AndOrGraph) -> PDVNSearchData:
    """
    Given an AndOrGraph, extract training data for the PDVN model.
    This data includes:
    - for each molecule in the graph, whether it was solved, and if it was not solved whether it was a dead end.
    - for each solved molecule, the minimum synthesis cost for that molecule.
    - for each solved molecule which is not purchasable, the reaction(s) used for the minimum cost synthesis route.
    """

    # Ensure depth is set for all nodes
    run_message_passing(graph, graph.nodes(), update_fns=[depth_update], update_predecessors=False)

    # Run message passing to recursively compute "has solution" and min cost
    run_message_passing(
        graph,
        sorted(graph.nodes(), key=lambda n: -n.depth),
        update_fns=[has_solution_update, pdvn_min_cost_update],  # type: ignore[list-item]
        update_successors=False,
    )

    # For each molecule node in the graph, find whether it is solved and its minimum cost.
    # Because molecules may occur in multiple places in the graph, we track all values
    # encountered and take the best outcome.
    mol_to_all_route_costs = defaultdict(set)
    mol_to_all_has_solution = defaultdict(set)
    for node in graph.nodes():
        if isinstance(node, OrNode):
            # Does it have a solution?
            if node.has_solution:
                syn_outcome = SynthesizabilityOutcome.SOLN_FOUND
            elif node.is_expanded and len(list(graph.successors(node))) == 0:
                # Being expanded but having no children implies this is a dead end
                syn_outcome = SynthesizabilityOutcome.DEAD_END
            else:
                syn_outcome = SynthesizabilityOutcome.NO_SOLN_FOUND
            mol_to_all_has_solution[node.mol].add(syn_outcome)

            # What is the minimum synthesis cost?
            # Check that it is consistent with has solution
            min_syn_cost = node.data["pdvn_min_syn_cost"]
            if syn_outcome <= 0:
                assert math.isinf(min_syn_cost)
            mol_to_all_route_costs[node.mol].add(min_syn_cost)

            del syn_outcome, min_syn_cost
    mol_to_min_route_cost = {mol: min(costs) for mol, costs in mol_to_all_route_costs.items()}
    mol_to_solution_status = {
        mol: max(has_solution) for mol, has_solution in mol_to_all_has_solution.items()
    }
    del mol_to_all_route_costs, mol_to_all_has_solution

    # For each molecule, find all reactions in the graph which achieve the minimum synthesis cost
    mol_to_reactions_for_min_syn = defaultdict(set)
    for node in graph.nodes():
        if (
            isinstance(node, OrNode)
            and mol_to_solution_status[node.mol] == SynthesizabilityOutcome.SOLN_FOUND
        ):
            for and_child in graph.successors(node):
                assert isinstance(and_child, AndNode)
                if math.isclose(
                    and_child.data["pdvn_min_syn_cost"], mol_to_min_route_cost[node.mol]
                ):
                    mol_to_reactions_for_min_syn[node.mol].add(and_child.reaction)

    # Make return object and do some checks
    output = PDVNSearchData(
        mol_to_synthesizability=mol_to_solution_status,
        mol_to_min_syn_cost={mol: c for mol, c in mol_to_min_route_cost.items() if c < math.inf},
        mol_to_reactions_for_min_syn=mol_to_reactions_for_min_syn,
    )
    assert set(output.mol_to_synthesizability.keys()) >= set(
        output.mol_to_min_syn_cost.keys()
    ), "The solution status of every molecule with a min synthesis cost should be recorded"
    assert set(output.mol_to_min_syn_cost.keys()) >= set(
        output.mol_to_reactions_for_min_syn.keys()
    ), "Every molecule with reactions should have a min synthesis cost"
    return output
