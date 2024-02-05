from __future__ import annotations

import math

# NOTE: Collection imported here instead of from collections.abc
# to make casting work for python <3.9
from typing import (
    Collection,
    Optional,
    Sequence,
    cast,
)

from syntheseus.search.algorithms.base import AndOrSearchAlgorithm
from syntheseus.search.algorithms.best_first.base import GeneralBestFirstSearch
from syntheseus.search.algorithms.mixins import ValueFunctionMixin
from syntheseus.search.graph.and_or import ANDOR_NODE, AndNode, AndOrGraph, OrNode
from syntheseus.search.graph.message_passing import run_message_passing
from syntheseus.search.node_evaluation.base import BaseNodeEvaluator, NoCacheNodeEvaluator


class MolIsPurchasableCost(NoCacheNodeEvaluator[OrNode]):
    def _evaluate_nodes(  # type: ignore[override]
        self,
        nodes: Sequence[OrNode],
        graph: Optional[AndOrGraph] = None,
    ) -> list[float]:
        return [0.0 if node.mol.metadata.get("is_purchasable") else math.inf for node in nodes]


class RetroStarSearch(
    AndOrSearchAlgorithm[int],
    GeneralBestFirstSearch[AndOrGraph],
    ValueFunctionMixin[OrNode],
):
    def __init__(
        self,
        *args,
        and_node_cost_fn: BaseNodeEvaluator[AndNode],
        or_node_cost_fn: Optional[BaseNodeEvaluator[OrNode]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if or_node_cost_fn is None:
            or_node_cost_fn = MolIsPurchasableCost()
        self.or_node_cost_fn = or_node_cost_fn
        self.and_node_cost_fn = and_node_cost_fn

    @property
    def reaction_number_estimator(self) -> BaseNodeEvaluator[OrNode]:
        """Alias for value function (they use this term in the paper)"""
        return self.value_function

    def priority_function(self, node: ANDOR_NODE, _: AndNode) -> float:  # type: ignore[override]
        return node.data["retro_star_value"]

    def setup(self, graph: AndOrGraph) -> None:
        # If there is only one node, set its reaction number estimate to 0.
        # This saves a call to the value function
        if len(graph) == 1:
            graph.root_node.data.setdefault("reaction_number_estimate", 0.0)

        return super().setup(graph)

    def set_node_values(  # type: ignore[override]
        self, nodes: Collection[ANDOR_NODE], graph: AndOrGraph
    ) -> Collection[ANDOR_NODE]:
        # Call superclass
        output_nodes = super().set_node_values(nodes, graph)
        del nodes  # unused

        # Fill in node costs and reaction number estimates
        self._set_or_node_costs(
            or_nodes=[
                node
                for node in output_nodes
                if isinstance(node, OrNode) and "retro_star_mol_cost" not in node.data
            ],
            graph=graph,
        )
        self._set_and_node_costs(
            and_nodes=[
                node
                for node in output_nodes
                if isinstance(node, AndNode) and "retro_star_rxn_cost" not in node.data
            ],
            graph=graph,
        )
        self._set_reaction_number_estimate(  # only for leaf nodes
            or_nodes=[
                node
                for node in output_nodes
                if isinstance(node, OrNode)
                and "reaction_number_estimate" not in node.data
                and self.can_expand_node(node, graph)
            ],
            graph=graph,
        )

        # Run updates of reaction number and retro-star value
        return self._run_retro_star_updates(output_nodes, graph)

    def _set_or_node_costs(self, or_nodes: Sequence[OrNode], graph: AndOrGraph) -> None:
        costs = self.or_node_cost_fn(or_nodes, graph=graph)
        assert len(costs) == len(or_nodes)
        for node, cost in zip(or_nodes, costs):
            node.data["retro_star_mol_cost"] = cost

    def _set_and_node_costs(self, and_nodes: Sequence[AndNode], graph: AndOrGraph) -> None:
        costs = self.and_node_cost_fn(and_nodes, graph=graph)
        assert len(costs) == len(and_nodes)
        for node, cost in zip(and_nodes, costs):
            node.data["retro_star_rxn_cost"] = cost

    def _set_reaction_number_estimate(self, or_nodes: Sequence[OrNode], graph: AndOrGraph) -> None:
        costs = self.reaction_number_estimator(or_nodes, graph=graph)
        assert len(costs) == len(or_nodes)
        for node, cost in zip(or_nodes, costs):
            node.data["reaction_number_estimate"] = cost

    def _run_retro_star_updates(
        self, nodes: Collection[ANDOR_NODE], graph: AndOrGraph
    ) -> Collection[ANDOR_NODE]:
        # Initialize all reaction numbers and retro star values
        for node in nodes:
            node.data.setdefault("retro_star_min_cost", math.inf)
            node.data.setdefault("retro_star_reaction_number", math.inf)
            node.data.setdefault("retro_star_value", math.inf)
        nodes_to_update = set(cast(Collection[ANDOR_NODE], nodes))

        # NOTE: the following updates assume that depth is set correctly.

        # Perform bottom-up update of `retro_star_min_cost`,
        # sorting by decreasing depth and not updating children for efficiency
        # (min cost depends only on children)
        nodes_to_update.update(
            cast(  # mypy doesn't know that `run_message_passing` returns a `Collection[ANDOR_NODE]`
                Collection[ANDOR_NODE],
                run_message_passing(
                    graph=graph,
                    nodes=sorted(nodes_to_update, key=lambda node: node.depth, reverse=True),
                    update_fns=[min_cost_update],  # type: ignore[list-item]  # confusion about AndOrGraph type
                    update_predecessors=True,
                    update_successors=False,
                ),
            )
        )

        # Perform bottom-up update of `reaction number`,
        # sorting by decreasing depth and not updating children for efficiency
        # (reaction number depends only on children)
        nodes_to_update.update(
            cast(  # mypy doesn't know that `run_message_passing` returns a `Collection[ANDOR_NODE]`
                Collection[ANDOR_NODE],
                run_message_passing(
                    graph=graph,
                    nodes=sorted(nodes_to_update, key=lambda node: node.depth, reverse=True),
                    update_fns=[reaction_number_update],  # type: ignore[list-item]  # confusion about AndOrGraph type
                    update_predecessors=True,
                    update_successors=False,
                ),
            )
        )

        # Perform top-down update of retro-star value,
        # sorting by increasing depth and not updating parents for efficiency
        # (retro star value depends only on parents)
        nodes_to_update.update(
            cast(
                Collection[ANDOR_NODE],
                run_message_passing(
                    graph=graph,
                    nodes=sorted(nodes_to_update, key=lambda node: node.depth, reverse=False),
                    update_fns=[retro_star_value_update],  # type: ignore[list-item]  # confusion about AndOrGraph type
                    update_predecessors=False,
                    update_successors=True,
                ),
            )
        )

        return nodes_to_update


def min_cost_update(node: ANDOR_NODE, graph: AndOrGraph) -> bool:
    """
    Updates a node's `retro_star_min_cost` value (minimum cost route found so far).

    Returns whether the value changed.
    """
    if isinstance(node, AndNode):
        new_cost = node.data["retro_star_rxn_cost"] + sum(
            c.data["retro_star_min_cost"] for c in graph.successors(node)
        )
    elif isinstance(node, OrNode):
        possible_costs = [node.data["retro_star_mol_cost"]] + [
            c.data["retro_star_min_cost"] for c in graph.successors(node)
        ]
        new_cost = min(possible_costs)
    else:
        raise TypeError(f"Unexpected node type: {type(node)}")

    # Do update and return whether the value changed
    old_cost = node.data["retro_star_min_cost"]
    node.data["retro_star_min_cost"] = new_cost
    return not math.isclose(new_cost, old_cost)


def reaction_number_update(node: ANDOR_NODE, graph: AndOrGraph) -> bool:
    """
    Updates a node's "reaction number", which is the current minimum cost
    estimate of synthesizing a molecule from everything below it.
    Returns whether the node's reaction number was updated.
    """
    if isinstance(node, AndNode):
        # Reaction number from equation 7 in Retro*
        new_rn = node.data["retro_star_rxn_cost"] + sum(
            c.data["retro_star_reaction_number"] for c in graph.successors(node)
        )
    elif isinstance(node, OrNode):
        # Reaction number is the minimum the molecule's purchase cost
        # and the cost of all child synthesis paths,
        # and potentially its reaction number estimate
        possible_costs = [node.data["retro_star_mol_cost"]]
        if node.is_expanded:
            # If the node is expanded, the cost of each child is also an option
            possible_costs.extend(
                [c.data["retro_star_reaction_number"] for c in graph.successors(node)]
            )
        elif "reaction_number_estimate" in node.data:
            # Otherwise the cost of the reaction number estimate is an option.
            # By design, it will only be present if the node can be expanded
            possible_costs.append(node.data["reaction_number_estimate"])
        new_rn = min(possible_costs)
    else:
        raise TypeError(f"Unexpected node type: {type(node)}")

    # Do update and return whether the value changed
    old_rn = node.data["retro_star_reaction_number"]
    node.data["retro_star_reaction_number"] = new_rn
    return not math.isclose(new_rn, old_rn)


def retro_star_value_update(node: ANDOR_NODE, graph: AndOrGraph) -> bool:
    """
    Updates a node's "retro_star_value",
    which is the lowest total cost of any tree containing this node,
    rooted at the root node, assuming that the current costs of each node
    are correct (which is probably not the case for unexpanded nodes).
    This is called V(m|T) in the original Retro* paper (Chen et al 2020).

    Returns whether the node's retro_star_value changed.
    """

    parents = list(graph.predecessors(node))
    if isinstance(node, AndNode):
        # Cost is parent's value, - any contributions from other AND branches,
        # + the current reaction number
        assert len(parents) == 1
        parent = parents[0]
        assert isinstance(parent, OrNode)
        new_value = (
            parent.data["retro_star_value"]
            - parent.data["retro_star_reaction_number"]
            + node.data["retro_star_reaction_number"]
        )

        # Special cases to prevent NaNs
        # Could happen if things are initialized as infs,
        # or in certain cases with non-purchasable molecules.
        # In both cases the cause is inf-inf = nan
        if math.isnan(new_value):
            new_value = math.inf

    elif isinstance(node, OrNode):
        # r* Value estimate is parent's value (this has no double counting)
        # Except the root node: it's r* value estimate is just its RN
        if len(parents) == 0:
            # Root node
            new_value = node.data["retro_star_reaction_number"]
        elif len(parents) == 1:
            # r* is parent's r*
            parent = parents[0]
            assert isinstance(parent, AndNode)
            new_value = parent.data["retro_star_value"]
        else:
            raise ValueError(
                f"Nodes with multiple parents not supported. {node} has {len(parents)} parents."
            )

    else:
        raise TypeError("Unexpected node type")

    # Do update and return whether the value changed
    old_value = node.data["retro_star_value"]
    node.data["retro_star_value"] = new_value
    return not math.isclose(new_value, old_value)
