from __future__ import annotations

import abc
import datetime
import math
from collections.abc import Collection
from dataclasses import dataclass, field

from syntheseus.interface.typed_dict import TypedDict


class _NodeData_Time(TypedDict, total=False):
    """Holds optional data about the time a node was created."""

    # How many times has the rxn model been called when this node was created?
    num_calls_rxn_model: int

    # How many times has the value function been called when this node was created?
    num_calls_value_function: int


class _NodeData_Algorithms(TypedDict, total=False):
    """Holds optional data used by specific algorithms."""

    # ==================================================
    # General
    # ==================================================
    policy_score: float

    # ==================================================
    # Retro*
    # ==================================================
    retro_star_min_cost: float  # minimum cost found so far
    retro_star_reaction_number: float
    reaction_number_estimate: float
    retro_star_value: float
    retro_star_rxn_cost: float
    retro_star_mol_cost: float

    # ==================================================
    # MCTS
    # ==================================================
    mcts_value: float
    mcts_prev_reward: float  # the most recent reward received

    # ==================================================
    # PDVN MCTS
    # ==================================================
    pdvn_reaction_cost: float
    pdvn_mcts_v_syn: float
    pdvn_mcts_v_cost: float
    pdvn_mcts_prev_reward_syn: float
    pdvn_mcts_prev_reward_cost: float
    pdvn_min_syn_cost: float


class _NodeData_Analysis(TypedDict, total=False):
    """Holds optional data used during analysis of search results."""

    analysis_time: float  # Used to hold a node's creation time (measured any way) for analysis purposes
    first_solution_time: float  # time of first solution (according to analysis_time)
    route_cost: float  # non-negative cost that this node contributes to the entire route


class NodeData(_NodeData_Time, _NodeData_Algorithms, _NodeData_Analysis):
    """Holds all kinds of node data."""

    pass


@dataclass
class BaseGraphNode(abc.ABC):
    # Whether the node is "solved" (has a synthesis route leading to it)
    has_solution: bool = False

    # How many times has the node been "visited".
    # The meaning of a "visit" will be different for different algorithms.
    num_visit: int = 0

    # How "deep" is this node, i.e. the length of the path from the root node to this node.
    # It is initialized to inf to indicate "not set" (and this is the only value which will be
    # stable with graphs with no root node where depth is ill-defined)
    depth: int = math.inf  # type: ignore

    # Whether the node has been expanded
    is_expanded: bool = False

    # Time when this node was created (used for analysis of search results).
    creation_time: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

    # Any other node data, stored as a TypedDict to allow arbitrary values to be tracked
    # while also allowing type-checking.
    data: NodeData = field(default_factory=lambda: NodeData())

    def __eq__(self, other):
        # No comparison of node values, only identity.
        return self is other

    def __hash__(self):
        # Hash nodes based on id:
        # this ensures distinct nodes always have a distinct hash.
        return id(self)

    @abc.abstractmethod
    def _has_intrinsic_solution(self) -> bool:
        """Whether this node has a solution without considering its children."""
        raise NotImplementedError

    @abc.abstractmethod
    def _has_solution_from_children(self, children: Collection[BaseGraphNode]) -> bool:
        """Whether this node has a solution, exclusively considering its children."""
        raise NotImplementedError
