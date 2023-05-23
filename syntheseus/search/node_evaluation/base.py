from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Generic, Optional, TypeVar

from syntheseus.search.graph.base_graph import RetrosynthesisSearchGraph
from syntheseus.search.graph.node import BaseGraphNode

NodeType = TypeVar("NodeType", bound=BaseGraphNode)


class BaseNodeEvaluator(Generic[NodeType], abc.ABC):
    """
    Parent class for functions which assign values to nodes.
    This includes value functions, policies, reward functions, etc.

    Also counts number of times it has been called.
    However, unlike for reaction models caching is not implemented by default.
    This is because different value functions might cache different things:
    for example, some might only depend on the node's molecules/reactions,
    while others might depend on the graph structure.
    Therefore this is left to the subclasses to implement.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset()

    def reset(self) -> None:
        """Resets this node evaluator."""
        pass

    @property
    def num_calls(self) -> int:
        """
        Return how many times this node evaluator has been called,
        accounting for caching if that is implemented.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(
        self, nodes: Sequence[NodeType], graph: Optional[RetrosynthesisSearchGraph] = None
    ) -> Sequence[float]:
        """
        Main method for this class which evaluates the nodes.
        Subclasses should put their functionality into this method,
        also including code for counting the number of calls.

        Args:
            nodes: List of nodes to be valued.
            graph: graph containing the nodes. Optional since not all functions will require this.

        Returns:
            A sequence "vals" of the same length as nodes, where "vals[i]"
            is the value estimate for "nodes[i]"
        """
        pass


class NoCacheNodeEvaluator(BaseNodeEvaluator[NodeType]):
    """Subclass which implements counting number of calls with no caching."""

    def reset(self) -> None:
        self._num_calls = 0

    @property
    def num_calls(self) -> int:
        return self._num_calls

    def __call__(
        self, nodes: Sequence[NodeType], graph: Optional[RetrosynthesisSearchGraph] = None
    ) -> Sequence[float]:
        self._num_calls += len(nodes)
        return self._evaluate_nodes(nodes, graph)

    @abc.abstractmethod
    def _evaluate_nodes(
        self, nodes: Sequence[NodeType], graph: Optional[RetrosynthesisSearchGraph] = None
    ) -> Sequence[float]:
        """Override this method to just evaluate the nodes, without counting the number of calls."""
        pass
