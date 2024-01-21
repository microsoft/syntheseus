from __future__ import annotations

import abc
from typing import Generic, Optional, Sequence, TypeVar

import numpy as np

from syntheseus.interface.reaction import SingleProductReaction
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
        if not nodes:  # handle the case when there are no nodes to score
            return []

        self._num_calls += len(nodes)
        return self._evaluate_nodes(nodes, graph)

    @abc.abstractmethod
    def _evaluate_nodes(
        self, nodes: Sequence[NodeType], graph: Optional[RetrosynthesisSearchGraph] = None
    ) -> Sequence[float]:
        """Override this method to just evaluate the nodes, without counting the number of calls."""
        pass


class ReactionModelBasedEvaluator(NoCacheNodeEvaluator[NodeType]):
    """
    Evaluator that computes its value based on the `probability` metadata from the underlying
    reaction objects (with optional clipping and normalization).
    """

    def __init__(
        self,
        return_log: bool,
        normalize: bool = False,
        temperature: float = 1.0,
        clip_probability_min: float = 1e-10,
        clip_probability_max: float = 0.999,
    ) -> None:
        """Initialized the evaluator.

        Args:
            return_log: Whether to return the logarithm of the probability instead of the
                probability itself.
            normalize: Whether to renormalize the output to be a distribution (or logarithms of
                probabilities corresponding to a valid distribution). This is especially useful when
                `temperature != 1.0`.
            temperature: Temperature to apply (i.e. divide the logits by).
            clip_probability_min: Minimum probability to clip to. Should be positive if `return_log`
                or `normalize` is set to avoid NaNs.
            clip_probability_max: Maximum probability to clip to.
        """
        super().__init__()

        assert 0.0 <= clip_probability_min <= clip_probability_max <= 1.0

        if (return_log or normalize) and clip_probability_min == 0.0:
            raise ValueError("Disabling clipping can lead to NaNs")

        self._return_log = return_log
        self._normalize = normalize
        self._temperature = temperature
        self._clip_probability_min = clip_probability_min
        self._clip_probability_max = clip_probability_max

    @abc.abstractmethod
    def _get_reaction(self, node, graph) -> SingleProductReaction:
        pass

    def _get_probability(self, node, graph) -> float:
        metadata = self._get_reaction(node, graph).metadata

        if "probability" not in metadata:
            raise ValueError("Cannot call node evaluator as reaction model probability is not set")
        return metadata["probability"]  # type: ignore

    def _evaluate_nodes(self, nodes, graph=None) -> Sequence[float]:
        probs = np.asarray([self._get_probability(n, graph) for n in nodes])
        probs = np.clip(probs, a_min=self._clip_probability_min, a_max=self._clip_probability_max)

        # Process the output based on the options passed in `__init__`. Note that the handling of
        # temperature and normalization is equivalent in the two branches below; the only difference
        # is one happenging in log space.
        if self._return_log:
            outputs = np.log(probs) / self._temperature
            if self._normalize:
                # Apply `log_softmax` to make `outputs.exp()` a valid probability distribution.
                outputs -= outputs.max()  # shift before taking exp for numerical stability
                outputs -= np.log(np.exp(outputs).sum())
        else:
            outputs = probs ** (1.0 / self._temperature)
            if self._normalize:
                outputs /= outputs.sum()

        return outputs.tolist()
