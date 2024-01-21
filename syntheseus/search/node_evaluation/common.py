"""Common node evaluation functions."""

from __future__ import annotations

from typing import Sequence, Union

from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.search.graph.and_or import AndNode
from syntheseus.search.graph.molset import MolSetNode
from syntheseus.search.node_evaluation.base import NoCacheNodeEvaluator, ReactionModelBasedEvaluator


class ConstantNodeEvaluator(NoCacheNodeEvaluator):
    def __init__(self, constant: float, **kwargs):
        super().__init__(**kwargs)
        self.constant = constant

    def _evaluate_nodes(self, nodes, graph=None):
        return [self.constant] * len(nodes)


class HasSolutionValueFunction(NoCacheNodeEvaluator):
    def _evaluate_nodes(self, nodes, graph=None):
        return [float(n.has_solution) for n in nodes]


class ReactionModelLogProbCost(ReactionModelBasedEvaluator[AndNode]):
    """Evaluator that uses the reactions' negative logprob to form a cost (useful for Retro*)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(return_log=True, **kwargs)

    def _get_reaction(self, node: AndNode, graph) -> SingleProductReaction:
        return node.reaction

    def _evaluate_nodes(self, nodes, graph=None) -> Sequence[float]:
        return [-v for v in super()._evaluate_nodes(nodes, graph)]


class ReactionModelProbPolicy(ReactionModelBasedEvaluator[Union[MolSetNode, AndNode]]):
    """Evaluator that uses the reactions' probability to form a policy (useful for MCTS)."""

    def __init__(self, **kwargs) -> None:
        kwargs["normalize"] = kwargs.get("normalize", True)  # set `normalize = True` by default
        super().__init__(return_log=False, **kwargs)

    def _get_reaction(self, node: Union[MolSetNode, AndNode], graph) -> SingleProductReaction:
        if isinstance(node, MolSetNode):
            parents = list(graph.predecessors(node))
            assert len(parents) == 1, "Graph must be a tree"
            return graph._graph.edges[parents[0], node]["reaction"]
        elif isinstance(node, AndNode):
            return node.reaction
        else:
            raise ValueError(f"ReactionModelProbPolicy does not support nodes of type {type(node)}")
