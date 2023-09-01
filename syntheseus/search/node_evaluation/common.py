"""Common node evaluation functions."""

from collections.abc import Sequence

from syntheseus.search.chem import BackwardReaction
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
    def __init__(self, **kwargs) -> None:
        super().__init__(return_log=True, **kwargs)

    def _get_reaction(self, node: AndNode, graph) -> BackwardReaction:
        return node.reaction

    def _evaluate_nodes(self, nodes, graph=None) -> Sequence[float]:
        return [-v for v in super()._evaluate_nodes(nodes, graph)]


class ReactionModelProbPolicy(ReactionModelBasedEvaluator[MolSetNode]):
    def __init__(
        self,
        normalize: bool = True,
        clip_probability_min: float = 0.0,
        clip_probability_max: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            return_log=False,
            normalize=normalize,
            clip_probability_min=clip_probability_min,
            clip_probability_max=clip_probability_max,
            **kwargs,
        )

    def _get_reaction(self, node: MolSetNode, graph) -> BackwardReaction:
        parents = list(graph.predecessors(node))
        assert len(parents) == 1, "Graph must be a tree"

        return graph._graph.edges[parents[0], node]["reaction"]
