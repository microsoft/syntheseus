"""Common node evaluation functions."""

from syntheseus.search.node_evaluation.base import NoCacheNodeEvaluator


class ConstantNodeEvaluator(NoCacheNodeEvaluator):
    def __init__(self, constant: float, **kwargs):
        super().__init__(**kwargs)
        self.constant = constant

    def _evaluate_nodes(self, nodes, graph=None):
        return [self.constant] * len(nodes)


class HasSolutionValueFunction(NoCacheNodeEvaluator):
    def _evaluate_nodes(self, nodes, graph=None):
        return [float(n.has_solution) for n in nodes]
