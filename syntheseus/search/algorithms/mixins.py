from __future__ import annotations

from typing import Generic, TypeVar

from syntheseus.search.algorithms.base import SearchAlgorithm
from syntheseus.search.graph.node import BaseGraphNode
from syntheseus.search.node_evaluation import BaseNodeEvaluator

NodeType = TypeVar("NodeType", bound=BaseGraphNode)


class ValueFunctionMixin(SearchAlgorithm, Generic[NodeType]):
    def __init__(self, *args, value_function: BaseNodeEvaluator[NodeType], **kwargs):
        super().__init__(*args, **kwargs)
        self.value_function = value_function

    def set_node_values(self, nodes, graph):
        output_nodes = super().set_node_values(nodes, graph)
        for node in output_nodes:
            node.data.setdefault("num_calls_value_function", self.value_function.num_calls)
        return output_nodes

    def reset(self) -> None:
        super().reset()
        self.value_function.reset()
