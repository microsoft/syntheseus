from __future__ import annotations

from typing import Generic, TypeVar

from syntheseus.search.algorithms.base import SearchAlgorithm
from syntheseus.search.graph.node import BaseGraphNode
from syntheseus.search.node_evaluation import BaseNodeEvaluator

NodeType = TypeVar("NodeType", bound=BaseGraphNode)


class SearchHeuristicMixin(SearchAlgorithm, Generic[NodeType]):
    def __init__(self, *args, search_heuristic: BaseNodeEvaluator[NodeType], **kwargs):
        super().__init__(*args, **kwargs)
        self.search_heuristic = search_heuristic

    def set_node_values(self, nodes, graph):
        output_nodes = super().set_node_values(nodes, graph)
        for node in output_nodes:
            node.data.setdefault("num_calls_search_heuristic", self.search_heuristic.num_calls)
        return output_nodes

    def reset(self) -> None:
        super().reset()
        self.search_heuristic.reset()
