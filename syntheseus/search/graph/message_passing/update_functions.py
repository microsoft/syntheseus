from syntheseus.search.graph.base_graph import RetrosynthesisSearchGraph
from syntheseus.search.graph.node import BaseGraphNode


def depth_update(node: BaseGraphNode, graph: RetrosynthesisSearchGraph) -> bool:
    parent_depths = [n.depth for n in graph.predecessors(node)]
    if len(parent_depths) == 0:
        new_depth = 0
    else:
        new_depth = min(parent_depths) + 1

    depth_changed = node.depth != new_depth
    node.depth = new_depth
    return depth_changed


def has_solution_update(node: BaseGraphNode, graph: RetrosynthesisSearchGraph) -> bool:
    new_has_solution = node._has_intrinsic_solution() or node._has_solution_from_children(
        list(graph.successors(node))
    )

    old_has_solution = node.has_solution
    node.has_solution = new_has_solution
    return new_has_solution != old_has_solution
