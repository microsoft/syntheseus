import math

from syntheseus.search.graph.and_or import AndNode, OrNode
from syntheseus.search.graph.base_graph import RetrosynthesisSearchGraph
from syntheseus.search.graph.message_passing import run_message_passing
from syntheseus.search.graph.molset import MolSetNode
from syntheseus.search.graph.node import BaseGraphNode


def get_first_solution_time(graph: RetrosynthesisSearchGraph) -> float:
    """Get the time of the first solution. Also sets 'first_solution_time' node attribute."""
    run_message_passing(
        graph=graph,
        nodes=list(graph._graph.nodes()),
        update_fns=[first_solution_time_update],
        update_successors=False,  # only affects predecessor nodes
    )
    return graph.root_node.data["first_solution_time"]


def first_solution_time_update(node: BaseGraphNode, graph: RetrosynthesisSearchGraph) -> bool:
    NO_SOLUTION_TIME = math.inf  # being unsolved = inf time until solution found

    # Calculate "intrinsic solution time"
    if node._has_intrinsic_solution():
        intrinsic_solution_time = node.data["analysis_time"]
    else:
        intrinsic_solution_time = NO_SOLUTION_TIME

    # Calculate solution age from children
    children_soln_time_list = [
        c.data.get("first_solution_time", NO_SOLUTION_TIME) for c in graph.successors(node)
    ]
    if len(children_soln_time_list) == 0:
        children_solution_time = NO_SOLUTION_TIME
    elif isinstance(node, (OrNode, MolSetNode)):
        # Or node is first solved when one child is solved,
        # so its solution time is the min of its children's
        children_solution_time = min(children_soln_time_list)
    elif isinstance(node, AndNode):
        # AndNode requires all children to be solved,
        # so it is first solved when the LAST child is solved
        children_solution_time = max(children_soln_time_list)
    else:
        raise TypeError(f"Node type {type(node)} not supported.")

    # Min solution time is time of first intrinsic solution or solution from children
    new_min_soln_time = min(intrinsic_solution_time, children_solution_time)

    # Correct one case that can arise with loops: the children could potentially
    # be solved before this node was created.
    # Ensure that new min soln time is at least this node's age!
    new_min_soln_time = max(new_min_soln_time, node.data["analysis_time"])

    # Perform update
    old_min_solution_time = node.data.get("first_solution_time")
    node.data["first_solution_time"] = new_min_soln_time
    return old_min_solution_time is None or not math.isclose(
        old_min_solution_time, new_min_soln_time
    )
