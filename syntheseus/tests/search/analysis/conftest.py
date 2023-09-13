from __future__ import annotations

import pytest

from syntheseus.search.algorithms.breadth_first import AndOr_BreadthFirstSearch
from syntheseus.search.analysis.route_extraction import iter_routes_cost_order
from syntheseus.search.graph.and_or import AndNode, AndOrGraph
from syntheseus.search.graph.molset import MolSetNode
from syntheseus.search.graph.route import SynthesisGraph
from syntheseus.tests.search.conftest import RetrosynthesisTask


def set_uniform_costs(graph) -> None:
    """Set a unit cost of 1 for all nodes with reactions."""
    for node in graph.nodes():
        if isinstance(node, (MolSetNode, AndNode)):
            node.data["route_cost"] = 1.0
        else:
            node.data["route_cost"] = 0.0


@pytest.fixture
def andor_graph_with_many_routes(retrosynthesis_task6: RetrosynthesisTask) -> AndOrGraph:
    task = retrosynthesis_task6
    alg = AndOr_BreadthFirstSearch(
        reaction_model=task.reaction_model, mol_inventory=task.inventory, unique_nodes=True
    )
    output_graph, _ = alg.run_from_mol(task.target_mol)
    assert len(output_graph) == 278  # make sure number of nodes is always the same
    set_uniform_costs(output_graph)
    return output_graph


@pytest.fixture
def sample_synthesis_routes(andor_graph_with_many_routes: AndOrGraph) -> list[SynthesisGraph]:
    """Return 11 synthesis routes extracted from the graph of length <= 3."""
    output = list(
        iter_routes_cost_order(andor_graph_with_many_routes, max_routes=10_000, stop_cost=4.0)
    )
    assert len(output) == 11
    return [
        andor_graph_with_many_routes.to_synthesis_graph(route)  # type: ignore  # node type unclear
        for route in output
    ]
