from __future__ import annotations

import threading
from concurrent.futures import CancelledError, ThreadPoolExecutor
from typing import Collection, cast

import pytest

from syntheseus.interface.molecule import Molecule
from syntheseus.reaction_prediction.inference.toy_models import LinearMoleculesToyModel
from syntheseus.reaction_prediction.utils.batching import (
    BrokeredBackwardReactionModel,
    InferenceBroker,
)
from syntheseus.search.algorithms.best_first.retro_star import RetroStarSearch
from syntheseus.search.algorithms.breadth_first import AndOr_BreadthFirstSearch
from syntheseus.search.analysis.route_extraction import iter_routes_cost_order
from syntheseus.search.graph.and_or import ANDOR_NODE, AndNode, AndOrGraph, OrNode
from syntheseus.search.mol_inventory import SmilesListInventory
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator

ALGORITHMS = [AndOr_BreadthFirstSearch, RetroStarSearch]
TARGETS = ["COCS", "CCOC"]


def _make_search(algorithm_class, model, **kwargs):
    if algorithm_class is RetroStarSearch:
        kwargs.update(
            and_node_cost_fn=ConstantNodeEvaluator(1.0),
            value_function=ConstantNodeEvaluator(0.0),
        )
    return algorithm_class(
        reaction_model=model,
        mol_inventory=SmilesListInventory(["C", "O", "S"]),
        limit_iterations=1000,
        **kwargs,
    )


def _graph_signature(graph: AndOrGraph):
    """Compare topology and search accounting without wall-clock timestamps or object identity."""
    nodes = list(graph.nodes())
    indices = {node: index for index, node in enumerate(nodes)}
    return (
        [
            (
                node.mol.smiles if isinstance(node, OrNode) else node.reaction.reaction_smiles,
                node.depth,
                node.is_expanded,
                node.has_solution,
                node.data["num_calls_rxn_model"],
            )
            for node in nodes
        ],
        [
            (indices[parent], indices[child])
            for parent in nodes
            for child in graph.successors(parent)
        ],
    )


def _routes(graph: AndOrGraph):
    for node in graph.nodes():
        node.data["route_cost"] = float(isinstance(node, AndNode))
    return [
        graph.to_synthesis_graph(cast(Collection[ANDOR_NODE], route))
        for route in iter_routes_cost_order(graph, max_routes=100)
    ]


@pytest.mark.parametrize("algorithm_class", ALGORITHMS)
@pytest.mark.parametrize("call_limit", [1, 4, 100])
def test_concurrent_searches_match_sequential_searches(algorithm_class, call_limit):
    sequential = []
    for target in TARGETS:
        model = LinearMoleculesToyModel(allow_substitution=False, use_cache=True)
        search = _make_search(algorithm_class, model, limit_reaction_model_calls=call_limit)
        graph, _ = search.run_from_mol(Molecule(target))
        sequential.append((graph, model.num_calls()))

    backend = LinearMoleculesToyModel(allow_substitution=False, use_cache=False)
    broker = InferenceBroker(backend, batch_size=2, batch_wait_s=0.01, max_queue_size=2)
    with broker:
        models = [BrokeredBackwardReactionModel(broker, use_cache=True) for _ in TARGETS]
        searches = [
            _make_search(algorithm_class, model, limit_reaction_model_calls=call_limit)
            for model in models
        ]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(search.run_from_mol, Molecule(target))
                for search, target in zip(searches, TARGETS)
            ]
            graphs = [future.result(timeout=10)[0] for future in futures]

        for graph, model, (expected_graph, expected_calls) in zip(graphs, models, sequential):
            graph.assert_validity()
            assert _graph_signature(graph) == _graph_signature(expected_graph)
            assert model.num_calls() == expected_calls <= call_limit
            if call_limit == 1:
                assert model.num_calls() == 1
            actual_routes, expected_routes = _routes(graph), _routes(expected_graph)
            assert len(actual_routes) == len(expected_routes)
            assert all(route in expected_routes for route in actual_routes)
            if call_limit == 100:
                assert actual_routes
                assert graph.root_node.has_solution
        models[0].reset()
        assert models[0].num_calls() == 0
        assert models[1].num_calls() == sequential[1][1]


@pytest.mark.parametrize("algorithm_class", ALGORITHMS)
def test_search_cleanup_during_brokered_inference(algorithm_class):
    started = threading.Event()
    release_inference = threading.Event()
    cancel = threading.Event()

    class BlockingModel(LinearMoleculesToyModel):
        def _get_reactions(self, inputs, num_results):
            if not started.is_set() and any(
                "S" in mol.smiles and mol.smiles not in TARGETS for mol in inputs
            ):
                started.set()
                assert release_inference.wait(timeout=5)
            return super()._get_reactions(inputs, num_results)

    backend = BlockingModel(allow_substitution=False, use_cache=False)
    broker = InferenceBroker(backend, batch_size=2, batch_wait_s=0.01, max_queue_size=2)
    with broker:
        models = [
            BrokeredBackwardReactionModel(broker, cancel, use_cache=True),
            BrokeredBackwardReactionModel(broker, use_cache=True),
        ]
        searches = [
            _make_search(
                algorithm_class,
                model,
                limit_reaction_model_calls=4,
                should_cancel=cancel.is_set if index == 0 else None,
            )
            for index, model in enumerate(models)
        ]
        graphs = [
            search.create_graph(Molecule(target)) for search, target in zip(searches, TARGETS)
        ]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(searches[0].run_from_graph, graphs[0])]
            try:
                assert started.wait(timeout=5)
                assert graphs[0].root_node.is_expanded and len(graphs[0]) > 1
                futures.append(executor.submit(searches[1].run_from_graph, graphs[1]))
                cancel.set()
                release_inference.set()
                with pytest.raises(CancelledError):
                    futures[0].result(timeout=5)
                futures[1].result(timeout=5)
                for graph in graphs:
                    graph.assert_validity()
                baseline = _make_search(
                    algorithm_class,
                    LinearMoleculesToyModel(allow_substitution=False, use_cache=True),
                    limit_reaction_model_calls=4,
                )
                expected_graph, _ = baseline.run_from_mol(Molecule(TARGETS[1]))
                assert _graph_signature(graphs[1]) == _graph_signature(expected_graph)
                assert models[1].num_calls() == baseline.reaction_model.num_calls()
                # Completed inference counts even when cancellation prevents graph expansion.
                assert 2 <= models[0].num_calls() < searches[0].limit_reaction_model_calls
            finally:
                release_inference.set()


@pytest.mark.parametrize("algorithm_class", ALGORITHMS)
def test_backend_failure_reaches_concurrent_searches(algorithm_class):
    class FailingModel(LinearMoleculesToyModel):
        def _get_reactions(self, inputs, num_results):
            raise RuntimeError("inference failed during search")

    backend = FailingModel(use_cache=False)
    with InferenceBroker(backend, batch_size=2, batch_wait_s=0.01, max_queue_size=2) as broker:
        searches = [
            _make_search(algorithm_class, BrokeredBackwardReactionModel(broker, use_cache=True))
            for _ in TARGETS
        ]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(search.run_from_mol, Molecule(target))
                for search, target in zip(searches, TARGETS)
            ]
            for future in futures:
                with pytest.raises(RuntimeError, match="inference failed during search"):
                    future.result(timeout=5)
