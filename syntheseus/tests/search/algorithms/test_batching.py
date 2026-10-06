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


def _synchronize_first_requests(broker, monkeypatch) -> threading.Event:
    """Keep the worker idle until both searches have submitted their first request."""
    release = threading.Event()
    queued = threading.Barrier(2)
    original_get = broker._queue.get
    original_submit = broker.submit

    def get(*args, **kwargs):
        assert release.wait(timeout=5)
        return original_get(*args, **kwargs)

    def submit(inputs, num_results):
        futures = original_submit(inputs, num_results)
        if not release.is_set():
            queued.wait(timeout=5)
            release.set()
        return futures

    monkeypatch.setattr(broker._queue, "get", get)
    monkeypatch.setattr(broker, "submit", submit)
    return release


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
        [(indices[parent], indices[child]) for parent, child in graph._graph.edges()],
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
def test_concurrent_searches_match_sequential_searches(monkeypatch, algorithm_class, call_limit):
    sequential = []
    for target in TARGETS:
        model = LinearMoleculesToyModel(allow_substitution=False, use_cache=True)
        search = _make_search(algorithm_class, model, limit_reaction_model_calls=call_limit)
        graph, _ = search.run_from_mol(Molecule(target))
        sequential.append((graph, model.num_calls()))

    backend = LinearMoleculesToyModel(allow_substitution=False, use_cache=False)
    broker = InferenceBroker(backend, 2, 0, 2)
    release = _synchronize_first_requests(broker, monkeypatch)
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
            try:
                graphs = [future.result(timeout=10)[0] for future in futures]
            finally:
                release.set()

        assert broker.batch_sizes[0] == 2
        for graph, model, search, (expected_graph, expected_calls) in zip(
            graphs, models, searches, sequential
        ):
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
            assert not hasattr(search, "_start_time")
        models[0].reset()
        assert models[0].num_calls() == 0
        assert models[1].num_calls() == sequential[1][1]
    assert broker._thread is not None and not broker._thread.is_alive()


@pytest.mark.parametrize("algorithm_class", ALGORITHMS)
@pytest.mark.parametrize("fail", [False, True], ids=["caller-cancellation", "backend-failure"])
def test_search_cleanup_during_brokered_inference(monkeypatch, algorithm_class, fail):
    started = threading.Event()
    release_inference = threading.Event()
    cancel = threading.Event()

    class BlockingModel(LinearMoleculesToyModel):
        def _get_reactions(self, inputs, num_results):
            if self.num_calls() == 0:
                assert len(inputs) == 2
            elif not started.is_set():
                started.set()
                assert release_inference.wait(timeout=5)
                if fail:
                    raise RuntimeError("inference failed during search")
            return super()._get_reactions(inputs, num_results)

    backend = BlockingModel(allow_substitution=False, use_cache=False)
    broker = InferenceBroker(backend, 2, 0, 2)
    release_worker = _synchronize_first_requests(broker, monkeypatch)
    second_requests = threading.Barrier(3)
    caller_state = threading.local()
    original_submit = broker.submit

    def submit(inputs, num_results):
        futures = original_submit(inputs, num_results)
        caller_state.calls = getattr(caller_state, "calls", 0) + 1
        if caller_state.calls == 2:
            second_requests.wait(timeout=5)
        return futures

    monkeypatch.setattr(broker, "submit", submit)
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
            futures = [
                executor.submit(search.run_from_graph, graph)
                for search, graph in zip(searches, graphs)
            ]
            try:
                assert started.wait(timeout=5)
                # Both searches have expanded their roots and are waiting on another inference.
                second_requests.wait(timeout=5)
                assert all(graph.root_node.is_expanded and len(graph) > 1 for graph in graphs)
                if not fail:
                    cancel.set()
                release_inference.set()
                with pytest.raises(RuntimeError if fail else CancelledError):
                    futures[0].result(timeout=5)
                if fail:
                    with pytest.raises(RuntimeError, match="inference failed during search"):
                        futures[1].result(timeout=5)
                    with pytest.raises(RuntimeError, match="inference failed during search"):
                        broker.submit([Molecule("CC")], 1)
                    assert all(model.num_calls() == 1 for model in models)
                else:
                    futures[1].result(timeout=5)
                    baseline = _make_search(
                        algorithm_class,
                        LinearMoleculesToyModel(allow_substitution=False, use_cache=True),
                        limit_reaction_model_calls=4,
                    )
                    expected_graph, _ = baseline.run_from_mol(Molecule(TARGETS[1]))
                    assert _graph_signature(graphs[1]) == _graph_signature(expected_graph)
                    assert models[1].num_calls() == baseline.reaction_model.num_calls()
                    root_only_search = _make_search(
                        algorithm_class,
                        LinearMoleculesToyModel(allow_substitution=False, use_cache=True),
                        limit_reaction_model_calls=1,
                    )
                    root_only_graph, _ = root_only_search.run_from_mol(Molecule(TARGETS[0]))
                    assert _graph_signature(graphs[0]) == _graph_signature(root_only_graph)
                    # Completed inference counts even when cancellation prevents graph expansion.
                    assert models[0].num_calls() == 2
                assert all(not hasattr(search, "_start_time") for search in searches)
                assert all(future.done() for future in futures)
            finally:
                release_worker.set()
                release_inference.set()
    assert broker._thread is not None and not broker._thread.is_alive()
