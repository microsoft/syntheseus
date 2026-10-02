from __future__ import annotations

import json
import math
import pickle
import threading
import time
from concurrent.futures import CancelledError
from pathlib import Path
from typing import Sequence

import pytest
from omegaconf import OmegaConf

from syntheseus import BackwardReactionModel, Bag, ForwardReactionModel, Molecule, Reaction
from syntheseus.cli import search
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.reaction_prediction.inference.config import (
    BackwardModelClass,
    ForwardModelClass,
)


class RecordingBackwardModel(BackwardReactionModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.batches: list[list[str]] = []

    def _get_reactions(
        self, inputs: list[Molecule], num_results: int
    ) -> list[Sequence[SingleProductReaction]]:
        self.batches.append([input.smiles for input in inputs])
        return [
            [
                SingleProductReaction(
                    product=input,
                    reactants=Bag([Molecule("C")]),
                    metadata={"probability": 1.0},
                )
            ]
            for input in inputs
        ]


class RecordingForwardModel(ForwardReactionModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.batches: list[list[Bag[Molecule]]] = []

    def _get_reactions(
        self, inputs: list[Bag[Molecule]], num_results: int
    ) -> list[Sequence[Reaction]]:
        self.batches.append(inputs)
        return [[Reaction(reactants=input, products=Bag([Molecule("CC")]))] for input in inputs]


@pytest.fixture
def backends(monkeypatch):
    backward_models: list[RecordingBackwardModel] = []
    forward_models: list[RecordingForwardModel] = []

    def load_model(config, num_gpus, **kwargs):
        if config.model_class == ForwardModelClass.Chemformer:
            forward = RecordingForwardModel(**kwargs)
            forward_models.append(forward)
            return forward
        backward = RecordingBackwardModel(**kwargs)
        backward_models.append(backward)
        return backward

    monkeypatch.setattr(search, "get_model", load_model)
    return backward_models, forward_models


def _config(tmp_path: Path, targets: list[str], **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    targets_path = tmp_path / "targets.smi"
    targets_path.write_text("".join(f"{target}\n" for target in targets))
    inventory_path = tmp_path / "inventory.smi"
    inventory_path.write_text("C\n")
    config = OmegaConf.create(
        search.SearchConfig(
            model_class=BackwardModelClass.RetroChimera,
            model_dir="unused",
            search_targets_file=str(targets_path),
            inventory_smiles_file=str(inventory_path),
            results_dir=str(tmp_path / "results"),
            append_timestamp_to_dir=False,
            use_gpu=False,
            num_routes_to_plot=0,
            limit_iterations=1,
            inference_batch_wait_s=0.1,
        )
    )
    return OmegaConf.merge(config, kwargs)


def _stats(results_dir: Path, index: int):
    return json.loads((results_dir / str(index) / "stats.json").read_text())


def test_batches_targets_and_preserves_stats_graphs_and_order(tmp_path, backends) -> None:
    targets = ["CC", "CCC", "CCCC", "CCCCC"]
    config = _config(tmp_path, targets, max_active_searches=4, inference_batch_size=4)
    results_dir = search.run_from_config(config)
    backward_models, forward_models = backends
    assert len(backward_models) == 1
    assert not forward_models
    assert not backward_models[0]._use_cache
    assert any(len(batch) > 1 for batch in backward_models[0].batches)
    assert all(len(batch) <= 4 for batch in backward_models[0].batches)
    for index, smiles in enumerate(targets):
        stats = _stats(results_dir, index)
        assert set(stats) == {
            "index",
            "smiles",
            "target_in_inventory",
            "rxn_model_calls_used",
            "num_nodes_in_final_tree",
            "soln_time_rxn_model_calls",
            "soln_time_wallclock",
        }
        assert stats["index"] == index
        assert stats["smiles"] == smiles
        assert stats["rxn_model_calls_used"] == 1
        assert stats["soln_time_rxn_model_calls"] == 1
        assert not (results_dir / str(index) / ".lock").exists()
        with open(results_dir / str(index) / "graph.pkl", "rb") as file:
            graph = pickle.load(file)
        assert graph.root_mol.smiles == smiles
        assert graph.root_node.has_solution
    summary = json.loads((results_dir / "stats.json").read_text())
    assert summary["num_targets"] == summary["num_solved_targets"] == 4
    assert summary["average_rxn_model_calls_used"] == 1
    assert summary["median_soln_time_rxn_model_calls"] == 1


def test_default_serial_mode_bypasses_broker(tmp_path, backends, monkeypatch) -> None:
    def unexpected_broker(*args, **kwargs):
        raise AssertionError("Serial execution must not construct a broker")

    monkeypatch.setattr(search, "InferenceBroker", unexpected_broker)
    results_dir = search.run_from_config(_config(tmp_path, ["CC", "CC"]))
    backward_models, _ = backends
    assert backward_models[0]._use_cache
    assert backward_models[0].batches == [["CC"], ["CC"]]
    assert all(_stats(results_dir, index)["rxn_model_calls_used"] == 1 for index in range(2))


def test_fixed_call_statistics_match_serial_execution(tmp_path, backends) -> None:
    targets = ["CC", "CCC", "CCCC"]
    serial = search.run_from_config(_config(tmp_path / "serial", targets))
    concurrent = search.run_from_config(
        _config(tmp_path / "concurrent", targets, max_active_searches=3)
    )
    for index in range(len(targets)):
        serial_stats = _stats(serial, index)
        concurrent_stats = _stats(concurrent, index)
        assert set(serial_stats) == set(concurrent_stats)
        serial_stats.pop("soln_time_wallclock")
        concurrent_stats.pop("soln_time_wallclock")
        assert serial_stats == concurrent_stats
    assert set(json.loads((serial / "stats.json").read_text())) == set(
        json.loads((concurrent / "stats.json").read_text())
    )


def test_bounds_active_searches_and_isolates_algorithms(tmp_path, backends, monkeypatch) -> None:
    original = search._run_target
    barrier = threading.Barrier(3)
    lock = threading.Lock()
    active = 0
    maximum_active = 0
    algorithms = []

    def run_target(index, smiles, config, algorithm, *args):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            algorithms.append(algorithm)
        try:
            if index < 3:
                barrier.wait(timeout=5)
            return original(index, smiles, config, algorithm, *args)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(search, "_run_target", run_target)
    targets = ["C" * length for length in range(2, 9)]
    search.run_from_config(_config(tmp_path, targets, max_active_searches=3))
    assert maximum_active == 3
    assert len({id(algorithm) for algorithm in algorithms}) == len(targets)
    assert len({id(algorithm.reaction_model) for algorithm in algorithms}) == len(targets)


@pytest.mark.parametrize("remove_stereo", [False, True])
def test_honors_stereo_removal_and_inventory(tmp_path, backends, remove_stereo) -> None:
    targets = ["C[C@H](O)F", "C[C@@H](O)F"]
    config = _config(
        tmp_path,
        targets,
        max_active_searches=2,
        remove_stereo_from_targets=remove_stereo,
    )
    Path(config.inventory_smiles_file).write_text("C\nCC(O)F\n")
    results_dir = search.run_from_config(config)
    for index in range(2):
        stats = _stats(results_dir, index)
        assert stats["target_in_inventory"] == remove_stereo
        assert stats["rxn_model_calls_used"] == (0 if remove_stereo else 1)
        assert ("@" not in stats["smiles"]) == remove_stereo


def test_forward_filtering_keeps_per_target_acceptance_rates(tmp_path, backends) -> None:
    config = _config(
        tmp_path,
        ["CC", "CCC"],
        max_active_searches=2,
        forward_filter=search.ForwardFilterConfig(
            model_class=ForwardModelClass.Chemformer,
            top_k=1,
        ),
    )
    results_dir = search.run_from_config(config)
    backward_models, forward_models = backends
    assert len(backward_models) == len(forward_models) == 1
    assert not forward_models[0]._use_cache
    first, second = [_stats(results_dir, index) for index in range(2)]
    assert first["filter_acceptance_rate"] == 1.0
    assert second["filter_acceptance_rate"] == 0.0
    assert first["filter_acceptance_rate_per_filter"] == {"forward": 1.0}
    assert second["filter_acceptance_rate_per_filter"] == {"forward": 0.0}
    assert math.isfinite(first["soln_time_wallclock"])
    assert math.isinf(second["soln_time_wallclock"])
    summary = json.loads((results_dir / "stats.json").read_text())
    assert summary["num_solved_targets"] == 1
    assert summary["average_filter_acceptance_rate"] == 0.5
    assert summary["average_filter_acceptance_rate_per_filter"] == {"forward": 0.5}


def test_single_target_keeps_original_directory_layout(tmp_path, backends) -> None:
    results_dir = search.run_from_config(_config(tmp_path, ["CC"], max_active_searches=2))
    assert (results_dir / "graph.pkl").is_file()
    assert not (results_dir / "0").exists()
    stats = json.loads((results_dir / "stats.json").read_text())
    assert stats["index"] == 0
    assert stats["smiles"] == "CC"
    assert stats["rxn_model_calls_used"] == 1


def test_resumes_completed_targets_and_rejects_changed_targets(tmp_path, backends) -> None:
    config = _config(tmp_path, ["CC", "CCC", "CCCC"], max_active_searches=2)
    results_dir = search.run_from_config(config)
    completed_stats = [_stats(results_dir, index) for index in range(3)]
    search.run_from_config(config)
    backward_models, _ = backends
    assert not backward_models[-1].batches
    assert [_stats(results_dir, index) for index in range(3)] == completed_stats

    Path(config.search_targets_file).write_text("CC\nCCCCC\nCCCC\n")
    with pytest.raises(RuntimeError, match="does not match"):
        search.run_from_config(config)


def test_failure_cancels_siblings_and_retains_resumable_outputs(
    tmp_path, backends, monkeypatch
) -> None:
    completed = threading.Event()
    original = search._run_target

    def run_target(index, *args):
        stats = original(index, *args)
        if index == 0:
            completed.set()
        return stats

    class FailingModel(RecordingBackwardModel):
        def _get_reactions(self, inputs, num_results):
            if any(input.smiles == "CCC" for input in inputs):
                assert completed.wait(timeout=5)
                raise RuntimeError("inference failed")
            return super()._get_reactions(inputs, num_results)

    def load_model(config, num_gpus, **kwargs):
        return FailingModel(**kwargs)

    config = _config(
        tmp_path,
        ["CC", "CCC", "CCCC"],
        max_active_searches=2,
        inference_batch_size=1,
    )
    results_dir = Path(config.results_dir) / config.model_class.name
    with monkeypatch.context() as patch:
        patch.setattr(search, "_run_target", run_target)
        patch.setattr(search, "get_model", load_model)
        with pytest.raises(RuntimeError, match="inference failed"):
            search.run_from_config(config)
    assert (results_dir / "0" / "stats.json").is_file()
    assert not (results_dir / "0" / ".lock").exists()
    assert (results_dir / "1" / ".lock").is_file()
    assert not (results_dir / "stats.json").exists()
    assert not any(thread.name == "syntheseus-inference" for thread in threading.enumerate())

    search.run_from_config(config)
    assert all(not (results_dir / str(index) / ".lock").exists() for index in range(3))
    summary = json.loads((results_dir / "stats.json").read_text())
    assert summary["num_solved_targets"] == 3


def test_initial_submission_failure_cancels_started_searches(
    tmp_path, backends, monkeypatch
) -> None:
    started = threading.Event()
    cancelled = threading.Event()
    original_filter = search._filter_model
    calls = 0

    def filter_model(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            assert started.wait(timeout=5)
            raise RuntimeError("target setup failed")
        return original_filter(*args)

    def run_target(*args):
        cancel_event = args[-1]
        started.set()
        if cancel_event.wait(timeout=5):
            cancelled.set()
        raise CancelledError()

    monkeypatch.setattr(search, "_filter_model", filter_model)
    monkeypatch.setattr(search, "_run_target", run_target)
    config = _config(tmp_path, ["CC", "CCC"], max_active_searches=2)
    with pytest.raises(RuntimeError, match="target setup failed"):
        search.run_from_config(config)
    assert cancelled.is_set()
    assert not any(thread.name == "syntheseus-inference" for thread in threading.enumerate())


def test_plotting_is_serialized_and_routes_are_saved(tmp_path, backends, monkeypatch) -> None:
    lock = threading.Lock()
    active = 0
    maximum_active = 0
    plotted = []

    def visualize(graph, filename, nodes):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.02)
            plotted.append(filename)
            Path(filename).write_text("plot")
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(search, "VISUALIZATION_CODE_IMPORTED", True)
    monkeypatch.setattr(search, "visualize_andor", visualize, raising=False)
    config = _config(tmp_path, ["CC", "CCC"], max_active_searches=2, num_routes_to_plot=1)
    results_dir = search.run_from_config(config)
    assert maximum_active == 1
    assert len(plotted) == 2
    for index in range(2):
        assert (results_dir / str(index) / "route_0.pkl").is_file()
        assert (results_dir / str(index) / "route_0.pdf").is_file()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_active_searches", 0),
        ("inference_batch_size", 0),
        ("inference_batch_wait_s", -1),
        ("inference_batch_wait_s", math.inf),
        ("inference_batch_wait_s", math.nan),
    ],
)
def test_rejects_invalid_execution_configuration(tmp_path, backends, field, value) -> None:
    config = _config(tmp_path, ["CC"], **{field: value})
    with pytest.raises(ValueError, match=field):
        search.run_from_config(config)
    assert not backends[0]


def test_rejects_empty_target_file(tmp_path, backends) -> None:
    config = _config(tmp_path, [], max_active_searches=2)
    with pytest.raises(ValueError, match="empty"):
        search.run_from_config(config)
    assert not backends[0]
