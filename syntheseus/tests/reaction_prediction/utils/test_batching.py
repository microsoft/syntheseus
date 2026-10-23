from __future__ import annotations

import math
import queue
import threading
from concurrent.futures import CancelledError, ThreadPoolExecutor
from typing import Sequence

import pytest

from syntheseus import BackwardReactionModel, Bag, ForwardReactionModel, Molecule, Reaction
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.reaction_prediction.filters.forward import ForwardReactionFilterModel
from syntheseus.reaction_prediction.filters.wrapper import FilteredBackwardReactionModel
from syntheseus.reaction_prediction.utils.batching import (
    BrokeredBackwardReactionModel,
    BrokeredForwardReactionModel,
    InferenceBroker,
)


class RecordingModel(BackwardReactionModel):
    def __init__(
        self, fail: bool = False, release: threading.Event | None = None, **kwargs
    ) -> None:
        super().__init__(use_cache=False, **kwargs)
        self.fail = fail
        self.release = release
        self.started = threading.Event()
        self.calls: list[tuple[list[Molecule], int]] = []

    def _get_reactions(
        self, inputs: list[Molecule], num_results: int
    ) -> list[Sequence[SingleProductReaction]]:
        self.calls.append((inputs, num_results))
        self.started.set()
        if self.release is not None:
            assert self.release.wait(timeout=5)
        if self.fail:
            raise RuntimeError("inference failed")
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


@pytest.mark.parametrize("abort", [False, True])
def test_failed_admission_cancels_queued_inputs(monkeypatch, abort) -> None:
    release, blocked = threading.Event(), threading.Event()
    backend = RecordingModel(release=release)
    broker = InferenceBroker(backend, 1, 0, 1)
    molecules = [Molecule("C" * length) for length in range(2, 5)]
    put = broker._queue.put_nowait

    def observe_backpressure(ticket):
        try:
            put(ticket)
        except queue.Full:
            if ticket.input == molecules[-1]:
                blocked.set()
            raise

    monkeypatch.setattr(broker._queue, "put_nowait", observe_backpressure)
    with ThreadPoolExecutor(max_workers=1) as executor, broker:
        call = executor.submit(broker, molecules, 1)
        try:
            assert backend.started.wait(timeout=5)
            assert blocked.wait(timeout=5)
            broker.close(cancel_pending=abort, wait=False)
            with pytest.raises(RuntimeError, match="not running"):
                call.result(timeout=5)
        finally:
            release.set()
    assert backend.calls == [(molecules[:1], 1)]


def test_batches_inputs_and_preserves_order() -> None:
    backend = RecordingModel()
    molecules = [Molecule("C" * length) for length in range(2, 6)]
    with InferenceBroker(backend, 4, 0.5, 4) as broker:
        outputs = broker(molecules, num_results=3)
    assert broker.batch_sizes == [4]
    assert backend.calls == [(molecules, 3)]
    assert [output[0].product for output in outputs] == molecules


def test_batches_concurrent_callers() -> None:
    backend = RecordingModel()
    molecules = [Molecule("CC"), Molecule("CCC")]
    barrier = threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as executor, InferenceBroker(backend, 2, 1, 2) as broker:

        def call(molecule):
            barrier.wait(timeout=5)
            return broker([molecule])

        outputs = list(executor.map(call, molecules, timeout=5))
    assert broker.batch_sizes == [2]
    assert [output[0][0].product for output in outputs] == molecules


@pytest.mark.parametrize("wait_s", [0, 0.01])
def test_splits_large_calls_and_finishes_partial_batches(wait_s) -> None:
    backend = RecordingModel()
    molecules = [Molecule("C" * length) for length in range(2, 9)]
    with InferenceBroker(backend, 3, wait_s, 2) as broker:
        outputs = broker(molecules, num_results=1)
    assert sum(broker.batch_sizes) == len(molecules)
    assert all(0 < size <= 3 for size in broker.batch_sizes)
    assert [output[0].product for output in outputs] == molecules


def test_separates_incompatible_result_counts() -> None:
    backend = RecordingModel()
    inputs = [[Molecule("CC")], [Molecule("CCC")]]
    with ThreadPoolExecutor(max_workers=2) as executor, InferenceBroker(
        backend, 8, 0.05, 2
    ) as broker:
        outputs = list(executor.map(broker, inputs, [1, 2], timeout=5))
    assert [output[0][0].product for output in outputs] == [mols[0] for mols in inputs]
    assert sorted(num_results for _, num_results in backend.calls) == [1, 2]


def test_equal_inputs_keep_caller_metadata() -> None:
    backend = RecordingModel()
    first = Molecule("CC", metadata={"supplier": "first"})
    second = Molecule("CC", metadata={"supplier": "second"})
    with InferenceBroker(backend, 2, 0.05, 2) as broker:
        outputs = broker([first, second], num_results=1)
    assert [output[0].product.metadata["supplier"] for output in outputs] == ["first", "second"]
    assert broker.batch_sizes == [1, 1]


def test_local_caches_counters_resets_and_mutable_results() -> None:
    backend = RecordingModel()
    molecule = Molecule("CC")
    with InferenceBroker(backend, 2, 0.01, 2) as broker:
        first = BrokeredBackwardReactionModel(broker, use_cache=True)
        second = BrokeredBackwardReactionModel(broker, use_cache=True)
        first_output = first([molecule])
        second_output = second([molecule])
        assert first([molecule]) == first_output
        assert first.num_calls() == second.num_calls() == 1
        first.reset()
        assert first.num_calls() == 0
        assert second.num_calls() == 1
        first([molecule])
    first_output[0][0].metadata["probability"] = 0.0
    next(iter(first_output[0][0].reactants)).metadata["supplier"] = "changed"
    assert second_output[0][0].metadata["probability"] == 1.0
    assert "supplier" not in next(iter(second_output[0][0].reactants)).metadata
    assert len(backend.calls) == 3


def test_broker_preserves_backend_default_result_count() -> None:
    backend = RecordingModel(default_num_results=2)
    with InferenceBroker(backend, 1, 0, 1) as broker:
        inputs = [Molecule("CC")]
        assert broker(inputs)
        assert broker(inputs, num_results=None)
        assert broker(inputs, num_results=3)
    assert [num_results for _, num_results in backend.calls] == [2, 2, 3]


def test_facade_preserves_backend_defaults_and_cache_policy() -> None:
    backend = RecordingModel(default_num_results=2, count_cache_in_num_calls=True, max_cache_size=1)
    with InferenceBroker(backend, 1, 0, 1) as broker:
        model = BrokeredBackwardReactionModel(broker, use_cache=True)
        molecule = Molecule("CC")
        model([molecule])
        model([molecule])
        assert model.num_calls() == 2
        assert model.num_calls(count_cache=False) == 1
        model([Molecule("CCC")])
        assert model.cache_size == 1
    assert all(num_results == 2 for _, num_results in backend.calls)


def test_validates_output_count() -> None:
    class InvalidModel(RecordingModel):
        def __call__(self, inputs, num_results=None):
            return []

    with InferenceBroker(InvalidModel(), 1, 0, 1) as broker:
        with pytest.raises(RuntimeError, match="0 outputs for 1 inputs"):
            broker([Molecule("CC")], num_results=1)


def test_cancellation_checks_cache_hits_without_faking_counts() -> None:
    cancel_event = threading.Event()
    with InferenceBroker(RecordingModel(), 1, 0, 1) as broker:
        model = BrokeredBackwardReactionModel(broker, cancel_event, use_cache=True)
        molecule = Molecule("CC")
        model([molecule])
        cancel_event.set()
        with pytest.raises(CancelledError):
            model([molecule])
        assert model.num_calls() == 1


def test_cancellation_during_inference_counts_completed_calls() -> None:
    release, cancel = threading.Event(), threading.Event()
    backend = RecordingModel(release=release)
    with ThreadPoolExecutor(max_workers=1) as executor, InferenceBroker(backend, 1, 0, 1) as broker:
        model = BrokeredBackwardReactionModel(broker, cancel, use_cache=True)
        call = executor.submit(model, [Molecule("CC")])
        try:
            assert backend.started.wait(timeout=5)
            cancel.set()
            release.set()
            with pytest.raises(CancelledError):
                call.result(timeout=5)
        finally:
            release.set()
        assert model.num_calls() == 1


@pytest.mark.parametrize("mode", ["drain", "abort", "backend_failure"])
def test_shutdown_and_failure_reach_waiting_callers(mode) -> None:
    release = threading.Event()
    backend = RecordingModel(fail=mode == "backend_failure", release=release)
    broker = InferenceBroker(backend, 2, 30, 2)
    rejection = "inference failed" if mode == "backend_failure" else "not running"

    with ThreadPoolExecutor(max_workers=2) as executor, broker:
        try:
            running = executor.submit(broker, [Molecule("CC")], 1)
            waiting = executor.submit(broker, [Molecule("CCC")], 2)
            assert backend.started.wait(timeout=5)
            if mode == "backend_failure":
                release.set()
            else:
                broker.close(cancel_pending=mode == "abort", wait=False)
                assert not running.done()
            release.set()
            for call in [running, waiting]:
                if mode == "drain":
                    assert call.result(timeout=5)
                else:
                    error = CancelledError if mode == "abort" else RuntimeError
                    with pytest.raises(error):
                        call.result(timeout=5)
            with pytest.raises(RuntimeError, match=rejection):
                broker([Molecule("CCC")])
        finally:
            release.set()
    assert len(backend.calls) == (2 if mode == "drain" else 1)


def test_rejects_work_outside_context_and_cannot_restart() -> None:
    broker = InferenceBroker(RecordingModel(), 1, 0, 1)
    with pytest.raises(RuntimeError, match="not running"):
        broker([Molecule("CC")])
    with broker:
        assert broker([Molecule("CC")])
    with pytest.raises(RuntimeError, match="not running"):
        broker([Molecule("CCC")])
    with pytest.raises(RuntimeError, match="cannot be restarted"):
        with broker:
            pass


@pytest.mark.parametrize(
    ("batch_size", "wait_s", "queue_size"),
    [(0, 0, 1), (1, -1, 1), (1, math.inf, 1), (1, math.nan, 1), (1, 0, 0)],
)
def test_rejects_invalid_configuration(batch_size, wait_s, queue_size) -> None:
    with pytest.raises(ValueError):
        InferenceBroker(RecordingModel(), batch_size, wait_s, queue_size)


def test_rejects_shared_backend_cache() -> None:
    backend = RecordingModel()
    backend.reset(use_cache=True)
    with pytest.raises(ValueError, match="caching disabled"):
        InferenceBroker(backend, 1, 0, 1)


def test_forward_filtering_has_caller_local_state() -> None:
    class ForwardModel(ForwardReactionModel):
        def _get_reactions(
            self, inputs: list[Bag[Molecule]], num_results: int
        ) -> list[Sequence[Reaction]]:
            return [[Reaction(reactants=input, products=Bag([Molecule("CC")]))] for input in inputs]

    with InferenceBroker(RecordingModel(), 2, 0.01, 2) as backward_broker, InferenceBroker(
        ForwardModel(use_cache=False), 2, 0.01, 2
    ) as forward_broker:
        models = [
            FilteredBackwardReactionModel(
                backward_model=BrokeredBackwardReactionModel(backward_broker, use_cache=True),
                filter_models={
                    "forward": ForwardReactionFilterModel(
                        forward_model=BrokeredForwardReactionModel(forward_broker, use_cache=True),
                        top_k=1,
                    )
                },
            )
            for _ in range(2)
        ]
        assert models[0]([Molecule("CC")])[0]
        assert not models[1]([Molecule("CCC")])[0]
        assert models[0].acceptance_rate == 1.0
        assert models[1].acceptance_rate == 0.0
        models[0].reset()
        assert models[1].num_calls() == 1
        assert models[1].acceptance_rate == 0.0
