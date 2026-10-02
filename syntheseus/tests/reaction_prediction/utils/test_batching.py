from __future__ import annotations

import math
import threading
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import ExitStack
from typing import Sequence

import pytest

from syntheseus import BackwardReactionModel, Bag, ForwardReactionModel, Molecule, Reaction
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.reaction_prediction.filters.forward import ForwardReactionFilterModel
from syntheseus.reaction_prediction.filters.wrapper import FilteredBackwardReactionModel
from syntheseus.reaction_prediction.utils import batching
from syntheseus.reaction_prediction.utils.batching import (
    BrokeredBackwardReactionModel,
    BrokeredForwardReactionModel,
    InferenceBroker,
    _can_admit,
    _InferenceTicket,
    _read_timeout,
)


class RecordingModel(BackwardReactionModel):
    def __init__(self, fail: bool = False, **kwargs) -> None:
        super().__init__(use_cache=False, **kwargs)
        self.fail = fail
        self.calls: list[tuple[list[Molecule], int]] = []

    def _get_reactions(
        self, inputs: list[Molecule], num_results: int
    ) -> list[Sequence[SingleProductReaction]]:
        self.calls.append((inputs, num_results))
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


def _ticket(smiles: str, num_results: int) -> _InferenceTicket[Molecule, SingleProductReaction]:
    future: Future[Sequence[SingleProductReaction]] = Future()
    return _InferenceTicket(Molecule(smiles), num_results, 100.0, future)


@pytest.mark.parametrize(
    ("smiles", "num_results", "expected"),
    [("CCCC", 1, True), ("CCCC", 2, False), ("CC", 1, False), ("CCC", 1, False)],
)
def test_admission_policy_is_pure(smiles, num_results, expected) -> None:
    batch = [_ticket("CC", 1), _ticket("CCC", 1)]
    candidate = _ticket(smiles, num_results)
    assert _can_admit(batch, candidate) == expected
    assert len(batch) == 2
    assert all(not ticket.future.running() and not ticket.future.done() for ticket in batch)
    assert not candidate.future.running() and not candidate.future.done()


def test_admission_checks_result_count_before_comparing_inputs() -> None:
    class FailingEquality(Molecule):
        def __eq__(self, other):
            raise AssertionError("Inputs should not be compared")

    ticket = _ticket("CC", 1)
    ticket.input = FailingEquality("CC")
    assert not _can_admit([ticket], _ticket("CCC", 2))


@pytest.mark.parametrize(
    ("now", "closing", "expected"),
    [
        (99.0, False, 0.05),
        (99.98, False, 0.02),
        (100.0, False, 0.0),
        (101.0, False, 0.0),
        (99.0, True, 0.0),
    ],
)
def test_timeout_policy_uses_explicit_clock_and_lifecycle_values(now, closing, expected) -> None:
    timeout = _read_timeout(deadline=100.0, now=now, closing=closing)
    assert timeout == pytest.approx(expected)
    assert 0.0 <= timeout <= 0.05


@pytest.mark.parametrize(("wait_s", "closing"), [(0.0, False), (30.0, True)])
def test_polling_admits_already_queued_tickets(monkeypatch, wait_s, closing) -> None:
    backend = RecordingModel()
    broker = InferenceBroker(backend, 3, wait_s, 3)
    release = threading.Event()
    original_get = broker._queue.get
    first_read = True

    def get(*args, **kwargs):
        nonlocal first_read
        if first_read:
            first_read = False
            assert release.wait(timeout=5)
        return original_get(*args, **kwargs)

    monkeypatch.setattr(broker._queue, "get", get)
    with broker:
        try:
            futures = broker.submit([Molecule("CC"), Molecule("CCC"), Molecule("CCCC")], 1)
            if closing:
                broker.close(wait=False)
        finally:
            release.set()
        assert all(future.result(timeout=5) for future in futures)
    assert broker.batch_sizes == [3]
    assert len(backend.calls) == 1


def test_classification_failure_resolves_unclassified_and_queued_tickets(monkeypatch) -> None:
    release = threading.Event()

    def fail(batch, candidate):
        assert release.wait(timeout=5)
        raise RuntimeError("admission failed")

    monkeypatch.setattr(batching, "_can_admit", fail)
    backend = RecordingModel()
    with InferenceBroker(backend, 2, 0.5, 3) as broker:
        try:
            futures = broker.submit([Molecule("CC"), Molecule("CCC"), Molecule("CCCC")], 1)
        finally:
            release.set()
        for future in futures:
            with pytest.raises(RuntimeError, match="admission failed"):
                future.result(timeout=5)
        with pytest.raises(RuntimeError, match="admission failed"):
            broker.submit([Molecule("CCCCC")], 1)
    assert not backend.calls


def test_failure_retains_deferred_ticket_ownership_across_batches(monkeypatch) -> None:
    release = threading.Event()
    calls = 0

    def admit(batch, candidate):
        nonlocal calls
        calls += 1
        assert release.wait(timeout=5)
        if calls == 2:
            raise RuntimeError("deferred batch failed")
        return _can_admit(batch, candidate)

    monkeypatch.setattr(batching, "_can_admit", admit)
    backend = RecordingModel()
    with InferenceBroker(backend, 3, 0.5, 4) as broker:
        try:
            first = broker.submit([Molecule("CC")], 1)[0]
            remaining = broker.submit([Molecule("CCC"), Molecule("CCCC"), Molecule("CCCCC")], 2)
        finally:
            release.set()
        assert first.result(timeout=5)
        for future in remaining:
            with pytest.raises(RuntimeError, match="deferred batch failed"):
                future.result(timeout=5)
    assert broker.batch_sizes == [1]
    assert len(backend.calls) == 1


def test_initial_read_failure_resolves_queued_tickets(monkeypatch) -> None:
    broker = InferenceBroker(RecordingModel(), 3, 0, 3)
    release = threading.Event()
    original_get = broker._queue.get
    first_read = True

    def get(*args, **kwargs):
        nonlocal first_read
        if first_read:
            first_read = False
            assert release.wait(timeout=5)
            raise RuntimeError("queue read failed")
        return original_get(*args, **kwargs)

    monkeypatch.setattr(broker._queue, "get", get)
    with broker:
        try:
            futures = broker.submit([Molecule("CC"), Molecule("CCC"), Molecule("CCCC")], 1)
        finally:
            release.set()
        for future in futures:
            with pytest.raises(RuntimeError, match="queue read failed"):
                future.result(timeout=5)


@pytest.mark.parametrize("cancel", [True, False])
def test_publication_accounts_for_every_owned_ticket(cancel) -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingModel(RecordingModel):
        def _get_reactions(self, inputs, num_results):
            started.set()
            assert release.wait(timeout=5)
            return super()._get_reactions(inputs, num_results)

    backend = BlockingModel()
    with InferenceBroker(backend, 2, 0.1, 3) as broker:

        def after_first_result(future):
            if cancel:
                broker.close(cancel_pending=True, wait=False)
            else:
                raise SystemExit("publication failed")

        try:
            futures = broker.submit([Molecule("CC"), Molecule("CCC"), Molecule("CCCC")], 1)
            assert started.wait(timeout=5)
            futures[0].add_done_callback(after_first_result)
        finally:
            release.set()
        assert futures[0].result(timeout=5)
        for future in futures[1:]:
            with pytest.raises(CancelledError if cancel else SystemExit):
                future.result(timeout=5)
    assert len(backend.calls) == 1


def test_batches_requests_and_preserves_order() -> None:
    backend = RecordingModel()
    molecules = [Molecule("C" * length) for length in range(2, 6)]
    barrier = threading.Barrier(len(molecules))
    with InferenceBroker(backend, 4, 0.5, 4) as broker:
        models = [BrokeredBackwardReactionModel(broker) for _ in molecules]

        def call(index):
            barrier.wait(timeout=5)
            return models[index]([molecules[index]], num_results=3)

        with ThreadPoolExecutor(max_workers=4) as executor:
            outputs = list(executor.map(call, range(4)))
    assert broker.batch_sizes == [4]
    assert len(backend.calls) == 1
    assert [output[0][0].product for output in outputs] == molecules


def test_flushes_partial_batches_and_splits_large_calls() -> None:
    backend = RecordingModel()
    molecules = [Molecule("C" * length) for length in range(2, 9)]
    with InferenceBroker(backend, 3, 0.01, 2) as broker:
        model = BrokeredBackwardReactionModel(broker)
        outputs = model(molecules)
    assert sum(broker.batch_sizes) == len(molecules)
    assert all(0 < size <= 3 for size in broker.batch_sizes)
    assert [output[0].product for output in outputs] == molecules


def test_separates_incompatible_result_counts() -> None:
    backend = RecordingModel()
    with InferenceBroker(backend, 8, 0.05, 2) as broker:
        first = broker.submit([Molecule("CC")], 1)
        second = broker.submit([Molecule("CCC")], 2)
        assert first[0].result(timeout=5)
        assert second[0].result(timeout=5)
    assert [num_results for _, num_results in backend.calls] == [1, 2]


def test_equal_inputs_keep_caller_metadata() -> None:
    backend = RecordingModel()
    first = Molecule("CC", metadata={"supplier": "first"})
    second = Molecule("CC", metadata={"supplier": "second"})
    with InferenceBroker(backend, 2, 0.05, 2) as broker:
        futures = broker.submit([first, second], 1)
        outputs = [future.result(timeout=5) for future in futures]
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


def test_failure_resolves_all_requests_and_rejects_new_work() -> None:
    backend = RecordingModel(fail=True)
    with InferenceBroker(backend, 2, 0.1, 5) as broker:
        futures = broker.submit([Molecule("C" * length) for length in range(2, 7)], 1)
        for future in futures:
            with pytest.raises(RuntimeError, match="inference failed"):
                future.result(timeout=5)
        with pytest.raises(RuntimeError, match="inference failed"):
            broker.submit([Molecule("CCCCCCC")], 1)
    assert len(backend.calls) == 1


def test_validates_output_count() -> None:
    class InvalidModel(RecordingModel):
        def __call__(self, inputs, num_results=None):
            return []

    with InferenceBroker(InvalidModel(), 1, 0, 1) as broker:
        future = broker.submit([Molecule("CC")], 1)[0]
        with pytest.raises(RuntimeError, match="0 outputs for 1 inputs"):
            future.result(timeout=5)


def test_cancelled_ticket_does_not_poison_broker() -> None:
    backend = RecordingModel()
    with InferenceBroker(backend, 2, 0.1, 3) as broker:
        first = broker.submit([Molecule("CC")], 1)[0]
        cancelled = broker.submit([Molecule("CCC")], 1)[0]
        cancelled.cancel()
        last = broker.submit([Molecule("CCCC")], 1)[0]
        assert first.result(timeout=5)
        assert last.result(timeout=5)
    assert not cancelled.running()


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


def test_close_unblocks_backpressure_and_cancels_pending_work() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingModel(RecordingModel):
        def _get_reactions(self, inputs, num_results):
            started.set()
            assert release.wait(timeout=5)
            return super()._get_reactions(inputs, num_results)

    with InferenceBroker(BlockingModel(), 1, 0, 1) as broker:
        first = broker.submit([Molecule("CC")], 1)[0]
        assert started.wait(timeout=5)
        second = broker.submit([Molecule("CCC")], 1)[0]
        with ThreadPoolExecutor(max_workers=2) as executor:
            blocked = executor.submit(broker.submit, [Molecule("CCCC")], 1)
            closing = executor.submit(broker.close, True)
            try:
                with pytest.raises(RuntimeError, match="not running"):
                    blocked.result(timeout=5)
            finally:
                release.set()
            closing.result(timeout=5)
        for future in [first, second]:
            with pytest.raises(CancelledError):
                future.result(timeout=5)
        with pytest.raises(RuntimeError, match="not running"):
            broker.submit([Molecule("CC")], 1)


def test_close_drains_submitted_work_and_cannot_restart() -> None:
    broker = InferenceBroker(RecordingModel(), 8, 30, 2)
    with pytest.raises(RuntimeError, match="not running"):
        broker.submit([Molecule("CC")], 1)
    with broker:
        futures = broker.submit([Molecule("CC"), Molecule("CCC")], 1)
    assert all(future.result(timeout=5) for future in futures)
    with pytest.raises(RuntimeError, match="cannot be restarted"):
        with broker:
            pass


def test_close_can_cancel_without_waiting_for_running_inference() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingModel(RecordingModel):
        def _get_reactions(self, inputs, num_results):
            started.set()
            assert release.wait(timeout=5)
            return super()._get_reactions(inputs, num_results)

    backend = BlockingModel()
    with InferenceBroker(backend, 1, 0, 1) as broker:
        first = broker.submit([Molecule("CC")], 1)[0]
        assert started.wait(timeout=5)
        pending = broker.submit([Molecule("CCC")], 1)[0]
        try:
            broker.close(cancel_pending=True, wait=False)
            assert not first.done()
            with pytest.raises(RuntimeError, match="not running"):
                broker.submit([Molecule("CCCC")], 1)
        finally:
            release.set()
        for future in [first, pending]:
            with pytest.raises(CancelledError):
                future.result(timeout=5)
    assert len(backend.calls) == 1


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

    backward_backend = RecordingModel()
    forward_backend = ForwardModel(use_cache=False)
    with ExitStack() as stack:
        backward_broker = stack.enter_context(InferenceBroker(backward_backend, 2, 0.01, 2))
        forward_broker = stack.enter_context(InferenceBroker(forward_backend, 2, 0.01, 2))
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
