from __future__ import annotations

import math
import threading
from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import ExitStack
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
