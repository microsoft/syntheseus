"""Batch inference requests while keeping each caller's model state independent."""

from __future__ import annotations

import copy
import logging
import math
import queue
import threading
import time
from concurrent.futures import CancelledError, Future
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Generic, Iterable, Optional, Sequence

from syntheseus.interface.bag import Bag
from syntheseus.interface.models import (
    BackwardReactionModel,
    ForwardReactionModel,
    InputType,
    ReactionModel,
    ReactionType,
)
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import Reaction, SingleProductReaction

logger = logging.getLogger(__name__)


@dataclass
class _InferenceTicket(Generic[InputType, ReactionType]):
    input: InputType
    num_results: int
    queued_at: float
    future: Future[Sequence[ReactionType]]


def _can_admit(
    batch: Sequence[_InferenceTicket[InputType, ReactionType]],
    candidate: _InferenceTicket[InputType, ReactionType],
) -> bool:
    same_result_count = candidate.num_results == batch[0].num_results
    # Cache keys ignore metadata, so equal inputs must keep separate inference contexts.
    return same_result_count and not any(ticket.input == candidate.input for ticket in batch)


def _read_timeout(deadline: float, now: float, closing: bool) -> float:
    return 0.0 if closing else min(0.05, max(0.0, deadline - now))


class InferenceBroker(Generic[InputType, ReactionType]):
    """Own one inference worker and batch requests from independent model facades.

    The backend must have caching disabled. Enter the broker before submitting work;
    exiting drains submitted work, or cancels pending work when the context raises.
    Running inference is not interruptible.
    """

    def __init__(
        self,
        model: ReactionModel[InputType, ReactionType],
        batch_size: int,
        batch_wait_s: float,
        max_queue_size: int,
    ) -> None:
        if batch_size <= 0 or max_queue_size <= 0:
            raise ValueError("Batch size and queue size must be positive")
        if not math.isfinite(batch_wait_s) or batch_wait_s < 0:
            raise ValueError("Batch wait must be finite and non-negative")
        if model._use_cache:
            raise ValueError("The broker backend must have caching disabled")

        self.model = model
        self._batch_size = batch_size
        self._batch_wait_s = batch_wait_s
        self._queue: queue.Queue[_InferenceTicket[InputType, ReactionType]] = queue.Queue(
            maxsize=max_queue_size
        )
        self._state_lock = threading.Lock()
        self._closing = threading.Event()
        self._cancel_pending = threading.Event()
        self._failure: Optional[BaseException] = None
        self._thread: Optional[threading.Thread] = None
        self.batch_sizes: list[int] = []

    def __enter__(self) -> InferenceBroker[InputType, ReactionType]:
        with self._state_lock:
            if self._thread is not None or self._closing.is_set():
                raise RuntimeError("The inference broker cannot be restarted")
            self._thread = threading.Thread(target=self._run, name="syntheseus-inference")
            self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.close(cancel_pending=exc_type is not None)

    def close(self, cancel_pending: bool = False, wait: bool = True) -> None:
        """Reject new work and optionally wait for inference and submitted futures."""
        with self._state_lock:
            if cancel_pending:
                self._cancel_pending.set()
            self._closing.set()
            thread = self._thread
        if wait and thread is not None:
            thread.join()

    def submit(
        self, inputs: list[InputType], num_results: int
    ) -> list[Future[Sequence[ReactionType]]]:
        futures: list[Future[Sequence[ReactionType]]] = [Future() for _ in inputs]
        try:
            for input, future in zip(inputs, futures):
                ticket = _InferenceTicket(input, num_results, time.monotonic(), future)
                while True:
                    with self._state_lock:
                        if self._failure is not None:
                            raise self._failure
                        if self._thread is None or self._closing.is_set():
                            raise RuntimeError("The inference broker is not running")
                        try:
                            self._queue.put_nowait(ticket)
                        except queue.Full:
                            pass
                        else:
                            break
                    self._closing.wait(timeout=0.01)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        return futures

    def _run(self) -> None:
        deferred: Optional[_InferenceTicket[InputType, ReactionType]] = None
        owned: dict[Future[Sequence[ReactionType]], _InferenceTicket[InputType, ReactionType]] = {}
        try:
            while deferred is not None or not (self._closing.is_set() and self._queue.empty()):
                if deferred is not None:
                    ticket = deferred
                    deferred = None
                else:
                    try:
                        ticket = self._queue.get(timeout=0.05)
                    except queue.Empty:
                        continue

                owned[ticket.future] = ticket
                if not ticket.future.set_running_or_notify_cancel():
                    del owned[ticket.future]
                    continue
                batch = [ticket]
                deadline = ticket.queued_at + self._batch_wait_s
                while len(batch) < self._batch_size and not self._cancel_pending.is_set():
                    timeout = _read_timeout(deadline, time.monotonic(), self._closing.is_set())
                    try:
                        candidate = self._queue.get(timeout=timeout)
                    except queue.Empty:
                        if timeout > 0 and not self._closing.is_set():
                            continue
                        break

                    owned[candidate.future] = candidate
                    if not _can_admit(batch, candidate):
                        deferred = candidate
                        break
                    if candidate.future.set_running_or_notify_cancel():
                        batch.append(candidate)
                    else:
                        del owned[candidate.future]

                if self._cancel_pending.is_set():
                    for item in batch:
                        item.future.set_exception(CancelledError())
                else:
                    outputs = self._predict_batch(batch)
                    self._publish_batch(batch, outputs)
                for item in batch:
                    del owned[item.future]
        except BaseException as error:
            self._fail_owned_and_queued(owned.values(), error)

    def _predict_batch(
        self, batch: Sequence[_InferenceTicket[InputType, ReactionType]]
    ) -> list[Sequence[ReactionType]]:
        outputs = self.model([ticket.input for ticket in batch], num_results=batch[0].num_results)
        if len(outputs) != len(batch):
            raise RuntimeError(f"Model returned {len(outputs)} outputs for {len(batch)} inputs")
        outputs = [copy.deepcopy(output) for output in outputs]
        self.batch_sizes.append(len(batch))
        return outputs

    def _publish_batch(
        self,
        batch: Sequence[_InferenceTicket[InputType, ReactionType]],
        outputs: Sequence[Sequence[ReactionType]],
    ) -> None:
        for ticket, output in zip(batch, outputs):
            if self._cancel_pending.is_set():
                ticket.future.set_exception(CancelledError())
            else:
                ticket.future.set_result(output)

    def _reject_tickets(
        self, tickets: Iterable[_InferenceTicket[InputType, ReactionType]], error: BaseException
    ) -> None:
        for ticket in tickets:
            future = ticket.future
            if not future.done() and (future.running() or future.set_running_or_notify_cancel()):
                future.set_exception(error)

    def _fail_owned_and_queued(
        self, owned: Iterable[_InferenceTicket[InputType, ReactionType]], error: BaseException
    ) -> None:
        logger.exception("Shared inference worker failed")
        with self._state_lock:
            self._failure = error
            self._closing.set()
        self._reject_tickets(owned, error)
        while True:
            try:
                queued = self._queue.get_nowait()
            except queue.Empty:
                return
            self._reject_tickets([queued], error)


class _BrokeredReactionModel(ReactionModel[InputType, ReactionType]):
    def __init__(
        self,
        broker: InferenceBroker[InputType, ReactionType],
        cancel_event: Optional[threading.Event] = None,
        **kwargs: Any,
    ) -> None:
        self._broker = broker
        self._cancel_event = cancel_event
        kwargs.setdefault("default_num_results", broker.model.default_num_results)
        kwargs.setdefault("count_cache_in_num_calls", broker.model.count_cache_in_num_calls)
        kwargs.setdefault("max_cache_size", broker.model._max_cache_size)
        kwargs.setdefault("remove_duplicates", False)
        kwargs.setdefault("remove_product_in_reactants", False)
        super().__init__(**kwargs)

    def __call__(
        self, inputs: list[InputType], num_results: Optional[int] = None
    ) -> list[Sequence[ReactionType]]:
        self._check_cancelled()
        outputs = super().__call__(inputs, num_results=num_results)
        self._check_cancelled()
        return outputs

    def _check_cancelled(self) -> None:
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise CancelledError()

    def _get_reactions(
        self, inputs: list[InputType], num_results: int
    ) -> list[Sequence[ReactionType]]:
        return [future.result() for future in self._broker.submit(inputs, num_results)]

    def is_forward(self) -> bool:
        return self._broker.model.is_forward()

    def get_model_info(self) -> dict[str, Any]:
        return self._broker.model.get_model_info()

    def get_parameters(self):
        return self._broker.model.get_parameters()


class BrokeredBackwardReactionModel(
    _BrokeredReactionModel[Molecule, SingleProductReaction], BackwardReactionModel
):
    """Backward model facade with caller-local cache, counters, and cancellation."""


class BrokeredForwardReactionModel(
    _BrokeredReactionModel[Bag[Molecule], Reaction], ForwardReactionModel
):
    """Forward model facade with caller-local cache, counters, and cancellation."""
