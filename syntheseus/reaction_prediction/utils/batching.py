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
from typing import Any, Generic, Optional, Sequence

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

    def close(self, cancel_pending: bool = False) -> None:
        """Reject new work, finish running inference, and resolve all submitted futures."""
        with self._state_lock:
            if cancel_pending:
                self._cancel_pending.set()
            self._closing.set()
            thread = self._thread
        if thread is not None:
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
        pending: Optional[_InferenceTicket[InputType, ReactionType]] = None
        while pending is not None or not (self._closing.is_set() and self._queue.empty()):
            if pending is not None:
                ticket = pending
                pending = None
            else:
                try:
                    ticket = self._queue.get(timeout=0.05)
                except queue.Empty:
                    continue

            if not ticket.future.set_running_or_notify_cancel():
                continue
            batch = [ticket]
            try:
                deadline = ticket.queued_at + self._batch_wait_s
                while len(batch) < self._batch_size and not self._cancel_pending.is_set():
                    remaining = deadline - time.monotonic()
                    try:
                        if remaining <= 0 or self._closing.is_set():
                            next_ticket = self._queue.get_nowait()
                        else:
                            next_ticket = self._queue.get(timeout=min(remaining, 0.05))
                    except queue.Empty:
                        if remaining > 0 and not self._closing.is_set():
                            continue
                        break

                    # Model cache keys ignore metadata: equal inputs from different callers
                    # must not be deduplicated together, especially with stochastic models.
                    if next_ticket.num_results != ticket.num_results or any(
                        item.input == next_ticket.input for item in batch
                    ):
                        pending = next_ticket
                        break
                    if next_ticket.future.set_running_or_notify_cancel():
                        batch.append(next_ticket)

                if self._cancel_pending.is_set():
                    for item in batch:
                        item.future.set_exception(CancelledError())
                    continue

                outputs = self.model([item.input for item in batch], num_results=ticket.num_results)
                if len(outputs) != len(batch):
                    raise RuntimeError(
                        f"Model returned {len(outputs)} outputs for {len(batch)} inputs"
                    )
                outputs = [copy.deepcopy(output) for output in outputs]
                self.batch_sizes.append(len(batch))
                for item, output in zip(batch, outputs):
                    if self._cancel_pending.is_set():
                        item.future.set_exception(CancelledError())
                    else:
                        item.future.set_result(output)
            except BaseException as error:
                logger.exception("Shared reaction inference failed")
                with self._state_lock:
                    self._failure = error
                    self._closing.set()
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(error)
                if pending is not None:
                    if pending.future.set_running_or_notify_cancel():
                        pending.future.set_exception(error)
                    pending = None
                while True:
                    try:
                        queued = self._queue.get_nowait()
                    except queue.Empty:
                        return
                    if queued.future.set_running_or_notify_cancel():
                        queued.future.set_exception(error)


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
