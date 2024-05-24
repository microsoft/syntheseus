import math
from typing import Callable, List, Optional, Sequence

import torch
from more_itertools import chunked
from torch import nn

from syntheseus.interface.models import InputType, ReactionModel, ReactionType


class ChunkedModel(nn.Module):
    def __init__(self, model: ReactionModel[InputType, ReactionType], batch_size: int) -> None:
        self.model = model
        self.batch_size = batch_size

    def __call__(
        self, inputs: list[InputType], num_results: Optional[int] = None
    ) -> list[Sequence[ReactionType]]:
        return sum(
            [self.model(batch, num_results) for batch in chunked(inputs, self.batch_size)], []
        )


class ParallelReactionModel(ReactionModel[InputType, ReactionType]):
    """Wraps an arbitrary `ReactionModel` to enable multi-GPU inference.

    Unlike most off-the-shelf multi-GPU approaches (e.g. strategies in `pytorch_lightning`,
    `nn.DataParallel`, `nn.DistributedDataParallel`), this class only handles inference (not
    training), and because of that it can be much looser in terms of the constraints the
    parallelized model has to satisfy. It also works with lists of inputs (chunking them up
    appropriately), whereas other approaches usually only work with tensors.
    """

    def __init__(
        self, *args, model_fn: Callable, devices: List, batch_size: Optional[int] = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self._devices = devices

        if batch_size is not None:
            # If `batch_size` is set, this means that batches provided to `ParallelReactionModel`
            # should not be passed to the replicas directly, but they should be further chunked into
            # chunks of size at most `batch_size`. We override `__call__` of the models below, so
            # that `torch.nn.parallel.parallel_apply` can be agnostic to the extra chunking.
            orig_model_fn = model_fn
            model_fn = lambda device: ChunkedModel(orig_model_fn(device=device), batch_size)

        self._model_replicas = [model_fn(device=device) for device in devices]

    def _get_reactions(
        self, inputs: List[InputType], num_results: int
    ) -> List[Sequence[ReactionType]]:
        # Chunk up the inputs into (roughly) equal-sized chunks.
        chunk_size = math.ceil(len(inputs) / len(self._devices))
        input_chunks = list((input,) for input in chunked(inputs, chunk_size))

        # If `len(inputs)` is not divisible by `len(self._devices)` the last chunk may end up empty.
        num_chunks = len(input_chunks)

        outputs = torch.nn.parallel.parallel_apply(
            self._model_replicas[:num_chunks],
            input_chunks,
            tuple({"num_results": num_results} for _ in range(num_chunks)),
            self._devices[:num_chunks],
        )

        # Contatenate all outputs from the replicas.
        return sum(outputs, [])

    def is_forward(self) -> bool:
        first_model = self._model_replicas[0]

        if isinstance(first_model, ChunkedModel):
            first_model = first_model.model

        return first_model.is_forward()
