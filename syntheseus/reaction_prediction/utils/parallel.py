import math
from typing import Callable, List

import torch
from more_itertools import chunked

from syntheseus.interface.models import InputType, OutputType, PredictionList, ReactionModel


class ParallelReactionModel(ReactionModel):
    """Wraps an arbitrary `ReactionModel` to enable multi-GPU inference.

    Unlike most off-the-shelf multi-GPU approaches (e.g. strategies in `pytorch_lightning`,
    `nn.DataParallel`, `nn.DistributedDataParallel`), this class only handles inference (not
    training), and because of that it can be much looser in terms of the constraints the
    parallelized model has to satisfy. It also works with lists of inputs (chunking them up
    appropriately), whereas other approaches usually only work with tensors.
    """

    def __init__(self, model_fn: Callable, devices: List) -> None:
        self._devices = devices
        self._model_replicas = [model_fn(device=device) for device in devices]

    def __call__(
        self, inputs: List[InputType], num_results: int
    ) -> List[PredictionList[InputType, OutputType]]:
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
        return self._model_replicas[0].is_forward()
