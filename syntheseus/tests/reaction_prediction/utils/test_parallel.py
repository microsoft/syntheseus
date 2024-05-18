import pytest

from syntheseus import Molecule
from syntheseus.tests.cli.test_eval_single_step import DummyModel

try:
    import torch

    from syntheseus.reaction_prediction.utils.parallel import ParallelReactionModel

    torch_available = True
    cuda_available = torch.cuda.is_available()
except ModuleNotFoundError:
    torch_available = False
    cuda_available = False


@pytest.mark.skipif(
    not torch_available, reason="Simple testing of parallel inference requires torch"
)
def test_parallel_reaction_model_cpu() -> None:
    # We cannot really run this on CPU, so just check if the model creation works as normal.
    parallel_model: ParallelReactionModel = ParallelReactionModel(
        model_fn=DummyModel, devices=["cpu"] * 4
    )
    assert parallel_model([]) == []


@pytest.mark.skipif(
    not cuda_available, reason="Full testing of parallel inference requires GPU to be available"
)
def test_parallel_reaction_model_gpu() -> None:
    model = DummyModel()
    parallel_model: ParallelReactionModel = ParallelReactionModel(
        model_fn=DummyModel, devices=["cuda:0"] * 4
    )

    inputs = [Molecule("C" * length) for length in range(1, 6)]
    assert parallel_model(inputs) == model(inputs)
