import logging
from typing import Union

from omegaconf import OmegaConf

from syntheseus.interface.models import ReactionModel
from syntheseus.reaction_prediction.inference.config import BackwardModelConfig, ForwardModelConfig

logger = logging.getLogger(__file__)


def get_model(
    config: Union[BackwardModelConfig, ForwardModelConfig],
    batch_size: int,
    num_gpus: int,
    **model_kwargs,
) -> ReactionModel:
    # Check that model kwargs don't overlap
    overlapping_kwargs = set(config.model_kwargs.keys()) & set(model_kwargs.keys())
    if overlapping_kwargs:
        raise ValueError(f"Model kwargs overlap: {overlapping_kwargs}")

    def model_fn(device):
        return config.model_class.value(
            model_dir=OmegaConf.select(config, "model_dir"),
            device=device,
            **config.model_kwargs,
            **model_kwargs,
        )

    if num_gpus == 0:
        return model_fn("cpu")
    elif num_gpus == 1:
        return model_fn("cuda:0")
    else:
        if batch_size < num_gpus:
            raise ValueError(f"Cannot split batch of size {batch_size} across {num_gpus} GPUs")

        batch_size_per_gpu = batch_size // num_gpus

        if batch_size_per_gpu < 16:
            logger.warning(f"Batch size per GPU is very small: ~{batch_size_per_gpu}")

        try:
            from syntheseus.reaction_prediction.utils.parallel import ParallelReactionModel
        except ModuleNotFoundError:
            raise ValueError("Multi-GPU evaluation is only supported for torch-based models")

        return ParallelReactionModel(
            model_fn=model_fn, devices=[f"cuda:{idx}" for idx in range(num_gpus)]
        )
