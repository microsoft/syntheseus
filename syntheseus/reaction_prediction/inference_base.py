from pathlib import Path
from typing import Optional, Union

from syntheseus.interface.models import (
    BackwardReactionModel,
    ForwardReactionModel,
    InputType,
    ReactionModel,
    ReactionType,
)
from syntheseus.reaction_prediction.utils.downloading import get_default_model_dir_from_cache


class ExternalReactionModel(ReactionModel[InputType, ReactionType]):
    """Base class for the external reaction models, abstracting out common functinality."""

    def __init__(
        self, model_dir: Optional[Union[str, Path]] = None, device: Optional[str] = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        import torch

        self.model_dir = Path(model_dir or self.get_default_model_dir())
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_default_model_dir(self) -> Path:
        model_dir = get_default_model_dir_from_cache(self.name, is_forward=self.is_forward())

        if model_dir is None:
            raise ValueError(
                f"Could not obtain a default checkpoint for model {self.name}, "
                "please provide an explicit value for `model_dir`"
            )

        return model_dir

    @property
    def name(self) -> str:
        return self.__class__.__name__.removesuffix("Model")


class ExternalBackwardReactionModel(ExternalReactionModel, BackwardReactionModel):
    pass


class ExternalForwardReactionModel(ExternalReactionModel, ForwardReactionModel):
    pass
