from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict

from omegaconf import MISSING

from syntheseus.reaction_prediction.inference import (
    ChemformerModel,
    GLNModel,
    Graph2EditsModel,
    LocalRetroModel,
    MEGANModel,
    MHNreactModel,
    RetroKNNModel,
    RootAlignedModel,
)
from retrochimera.inference.smiles_transformer_forward import SmilesTransformerForwardModel


class ForwardModelClass(Enum):
    Chemformer = ChemformerModel
    ForwardChimeraDeNovo = SmilesTransformerForwardModel


class BackwardModelClass(Enum):
    Chemformer = ChemformerModel
    GLN = GLNModel
    Graph2Edits = Graph2EditsModel
    LocalRetro = LocalRetroModel
    MEGAN = MEGANModel
    MHNreact = MHNreactModel
    RetroKNN = RetroKNNModel
    RootAligned = RootAlignedModel


class FilterModelClass(Enum):
    pass


@dataclass
class ModelConfig:
    """Config for loading any reaction models, forward or backward."""

    model_dir: str = MISSING
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ForwardModelConfig(ModelConfig):
    """Config for loading one of the supported forward models."""

    model_class: ForwardModelClass = MISSING


@dataclass
class BackwardModelConfig(ModelConfig):
    """Config for loading one of the supported backward models."""

    model_class: BackwardModelClass = MISSING


@dataclass
class FilterModelConfig(ModelConfig):
    """Config for loading a model used to filter reactions during search."""

    model_class: FilterModelClass = MISSING
