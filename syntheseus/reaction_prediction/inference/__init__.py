from syntheseus.reaction_prediction.inference.chemformer import ChemformerModel
from syntheseus.reaction_prediction.inference.gln import GLNModel
from syntheseus.reaction_prediction.inference.graph2edits import Graph2EditsModel
from syntheseus.reaction_prediction.inference.local_retro import LocalRetroModel
from syntheseus.reaction_prediction.inference.megan import MEGANModel
from syntheseus.reaction_prediction.inference.mhnreact import MHNreactModel
from syntheseus.reaction_prediction.inference.retro_knn import RetroKNNModel
from syntheseus.reaction_prediction.inference.root_aligned import RootAlignedModel
from syntheseus.reaction_prediction.inference.toy_models import (
    LinearMoleculesToyModel,
    ListOfReactionsToyModel,
)
from syntheseus.reaction_prediction.utils.misc import get_unavailable_model_class

# RetroChimera is directly built on top of syntheseus, so we import from it directly.
try:
    from retrochimera import RetroChimeraDeNovoModel, RetroChimeraEditModel, RetroChimeraModel
except ImportError:
    RetroChimeraDeNovoModel = get_unavailable_model_class("RetroChimeraDeNovoModel", "retrochimera")
    RetroChimeraEditModel = get_unavailable_model_class("RetroChimeraEditModel", "retrochimera")
    RetroChimeraModel = get_unavailable_model_class("RetroChimeraModel", "retrochimera")


__all__ = [
    "ChemformerModel",
    "GLNModel",
    "Graph2EditsModel",
    "LinearMoleculesToyModel",
    "ListOfReactionsToyModel",
    "LocalRetroModel",
    "MEGANModel",
    "MHNreactModel",
    "RetroChimeraDeNovoModel",
    "RetroChimeraEditModel",
    "RetroChimeraModel",
    "RetroKNNModel",
    "RootAlignedModel",
]
