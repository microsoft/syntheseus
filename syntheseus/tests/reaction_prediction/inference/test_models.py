import pytest

from syntheseus.interface.bag import Bag
from syntheseus.interface.molecule import Molecule
from syntheseus.reaction_prediction.inference import ChemformerModel, LocalRetroModel, MEGANModel
from syntheseus.reaction_prediction.inference.base import ExternalBackwardReactionModel

try:
    # Try to import the single-step model repositories to check if they are installed. Technically,
    # it could be the case that these are installed but their dependencies are not, in which case
    # the tests will fail; nevertheless the check below is good enough for our usecase.

    import chemformer  # noqa: F401
    import local_retro  # noqa: F401
    import megan  # noqa: F401

    MODELS_INSTALLED = True
except ModuleNotFoundError:
    MODELS_INSTALLED = False


pytestmark = pytest.mark.skipif(
    not MODELS_INSTALLED, reason="Model tests require all single-step models to be installed"
)


# TODO(kmaziarz): Some of the models (MHNreact, RetroKNN and RootAligned) appear to only work on
# GPU. Make them also work on CPU, then add below.


@pytest.fixture(
    scope="module",
    params=[ChemformerModel, LocalRetroModel, MEGANModel],
)
def model(request) -> ExternalBackwardReactionModel:
    model_cls = request.param
    return model_cls()


def test_call(model: ExternalBackwardReactionModel) -> None:
    [result] = model([Molecule("Cc1ccc(-c2ccc(C)cc2)cc1")], num_results=10)
    model_predictions = [prediction.output for prediction in result.predictions]

    # Prepare some coupling reactions that are reasonable predictions for the product above.
    expected_predictions = [
        Bag([Molecule(f"Cc1ccc({leaving_group_1})cc1"), Molecule(f"Cc1ccc({leaving_group_2})cc1")])
        for leaving_group_1 in ["Br", "I"]
        for leaving_group_2 in ["B(O)O", "I", "[Mg+]"]
    ]

    # The model should recover at least two (out of six) in its top-10.
    assert len(set(expected_predictions) & set(model_predictions)) >= 2


def test_misc(model: ExternalBackwardReactionModel) -> None:
    import torch

    assert isinstance(model.name, str)
    assert isinstance(model.get_model_info(), dict)
    assert model.is_backward() is not model.is_forward()

    for p in model.get_parameters():
        assert isinstance(p, torch.Tensor)
