import pytest

from syntheseus.interface.bag import Bag
from syntheseus.interface.molecule import Molecule
from syntheseus.reaction_prediction.inference.base import ExternalBackwardReactionModel
from syntheseus.reaction_prediction.inference.config import BackwardModelClass
from syntheseus.reaction_prediction.utils.testing import are_single_step_models_installed

pytestmark = pytest.mark.skipif(
    not are_single_step_models_installed(),
    reason="Model tests require all single-step models to be installed",
)


MODEL_CLASSES_TO_TEST = set(BackwardModelClass) - {BackwardModelClass.GLN}


@pytest.fixture(scope="module", params=list(MODEL_CLASSES_TO_TEST) * 2)
def model(request) -> ExternalBackwardReactionModel:
    model_cls = request.param.value
    return model_cls()


def test_call(model: ExternalBackwardReactionModel) -> None:
    [result] = model([Molecule("Cc1ccc(-c2ccc(C)cc2)cc1")], num_results=20)
    model_predictions = [prediction.reactants for prediction in result]

    # Prepare some coupling reactions that are reasonable predictions for the product above.
    expected_predictions = [
        Bag([Molecule(f"Cc1ccc({leaving_group_1})cc1"), Molecule(f"Cc1ccc({leaving_group_2})cc1")])
        for leaving_group_1 in ["Br", "I"]
        for leaving_group_2 in ["B(O)O", "I", "[Mg+]"]
    ]

    # The model should recover at least two (out of six) in its top-20.
    assert len(set(expected_predictions) & set(model_predictions)) >= 2


def test_misc(model: ExternalBackwardReactionModel) -> None:
    import torch

    assert isinstance(model.name, str)
    assert isinstance(model.get_model_info(), dict)
    assert model.is_backward() is not model.is_forward()

    for p in model.get_parameters():
        assert isinstance(p, torch.Tensor)
