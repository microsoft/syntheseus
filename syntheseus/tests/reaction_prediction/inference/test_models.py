import threading

import pytest

from syntheseus.interface.bag import Bag
from syntheseus.interface.molecule import Molecule
from syntheseus.reaction_prediction.inference.config import BackwardModelClass, ForwardModelClass
from syntheseus.reaction_prediction.inference_base import ExternalBackwardReactionModel
from syntheseus.reaction_prediction.utils.batching import InferenceBroker
from syntheseus.reaction_prediction.utils.testing import are_single_step_models_installed

pytestmark = pytest.mark.skipif(
    not are_single_step_models_installed(),
    reason="Model tests require all single-step models to be installed",
)


MODEL_CLASSES_TO_TEST = [m for m in BackwardModelClass if m is not BackwardModelClass.GLN]
FORWARD_MODEL_CLASSES_TO_TEST = list(ForwardModelClass)

# Use a single rule application process for template-based models to reduce memory usage.
RULE_SERVER_KWARGS = {"num_processes": 1}
EXTRA_MODEL_KWARGS = {
    BackwardModelClass.RetroChimeraEdit: RULE_SERVER_KWARGS,
    BackwardModelClass.RetroChimera: {"template_localization": RULE_SERVER_KWARGS},
}
EXTRA_FORWARD_MODEL_KWARGS = {
    ForwardModelClass.Chemformer: {"is_forward": True},
}


@pytest.fixture(params=MODEL_CLASSES_TO_TEST)
def model(request) -> ExternalBackwardReactionModel:
    model_cls = request.param.value
    return model_cls(**EXTRA_MODEL_KWARGS.get(request.param, {}))


@pytest.mark.forked
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

    import torch

    # Additionally test some misc properties and methods.

    assert isinstance(model.name, str)
    assert isinstance(model.get_model_info(), dict)
    assert model.is_backward() is not model.is_forward()

    for p in model.get_parameters():
        assert isinstance(p, torch.Tensor)


@pytest.mark.forked
def test_brokered_retrochimera_call(monkeypatch) -> None:
    """Exercise real checkpoint inference and output copying on the broker's worker (CPU or GPU)."""
    model = BackwardModelClass.RetroChimera.value(
        use_cache=False, **EXTRA_MODEL_KWARGS[BackwardModelClass.RetroChimera]
    )
    molecules = [
        Molecule("Cc1ccc(-c2ccc(C)cc2)cc1"),
        Molecule("COc1ccc(-c2ccc(C)cc2)cc1"),
    ]
    broker = InferenceBroker(model, batch_size=2, batch_wait_s=0, max_queue_size=2)
    release = threading.Event()
    original_get = broker._queue.get
    original_predict = model._get_reactions
    inference_threads: list[int] = []

    def get(*args, **kwargs):
        assert release.wait(timeout=5)
        return original_get(*args, **kwargs)

    def predict(inputs, num_results):
        inference_threads.append(threading.get_ident())
        return original_predict(inputs, num_results)

    monkeypatch.setattr(broker._queue, "get", get)
    monkeypatch.setattr(model, "_get_reactions", predict)
    with broker:
        try:
            futures = broker.submit(molecules, num_results=20)
        finally:
            release.set()
        outputs = [future.result(timeout=600) for future in futures]
    assert broker.batch_sizes == [2]
    assert len(inference_threads) == 1
    assert inference_threads[0] != threading.get_ident()
    assert model.cache_size == 0
    assert model.num_calls() == 2
    for molecule, reactions in zip(molecules, outputs):
        assert reactions
        assert all(reaction.product == molecule for reaction in reactions)
        assert all(reaction.product is not molecule for reaction in reactions)
    outputs[0][0].metadata["broker_test"] = True
    assert all("broker_test" not in reaction.metadata for reaction in outputs[1])
    assert broker._thread is not None and not broker._thread.is_alive()


@pytest.mark.parametrize("model_class", FORWARD_MODEL_CLASSES_TO_TEST)
@pytest.mark.forked
def test_forward_call(model_class: ForwardModelClass) -> None:
    forward_model = model_class.value(**EXTRA_FORWARD_MODEL_KWARGS.get(model_class, {}))
    reactants = Bag([Molecule("Cc1ccc(Br)cc1"), Molecule("Cc1ccc(B(O)O)cc1")])
    [result] = forward_model([reactants], num_results=10)

    assert result
    assert all(prediction.reactants == reactants for prediction in result)
    assert forward_model.is_forward()
    assert not forward_model.is_backward()
