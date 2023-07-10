import tempfile
from itertools import cycle, islice
from typing import List

import pytest

from syntheseus.interface.bag import Bag
from syntheseus.interface.models import (
    BackwardPrediction,
    BackwardPredictionList,
    BackwardReactionModel,
)
from syntheseus.interface.molecule import Molecule
from syntheseus.reaction_prediction.cli.eval import (
    BackwardModelClass,
    EvalConfig,
    EvalResults,
    get_results,
    print_and_save,
)
from syntheseus.reaction_prediction.utils.metrics import ModelTimingResults


class DummyModel(BackwardReactionModel):
    RESULTS = [
        Bag([Molecule("C"), Molecule("N")]),
        Bag([Molecule("c1ccccc1")]),
        Bag([Molecule("N"), Molecule("C")]),
        Bag([Molecule("NC=O")]),
    ]

    def __call__(self, inputs: List[Molecule], num_results: int) -> List[BackwardPredictionList]:
        # Cyclically repeat `RESULTS` until the length reaches `num_results`.
        outputs = islice(cycle(DummyModel.RESULTS), num_results)

        # Return the same outputs for each input molecule.
        return [
            BackwardPredictionList(
                input=input,
                predictions=[BackwardPrediction(input=input, output=output) for output in outputs],
            )
            for input in inputs
        ]


@pytest.mark.parametrize("measure_time", [False, True])
def test_get_results(measure_time: bool) -> None:
    def get_model_results(**kwargs):
        model_results = get_results(
            model=DummyModel(), inputs=[Molecule("C")], measure_time=measure_time, **kwargs
        )

        assert (model_results.model_timing_results is not None) == measure_time

        prediction_list = model_results.results[0]
        return [prediction.output for prediction in prediction_list.predictions]

    for num_results in [1, 2, 3, 4, 20]:
        assert get_model_results(num_results=num_results) == [
            DummyModel.RESULTS[idx] for idx in [0, 1, 3] if idx < num_results
        ]

    assert get_model_results(num_results=40, skip_repeats=False) == 10 * DummyModel.RESULTS


def test_print_and_save():
    input_mol = Molecule("c1ccccc1N")
    output_mol_bag = Bag([Molecule("c1ccccc1"), Molecule("N")])

    results = EvalResults(
        eval_args={},
        model_info={},
        num_params=0,
        num_trainable_params=0,
        num_samples=1,
        top_k=[0.0] * 50,
        mrr=0.0,
        model_time_total=ModelTimingResults(time_model_call=1.0, time_post_processing=0.1),
        predictions=[
            BackwardPredictionList(
                input=input_mol,
                predictions=[BackwardPrediction(input=input_mol, output=output_mol_bag)],
            )
        ],
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        config = EvalConfig(
            data_dir=temp_dir,
            model_class=BackwardModelClass.RetroKNN,
            results_dir=temp_dir,
        )

        print_and_save(results, config)
