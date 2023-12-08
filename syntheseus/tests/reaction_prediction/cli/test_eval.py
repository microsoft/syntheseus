import tempfile
from itertools import cycle, islice
from typing import Iterable, List

import pytest

from syntheseus.cli.eval_single_step import (
    EvalConfig,
    EvalResults,
    get_results,
    print_and_save,
)
from syntheseus.interface.bag import Bag
from syntheseus.interface.models import (
    BackwardPrediction,
    BackwardPredictionList,
    BackwardReactionModel,
)
from syntheseus.interface.molecule import Molecule
from syntheseus.reaction_prediction.inference.config import BackwardModelClass
from syntheseus.reaction_prediction.utils.metrics import ModelTimingResults


class DummyModel(BackwardReactionModel):
    def __init__(self, repeat: bool) -> None:
        self._repeat = repeat

    RESULTS = [
        Bag([Molecule("C"), Molecule("N")]),
        Bag([Molecule("c1ccccc1")]),
        Bag([Molecule("N"), Molecule("C")]),
        Bag([Molecule("NC=O")]),
    ]

    def __call__(self, inputs: List[Molecule], num_results: int) -> List[BackwardPredictionList]:
        outputs: Iterable[Bag[Molecule]] = []

        if self._repeat:
            # Cyclically repeat `RESULTS` until the length reaches `num_results`.
            outputs = islice(cycle(DummyModel.RESULTS), num_results)
        else:
            outputs = DummyModel.RESULTS[:num_results]

        # Return the same outputs for each input molecule.
        return [
            BackwardPredictionList(
                input=input,
                predictions=[BackwardPrediction(input=input, output=output) for output in outputs],
            )
            for input in inputs
        ]


@pytest.mark.parametrize("repeat", [False, True])
@pytest.mark.parametrize("measure_time", [False, True])
def test_get_results(repeat: bool, measure_time: bool) -> None:
    def get_model_results(**kwargs):
        model_results = get_results(
            model=DummyModel(repeat), inputs=[Molecule("C")], measure_time=measure_time, **kwargs
        )

        assert (model_results.model_timing_results is not None) == measure_time

        prediction_list = model_results.results[0]
        return [prediction.output for prediction in prediction_list.predictions]

    for num_results in [1, 2, 3, 4, 20]:
        assert get_model_results(num_results=num_results) == [
            DummyModel.RESULTS[idx] for idx in [0, 1, 3] if idx < num_results
        ]

    results_with_repeats = get_model_results(num_results=40, skip_repeats=False)

    if repeat:
        # If single-step model repeats indefinitely, then we get as many results as we asked for...
        assert results_with_repeats == 10 * DummyModel.RESULTS
    else:
        # ...otherwise we get fewer.
        assert results_with_repeats == DummyModel.RESULTS


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
        min_num_predictions=1,
        max_num_predictions=50,
        mean_num_predictions=25.0,
        median_num_predictions=10.0,
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
