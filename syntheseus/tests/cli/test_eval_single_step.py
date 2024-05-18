import math
from itertools import cycle, islice
from pathlib import Path
from typing import Iterable, List, Sequence

import pytest

from syntheseus.cli.eval_single_step import (
    EvalConfig,
    EvalResults,
    compute_metrics,
    get_results,
    print_and_save,
)
from syntheseus.interface.bag import Bag
from syntheseus.interface.models import InputType, ReactionModel, ReactionType
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import Reaction, SingleProductReaction
from syntheseus.reaction_prediction.data.dataset import DataFold, DiskReactionDataset
from syntheseus.reaction_prediction.data.reaction_sample import ReactionSample
from syntheseus.reaction_prediction.inference.config import BackwardModelClass
from syntheseus.reaction_prediction.utils.metrics import ModelTimingResults


class DummyModel(ReactionModel):
    def __init__(
        self, device: str = "cpu", is_forward: bool = False, repeat: bool = False, **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.device = device
        self._is_forward = is_forward
        self._repeat = repeat

    RESULTS = [
        Bag([Molecule("C"), Molecule("N")]),
        Bag([Molecule("c1ccccc1")]),
        Bag([Molecule("N"), Molecule("C")]),
        Bag([Molecule("NC=O")]),
    ]

    def is_forward(self) -> bool:
        return self._is_forward

    def get_parameters(self):
        return []

    def _get_reactions(
        self, inputs: List[InputType], num_results: int
    ) -> List[Sequence[ReactionType]]:
        outputs: Iterable[Bag[Molecule]] = []

        if self._repeat:
            # Cyclically repeat `RESULTS` until the length reaches `num_results`.
            outputs = islice(cycle(DummyModel.RESULTS), num_results)
        else:
            outputs = DummyModel.RESULTS[:num_results]

        # Return the same outputs for each input molecule.
        results: List[Sequence] = []
        for input in inputs:
            if self._is_forward:
                assert isinstance(input, Bag)
                results.append([Reaction(reactants=input, products=output) for output in outputs])
            else:
                assert isinstance(input, Molecule)
                results.append(
                    [SingleProductReaction(product=input, reactants=output) for output in outputs]
                )

        return results


@pytest.mark.parametrize("repeat", [False, True])
@pytest.mark.parametrize("measure_time", [False, True])
def test_get_results(repeat: bool, measure_time: bool) -> None:
    def get_model_results(remove_duplicates: bool = True, **kwargs):
        model_results = get_results(
            model=DummyModel(is_forward=False, repeat=repeat, remove_duplicates=remove_duplicates),
            inputs=[Molecule("C")],
            measure_time=measure_time,
            **kwargs,
        )

        assert (model_results.model_timing_results is not None) == measure_time

        return [rxn.reactants for rxn in model_results.results[0]]

    for num_results in [1, 2, 3, 4, 20]:
        assert get_model_results(num_results=num_results) == [
            DummyModel.RESULTS[idx] for idx in [0, 1, 3] if idx < num_results
        ]

    results_with_repeats = get_model_results(num_results=40, remove_duplicates=False)

    if repeat:
        # If single-step model repeats indefinitely, then we get as many results as we asked for...
        assert results_with_repeats == 10 * DummyModel.RESULTS
    else:
        # ...otherwise we get fewer.
        assert results_with_repeats == DummyModel.RESULTS


@pytest.mark.parametrize("forward", [False, True])
@pytest.mark.parametrize("num_dataset_truncation", [2, 3, 4])
@pytest.mark.parametrize("num_top_results", [2, 3, 4, 5])
def test_compute_metrics(
    forward: bool, num_dataset_truncation: int, num_top_results: int, tmp_path: Path
) -> None:
    """Test the `compute_metrics` function.

    By extension this also tests `get_results` (and thus calls the model). As `compute_metrics` has
    a slightly different behaviour depending on whether the model is forward or backward, this test
    checks both cases.
    """
    samples = []
    for input, output in [("C.N", "CN"), ("NC=O", "CN"), ("C.C", "CC")]:
        if forward:
            input, output = output, input
        samples.append(
            ReactionSample.from_reaction_smiles_strict(f"{input}>>{output}", mapped=False)
        )

    for fold in DataFold:
        DiskReactionDataset.save_samples_to_file(data_dir=tmp_path, fold=fold, samples=samples)

    eval_results = compute_metrics(
        model=DummyModel(is_forward=forward, repeat=False),
        dataset=DiskReactionDataset(tmp_path, sample_cls=ReactionSample),
        num_dataset_truncation=num_dataset_truncation,
        num_top_results=num_top_results,
    )

    # Compute by hand what the resulting accuracy should be.
    num_samples = min(num_dataset_truncation, 3)
    is_correct_top_1 = [True, False, False][:num_samples]
    is_correct_top_10 = [True, num_top_results >= 4, False][:num_samples]

    expected_accuracy_top_1 = sum(is_correct_top_1) / num_samples
    expected_accuracy_top_max = sum(is_correct_top_10) / num_samples

    assert math.isclose(eval_results.top_k[0], expected_accuracy_top_1)
    assert math.isclose(eval_results.top_k[-1], expected_accuracy_top_max)


def test_print_and_save(tmp_path: Path) -> None:
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
        predictions=[[SingleProductReaction(product=input_mol, reactants=output_mol_bag)]],
    )

    config = EvalConfig(
        data_dir=str(tmp_path),
        model_class=BackwardModelClass.RetroKNN,  # Model choice is arbitrary here.
        results_dir=str(tmp_path),
    )

    print_and_save(results, config)
