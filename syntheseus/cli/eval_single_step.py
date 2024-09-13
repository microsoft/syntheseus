"""Script for evaluating a model on a given dataset.

Each of the model types can be loaded from a *single directory*, possibly containing several files
(e.g. checkpoint, config, etc). See individual model wrappers for the model directory formats.

Example invocation:
    python ./cli/eval_single_step.py \
        data_dir=[DATA_DIR] \
        model_class=RetroKNN \
        model_dir=[MODEL_DIR]
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, fields
from functools import partial
from itertools import islice
from statistics import mean, median
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Sequence, Tuple, cast

from more_itertools import batched
from omegaconf import MISSING, OmegaConf
from tqdm import tqdm

from syntheseus.interface.models import (
    ForwardReactionModel,
    InputType,
    ReactionModel,
    ReactionType,
)
from syntheseus.interface.reaction import Reaction, SingleProductReaction
from syntheseus.reaction_prediction.chem.utils import remove_stereo_information_from_reaction
from syntheseus.reaction_prediction.data.dataset import (
    DataFold,
    DiskReactionDataset,
    ReactionDataset,
)
from syntheseus.reaction_prediction.data.reaction_sample import ReactionSample
from syntheseus.reaction_prediction.inference.config import (
    BackwardModelClass,
    BackwardModelConfig,
    ForwardModelConfig,
)
from syntheseus.reaction_prediction.utils.config import (
    get_config as cli_get_config,
)
from syntheseus.reaction_prediction.utils.config import (
    get_error_message_for_missing_value,
)
from syntheseus.reaction_prediction.utils.metrics import (
    ModelTimingResults,
    TopKMetricsAccumulator,
    compute_total_time,
)
from syntheseus.reaction_prediction.utils.misc import asdict_extended, set_random_seed
from syntheseus.reaction_prediction.utils.model_loading import get_model

logger = logging.getLogger(__file__)


_RXN_WTIH_IDENTIFIER_ERROR = (
    "Reactions with the `identifier` field currently not supported in evaluation"
)


@dataclass
class BaseEvalConfig:
    data_dir: str = MISSING  # Directory containing preprocessed data
    num_top_results: int = 100  # Number of results to request from the model
    fold: DataFold = DataFold.TEST  # Dataset fold to evaluate on
    batch_size: int = 16  # Batch size to use
    skip_repeats: bool = True  # Whether repeated results should be skipped
    num_gpus: int = 1  # Number of GPUs to use (or 0 if running on CPU)
    num_dataset_truncation: Optional[int] = None  # Subset size to evaluate on
    include_results_modulo_stereo: bool = False  # Whether to include stereo-agnostic results

    # Fields for saving and printing results
    print_idxs: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20, 50])
    save_outputs: bool = True  # Whether to save the results as a JSON file
    include_predictions: bool = True  # Whether to include the full predictions lists
    results_dir: str = field(
        default_factory=lambda: os.path.join(os.path.dirname(__file__), "results")
    )
    filestring: Optional[str] = None  # Unique string (appended to filename) to identify the run

    # Fields relevant to back translation
    back_translation_config: ForwardModelConfig = field(
        default_factory=lambda: ForwardModelConfig(model_kwargs={"is_forward": True})
    )
    back_translation_num_results: int = 1


@dataclass
class EvalConfig(BackwardModelConfig, BaseEvalConfig):
    """Config for running evaluation on a given dataset."""

    pass


@dataclass(frozen=True)
class ModelResults(Generic[ReactionType]):
    results: List[Sequence[ReactionType]]
    model_timing_results: Optional[ModelTimingResults] = None


@dataclass(frozen=True)
class EvalResults:
    eval_args: Dict[str, Any]
    model_info: Dict[str, Any]
    num_params: int
    num_trainable_params: int
    num_samples: int
    top_k: List[float]
    mrr: float
    min_num_predictions: int
    max_num_predictions: int
    mean_num_predictions: float
    median_num_predictions: float
    model_time_total: ModelTimingResults
    match_modulo_stereo_top_k: Optional[List[float]] = None
    match_modulo_stereo_mrr: Optional[float] = None
    back_translation_top_k: Optional[List[float]] = None
    back_translation_mrr: Optional[float] = None
    predictions: Optional[List[Sequence[Reaction]]] = None
    back_translation_predictions: Optional[List[List[Sequence[Reaction]]]] = None
    back_translation_time_total: Optional[ModelTimingResults] = None


def get_results(
    model: ReactionModel[InputType, ReactionType],
    inputs: List[InputType],
    num_results: int,
    measure_time: bool = False,
) -> ModelResults[ReactionType]:
    """Given a batch of inputs to the reaction model, return a batch of (possibly filtered) results.

    Args:
        model: Reaction model to use.
        inputs: Batch of inputs to the reaction model, each either a molecule or a set of molecules,
            depending on directionality.
        num_results: Number of results we want to try to obtain for each input.
        measure_time: Whether to measure time taken by the different parts of the code.

    Returns:
        A dataclass containing the model outputs and (optionally) time measurements.
    """

    # Handle case of empty inputs without calling the model.
    if not inputs:
        return ModelResults(
            results=[],
            model_timing_results=(
                ModelTimingResults(time_model_call=0, time_post_processing=0)
                if measure_time
                else None
            ),
        )

    timing_results: Dict[str, float] = {}

    if measure_time:
        time_model_call_start = time.time()

    raw_batch_outputs = model(inputs, num_results=num_results)

    if measure_time:
        time_model_call_end = time.time()
        timing_results["time_model_call"] = time_model_call_end - time_model_call_start

    batch_outputs: List[Sequence[ReactionType]] = []

    for outputs in raw_batch_outputs:
        if len(outputs) > num_results:
            raise ValueError(f"Requested {num_results} results, but model produced {len(outputs)}")

        # Check identifiers are not set
        if any(reaction.identifier is not None for reaction in outputs):
            raise NotImplementedError(_RXN_WTIH_IDENTIFIER_ERROR)

        # Check for length
        if len(outputs) < num_results:
            logger.debug(f"Tried to get {num_results} results, but only got {len(outputs)}")

        batch_outputs.append(list(outputs))

    if measure_time:
        timing_results["time_post_processing"] = time.time() - time_model_call_end

    return ModelResults(
        results=batch_outputs,
        model_timing_results=ModelTimingResults(**timing_results) if measure_time else None,
    )


def compute_metrics(
    model: ReactionModel[InputType, ReactionType],
    dataset: ReactionDataset,
    num_dataset_truncation: Optional[int],
    num_top_results: int,
    include_results_modulo_stereo: bool = False,
    back_translation_model: Optional[ForwardReactionModel] = None,
    back_translation_num_results: int = 1,
    fold: DataFold = DataFold.VALIDATION,
    batch_size: int = 16,
    include_predictions: bool = False,
) -> EvalResults:
    """Compute top-k accuracies and Mean Reciprocal Rank of a model on a given dataset."""

    # Gather all evaluation args for the record.
    eval_args = {
        "model_class": model.__class__.__name__,
        "num_top_results": num_top_results,
        "fold": fold.name,
        "batch_size": batch_size,
    }

    ground_truth_match_metrics = TopKMetricsAccumulator(max_num_results=num_top_results)

    if include_results_modulo_stereo:
        ground_truth_match_modulo_stereo_metrics = TopKMetricsAccumulator(
            max_num_results=num_top_results
        )

    if back_translation_model is not None:
        eval_args.update(
            back_translation_model_class=back_translation_model.__class__.__name__,
            back_translation_num_results=back_translation_num_results,
        )

        if model.is_forward():
            raise ValueError("Back translation only supported when evaluating a backward model")

        if not back_translation_model.is_forward():
            raise ValueError("Back translation model should be a forward model")

        back_translation_metrics = TopKMetricsAccumulator(max_num_results=num_top_results)

    print(f"Testing model {model.__class__.__name__} with args {eval_args}")

    all_predictions: List[Sequence[Reaction]] = []
    all_back_translation_predictions: List[List[Sequence[Reaction]]] = []

    model_timing_results: List[ModelTimingResults] = []
    back_translation_timing_results: List[ModelTimingResults] = []

    test_dataset: Iterable[ReactionSample] = dataset[fold]
    test_dataset_size = dataset.get_num_samples(fold)

    num_predictions: List[int] = []

    if num_dataset_truncation is not None and num_dataset_truncation < test_dataset_size:
        print(
            f"Truncating the dataset from {test_dataset_size} to {num_dataset_truncation} samples"
        )

        test_dataset = islice(test_dataset, num_dataset_truncation)
        test_dataset_size = num_dataset_truncation

    for batch in tqdm(
        batched(test_dataset, batch_size),
        total=math.ceil(test_dataset_size / batch_size),
    ):
        # Tell mypy what the type of `batch` is as `batched` seems to lose it.
        batch = cast(Tuple[ReactionSample], batch)

        inputs: List[InputType] = []
        outputs: List[ReactionType] = []

        for sample in batch:
            # We need to cast when appending to `inputs`/`outputs` because link between types and
            # reaction models is unclear to mypy.

            output: Reaction
            if model.is_forward():
                inputs.append(sample.reactants)
                output = Reaction(reactants=sample.reactants, products=sample.products)
            else:
                assert (
                    len(sample.products) == 1
                ), f"Model expected a single target product, got {len(sample.products)}"
                [single_product] = sample.products

                inputs.append(cast(InputType, single_product))
                output = SingleProductReaction(reactants=sample.reactants, product=single_product)

            outputs.append(cast(ReactionType, output))
            del output

        results_with_timing = get_results(
            model,
            inputs=inputs,
            num_results=num_top_results,
            measure_time=True,
        )

        assert results_with_timing.model_timing_results is not None
        model_timing_results.append(results_with_timing.model_timing_results)

        batch_predictions: List[Sequence[ReactionType]] = results_with_timing.results

        batch_back_translation_predictions: List[List[Sequence[Reaction]]] = []
        for input, output, reaction_list in zip(inputs, outputs, batch_predictions):
            num_predictions.append(len(reaction_list))

            # Check that the identifier field is not used in the evaluation,
            # otherwise `==` below will not work as intended.
            if output.identifier is not None:
                raise NotImplementedError(_RXN_WTIH_IDENTIFIER_ERROR)
            ground_truth_matches = [rxn == output for rxn in reaction_list]
            ground_truth_match_metrics.add(ground_truth_matches)

            for rxn, ground_truth_match in zip(reaction_list, ground_truth_matches):
                rxn.metadata["ground_truth_match"] = ground_truth_match

            if include_results_modulo_stereo:
                output_without_stereo = remove_stereo_information_from_reaction(output)
                ground_truth_match_modulo_stereo_metrics.add(
                    [
                        remove_stereo_information_from_reaction(rxn) == output_without_stereo
                        for rxn in reaction_list
                    ]
                )

            if back_translation_model is not None:
                assert back_translation_metrics is not None

                back_translation_results_with_timing = get_results(
                    back_translation_model,
                    [rxn.reactants for rxn in reaction_list],
                    num_results=back_translation_num_results,
                    measure_time=True,
                )

                assert back_translation_results_with_timing.model_timing_results is not None
                back_translation_timing_results.append(
                    back_translation_results_with_timing.model_timing_results
                )

                back_translation_results = back_translation_results_with_timing.results

                if include_predictions:
                    batch_back_translation_predictions.append(back_translation_results)

                # Back translation is successful if any of the `back_translation_num_results` bags
                # of products returned by the forward model contains the input.
                back_translation_matches = [
                    any(input in rxn.products for rxn in result)
                    for result in back_translation_results
                ]

                back_translation_metrics.add(back_translation_matches)

        if include_predictions:
            all_predictions.extend(batch_predictions)
            all_back_translation_predictions.extend(batch_back_translation_predictions)

    extra_args: Dict[str, Any] = {}

    if include_predictions:
        extra_args.update(predictions=all_predictions)
        extra_args.update(back_translation_predictions=all_back_translation_predictions)

    if include_results_modulo_stereo:
        extra_args.update(
            match_modulo_stereo_top_k=ground_truth_match_modulo_stereo_metrics.top_k,
            match_modulo_stereo_mrr=ground_truth_match_modulo_stereo_metrics.mrr,
        )

    if back_translation_model is not None:
        extra_args.update(
            back_translation_top_k=back_translation_metrics.top_k,
            back_translation_mrr=back_translation_metrics.mrr,
            back_translation_time_total=compute_total_time(back_translation_timing_results),
        )

    # Add the number of parameters/trainable parameters
    num_params = sum(p.numel() for p in model.get_parameters())
    num_trainable_params = sum(p.numel() for p in model.get_parameters() if p.requires_grad)

    return EvalResults(
        eval_args=eval_args,
        model_info=model.get_model_info(),
        num_params=num_params,
        num_trainable_params=num_trainable_params,
        num_samples=ground_truth_match_metrics.num_samples,
        top_k=ground_truth_match_metrics.top_k,
        mrr=ground_truth_match_metrics.mrr,
        min_num_predictions=min(num_predictions),
        max_num_predictions=max(num_predictions),
        mean_num_predictions=float(mean(num_predictions)),
        median_num_predictions=float(median(num_predictions)),
        model_time_total=compute_total_time(model_timing_results),
        **extra_args,
    )


def compute_metrics_from_config(
    model: ReactionModel[InputType, ReactionType],
    dataset: ReactionDataset,
    back_translation_model: Optional[ForwardReactionModel],
    config: BaseEvalConfig,
) -> EvalResults:
    """Variant of `compute_metrics` that uses an eval config instead of explicit arguments."""

    return compute_metrics(
        model,
        dataset=dataset,
        num_dataset_truncation=config.num_dataset_truncation,
        num_top_results=config.num_top_results,
        include_results_modulo_stereo=config.include_results_modulo_stereo,
        back_translation_model=back_translation_model,
        back_translation_num_results=config.back_translation_num_results,
        fold=config.fold,
        batch_size=config.batch_size,
        include_predictions=config.include_predictions,
    )


def print_and_save(results: EvalResults, config: EvalConfig, suffix: str = "") -> None:
    chosen_top_k_results: Dict[str, Dict[int, List[float]]] = {}

    # Print the results in a concise form.
    for f in fields(results):
        if f.name not in ("model_info", "predictions", "back_translation_predictions"):
            if f.name.endswith("top_k"):
                top_k = getattr(results, f.name)
                if top_k is not None:
                    # Extract the top-k for the chosen values of `k`; other values are not printed.
                    chosen_top_k_results[f.name] = {x: top_k[x - 1] for x in config.print_idxs}
            else:
                print(f"{f.name}: {getattr(results, f.name)}")

    for key, top_k in chosen_top_k_results.items():
        if suffix:
            print(f"{key} results {suffix}:")
        else:
            print(f"{key} results:")

        for k, result in top_k.items():
            print(f"{k}: {result}", flush=True)

    # Save the results into a json file, generate its name from the timestamp for uniqueness.
    if config.save_outputs:
        filestr = ("_" + config.filestring) if config.filestring else ""
        suffix = ("_" + suffix) if suffix else ""
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")

        os.makedirs(config.results_dir, exist_ok=True)
        outfile = os.path.join(
            config.results_dir,
            f"{config.model_class.name}_{str(timestamp)}{filestr}{suffix}.json",
        )
        results_dict = asdict_extended(results)

        for key, top_k in chosen_top_k_results.items():
            results_dict[f"chosen_{key}"] = top_k

        with open(outfile, "w") as outfile_stream:
            outfile_stream.write(json.dumps(results_dict))


def run_from_config(
    config: EvalConfig,
    extra_steps: List[Callable[[ReactionModel, ReactionDataset, Optional[ReactionModel]], None]],
) -> None:
    set_random_seed(0)

    print("Running eval with the following config:")
    print(config)

    if OmegaConf.is_missing(config, "data_dir"):
        raise ValueError("data_dir should be set to a directory containing a reaction dataset")

    if OmegaConf.is_missing(config, "model_class"):
        raise ValueError(
            get_error_message_for_missing_value("model_class", [c.name for c in BackwardModelClass])
        )

    get_model_fn = partial(get_model, batch_size=config.batch_size, num_gpus=config.num_gpus)
    model = get_model_fn(config, remove_duplicates=config.skip_repeats)

    if OmegaConf.is_missing(config.back_translation_config, "model_class"):
        back_translation_model = None
    else:
        # Back translation model must be a forward model (checked in `compute_metrics`).
        back_translation_model = cast(
            ForwardReactionModel, get_model_fn(config.back_translation_config)
        )

    dataset = DiskReactionDataset(config.data_dir, sample_cls=ReactionSample)
    results = compute_metrics_from_config(
        model=model, dataset=dataset, back_translation_model=back_translation_model, config=config
    )
    print_and_save(results, config)

    for extra_step in extra_steps:
        extra_step(model, dataset, back_translation_model)


def main(argv: Optional[List[str]] = None) -> None:
    config: EvalConfig = cli_get_config(argv=argv, config_cls=EvalConfig)
    run_from_config(config, extra_steps=[])


if __name__ == "__main__":
    main()
