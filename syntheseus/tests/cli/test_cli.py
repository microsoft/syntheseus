import glob
import json
import math
import sys
import tempfile
import urllib
import zipfile
from pathlib import Path
from typing import Generator, List

import omegaconf
import pytest

from syntheseus.reaction_prediction.inference.config import BackwardModelClass
from syntheseus.reaction_prediction.utils.testing import are_single_step_models_installed

pytestmark = pytest.mark.skipif(
    not are_single_step_models_installed(),
    reason="CLI tests require all single-step models to be installed",
)


MODEL_CLASSES_TO_TEST = set(BackwardModelClass) - {BackwardModelClass.GLN}


@pytest.fixture(scope="module")
def data_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as raw_tempdir:
        tempdir = Path(raw_tempdir)

        # Download the raw USPTO-50K data released by the authors of GLN.
        uspto50k_zip_path = tempdir / "uspto50k.zip"
        urllib.request.urlretrieve(
            "https://figshare.com/ndownloader/files/45206101", uspto50k_zip_path
        )

        with zipfile.ZipFile(uspto50k_zip_path, "r") as f_zip:
            f_zip.extractall(tempdir)

        # Create a simple search targets file with a single target and a matching inventory following
        # the same example class of reactions as those used in `test_call`.

        search_targets_file_path = tempdir / "search_targets.smiles"
        with open(search_targets_file_path, "wt") as f_search_targets:
            f_search_targets.write("Cc1ccc(-c2ccc(C)cc2)cc1\n")

        inventory_file_path = tempdir / "inventory.smiles"
        with open(inventory_file_path, "wt") as f_inventory:
            for leaving_group in ["Br", "B(O)O", "I", "[Mg+]"]:
                f_inventory.write(f"Cc1ccc({leaving_group})cc1\n")

        yield tempdir


@pytest.fixture
def eval_cli_argv() -> List[str]:
    import torch

    return ["eval-single-step", f"num_gpus={int(torch.cuda.is_available())}"]


@pytest.fixture
def search_cli_argv() -> List[str]:
    import torch

    return ["search", f"use_gpu={torch.cuda.is_available()}"]


def run_cli_with_argv(argv: List[str]) -> None:
    # The import below pulls in some optional dependencies, so do it locally to avoid executing it
    # if the test suite is being skipped.
    from syntheseus.cli.main import main

    sys.argv = ["syntheseus"] + argv
    main()


def test_cli_invalid(
    data_dir: Path, tmpdir: Path, eval_cli_argv: List[str], search_cli_argv: List[str]
) -> None:
    """Test various incomplete or invalid CLI calls that should all raise an error."""
    argv_lists: List[List[str]] = [
        [],
        ["not-a-real-command"],
        eval_cli_argv + ["model_class=LocalRetro"],  # No data dir.
        eval_cli_argv + ["model_class=LocalRetro", f"data_dir={tmpdir}"],  # No data.
        eval_cli_argv + ["model_class=FakeModel", f"data_dir={data_dir}"],  # Not a real model.
        search_cli_argv
        + [
            "model_class=LocalRetro",
            f"search_targets_file={data_dir}/search_targets.smiles",
        ],  # No inventory.
        search_cli_argv
        + [
            "model_class=LocalRetro",
            f"inventory_smiles_file={data_dir}/inventory.smiles",
        ],  # No search targets.
        search_cli_argv
        + [
            "model_class=FakeModel",
            f"search_targets_file={data_dir}/search_targets.smiles",
            f"inventory_smiles_file={data_dir}/inventory.smiles",
        ],  # Not a real model.
    ]

    for argv in argv_lists:
        with pytest.raises((ValueError, omegaconf.errors.MissingMandatoryValue)):
            run_cli_with_argv(argv)


@pytest.mark.parametrize("model_class", MODEL_CLASSES_TO_TEST)
def test_cli_eval_single_step(
    model_class: BackwardModelClass, data_dir: Path, tmpdir: Path, eval_cli_argv: List[str]
) -> None:
    run_cli_with_argv(
        eval_cli_argv
        + [
            f"model_class={model_class}",
            f"data_dir={data_dir}",
            f"results_dir={tmpdir}",
            "num_top_results=5",
            "print_idxs=[1,5]",
            "num_dataset_truncation=10",
        ]
    )

    [results_path] = glob.glob(f"{tmpdir}/{model_class.name}_*.json")

    with open(results_path, "rt") as f:
        results = json.load(f)

    top_1_accuracy = results["top_k"][0]

    # We just evaluated a tiny sample of the data, so only make a rough check that the accuracy is
    # ballpark reasonable (full test set accuracy would be around ~50%).
    assert 0.2 <= top_1_accuracy <= 0.8


@pytest.mark.parametrize("model_class", MODEL_CLASSES_TO_TEST)
@pytest.mark.parametrize("search_algorithm", ["retro_star", "mcts", "pdvn"])
def test_cli_search(
    model_class: BackwardModelClass,
    search_algorithm: str,
    data_dir: Path,
    tmpdir: Path,
    search_cli_argv: List[str],
) -> None:
    run_cli_with_argv(
        search_cli_argv
        + [
            f"model_class={model_class}",
            f"search_algorithm={search_algorithm}",
            f"results_dir={tmpdir}",
            f"search_targets_file={data_dir}/search_targets.smiles",
            f"inventory_smiles_file={data_dir}/inventory.smiles",
            "limit_iterations=3",
            "num_top_results=5",
        ]
    )

    results_dir = f"{tmpdir}/{model_class.name}_*/"
    [results_path] = glob.glob(f"{results_dir}/stats.json")

    with open(results_path, "rt") as f:
        results = json.load(f)

    # Assert that a solution was found.
    assert results["soln_time_rxn_model_calls"] < math.inf
    assert len(glob.glob(f"{results_dir}/route_*.pdf")) >= 1
