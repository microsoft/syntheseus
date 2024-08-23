from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import pytest
from omegaconf import OmegaConf

from syntheseus import BackwardReactionModel, Bag, Molecule, SingleProductReaction
from syntheseus.cli.search import SearchConfig, run_from_config
from syntheseus.reaction_prediction.inference.config import BackwardModelClass


class FlakyReactionModel(BackwardReactionModel):
    """Dummy reaction model that only works when called for the first time."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._used = False

    def _get_reactions(
        self, inputs: list[Molecule], num_results: int
    ) -> list[Sequence[SingleProductReaction]]:
        if self._used:
            raise RuntimeError()

        self._used = True
        return [
            [
                SingleProductReaction(
                    reactants=Bag([Molecule("C")]), product=product, metadata={"probability": 1.0}
                )
            ]
            for product in inputs
        ]


def test_resume_search(tmpdir: Path) -> None:
    search_targets_file_path = tmpdir / "search_targets.smiles"
    with open(search_targets_file_path, "wt") as f_search_targets:
        f_search_targets.write("CC\nCC\nCC\nCC\n")

    inventory_file_path = tmpdir / "inventory.smiles"
    with open(inventory_file_path, "wt") as f_inventory:
        f_inventory.write("C\n")

    # Inject our flaky reaction model into the set of supported model classes.
    BackwardModelClass._member_map_["FlakyReactionModel"] = SimpleNamespace(  # type: ignore
        name="FlakyReactionModel", value=FlakyReactionModel
    )

    config = OmegaConf.create(  # type: ignore
        SearchConfig(
            model_class="FlakyReactionModel",  # type: ignore[arg-type]
            search_algorithm="retro_star",
            search_targets_file=str(search_targets_file_path),
            inventory_smiles_file=str(inventory_file_path),
            results_dir=str(tmpdir),
            append_timestamp_to_dir=False,
            limit_iterations=1,
            num_routes_to_plot=0,
        )
    )

    results_dir = tmpdir / "FlakyReactionModel"

    def file_exist(idx: int, name: str) -> bool:
        return (results_dir / str(idx) / name).exists()

    # Try to run search three times; each time we will succeed solving one target (which requires one
    # call) and then fail on the next one.
    for trial_idx in range(3):
        with pytest.raises(RuntimeError):
            run_from_config(config)

        for idx in range(trial_idx + 1):
            assert file_exist(idx, "stats.json")
            assert not file_exist(idx, ".lock")

        assert not file_exist(trial_idx + 1, "stats.json")
        assert file_exist(trial_idx + 1, ".lock")

    run_from_config(config)

    # The last search needs to solve one final target so it will succeed.
    for idx in range(4):
        assert file_exist(idx, "stats.json")
        assert not file_exist(idx, ".lock")

    with open(results_dir / "stats.json", "rt") as f_stats:
        stats = json.load(f_stats)

    # Even though each search only solved a single target, final stats should include everything.
    assert stats["num_targets"] == stats["num_solved_targets"] == 4

    # Finally change the targets and verify that the discrepancy will be detected.
    with open(search_targets_file_path, "wt") as f_search_targets:
        f_search_targets.write("CC\nCCCC\nCC\nCC\n")

    with pytest.raises(RuntimeError):
        run_from_config(config)
