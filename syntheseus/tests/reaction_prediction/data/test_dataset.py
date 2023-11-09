import pytest
from pathlib import Path

from syntheseus.reaction_prediction.data.dataset import DataFold, DiskReactionDataset
from syntheseus.reaction_prediction.data.reaction_sample import ReactionSample


@pytest.mark.parametrize("mapped", [False, True])
def test_save_and_load(tmp_path: Path, mapped: bool) -> None:
    samples = [
        ReactionSample.from_reaction_smiles_strict(reaction_smiles, mapped=mapped)
        for reaction_smiles in [
            "O[c:1]1[cH:2][c:3](=[O:4])[nH:5][cH:6][cH:7]1>>[cH:1]1[cH:2][c:3](=[O:4])[nH:5][cH:6][cH:7]1",
            "CC(C)(C)OC(=O)[N:1]1[CH2:2][CH2:3][C@H:4]([F:5])[CH2:6]1>>[NH:1]1[CH2:2][CH2:3][C@H:4]([F:5])[CH2:6]1",
        ]
    ]

    for fold in DataFold:
        DiskReactionDataset.save_samples_to_file(data_dir=tmp_path, fold=fold, samples=samples)

    # Now try to load the data we just saved.
    dataset = DiskReactionDataset(tmp_path, sample_cls=ReactionSample)

    for fold in DataFold:
        assert list(dataset[fold]) == samples
