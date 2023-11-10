import csv
from pathlib import Path

import pytest

from syntheseus.interface.bag import Bag
from syntheseus.interface.molecule import Molecule
from syntheseus.reaction_prediction.data.dataset import (
    CSV_REACTION_SMILES_COLUMN_NAME,
    DataFold,
    DataFormat,
    DiskReactionDataset,
)
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

    for load_format in [None, DataFormat.JSONL]:
        # Now try to load the data we just saved.
        dataset = DiskReactionDataset(tmp_path, sample_cls=ReactionSample, data_format=load_format)

        for fold in DataFold:
            assert list(dataset[fold]) == samples


@pytest.mark.parametrize("format", [DataFormat.CSV, DataFormat.SMILES])
def test_load_external_format(tmp_path: Path, format: DataFormat) -> None:
    # Example reaction SMILES, purposefully using non-canonical forms of reactants and product.
    reaction_smiles = (
        "[cH:1]1[cH:2][c:3]([CH3:4])[cH:5][cH:6][c:7]1Br.B(O)(O)[c:8]1[cH:9][cH:10][c:11]([CH3:12])[cH:13][cH:14]1>>"
        "[cH:1]1[cH:2][c:3]([CH3:4])[cH:5][cH:6][c:7]1[c:8]2[cH:14][cH:13][c:11]([CH3:12])[cH:10][cH:9]2"
    )

    filename = DiskReactionDataset.get_filename_suffix(format=format, fold=DataFold.TRAIN)
    with open(tmp_path / filename, "wt") as f:
        if format == DataFormat.CSV:
            writer = csv.DictWriter(f, fieldnames=["id", "class", CSV_REACTION_SMILES_COLUMN_NAME])
            writer.writeheader()
            writer.writerow(
                {"id": 0, "class": "UNK", CSV_REACTION_SMILES_COLUMN_NAME: reaction_smiles}
            )
        else:
            f.write(f"{reaction_smiles}\n")

    for load_format in [None, format]:
        dataset = DiskReactionDataset(tmp_path, sample_cls=ReactionSample, data_format=load_format)
        assert dataset.get_num_samples(DataFold.TRAIN) == 1

        samples = list(dataset[DataFold.TRAIN])
        assert len(samples) == 1

        [sample] = samples

        # After loading, the reactants and products should be in canonical SMILES form, with the
        # atom mapping removed.
        assert sample == ReactionSample(
            reactants=Bag([Molecule("Cc1ccc(Br)cc1"), Molecule("Cc1ccc(B(O)O)cc1")]),
            products=Bag([Molecule("Cc1ccc(-c2ccc(C)cc2)cc1")]),
        )

        # The original reaction SMILES should have been saved separately.
        assert sample.mapped_reaction_smiles == reaction_smiles


@pytest.mark.parametrize("format", [DataFormat.JSONL, DataFormat.CSV, DataFormat.SMILES])
def test_format_detection(tmp_path: Path, format: DataFormat) -> None:
    other_format = (set(DataFormat) - {format}).pop()

    # Create two files with different extensions, so that it is ambiguous which format we want.
    (tmp_path / DiskReactionDataset.get_filename_suffix(format, DataFold.TRAIN)).touch()
    (tmp_path / DiskReactionDataset.get_filename_suffix(other_format, DataFold.TEST)).touch()

    # Loading with automatic resolution should fail.
    with pytest.raises(ValueError):
        DiskReactionDataset(data_dir=tmp_path, sample_cls=ReactionSample)

    # Loading with an explicit format should succeed.
    for f in [format, other_format]:
        DiskReactionDataset(data_dir=tmp_path, sample_cls=ReactionSample, data_format=f)

    # Loading with an explicit format but no matching files should fail.
    another_format = (set(DataFormat) - {format, other_format}).pop()
    with pytest.raises(ValueError):
        DiskReactionDataset(
            data_dir=tmp_path, sample_cls=ReactionSample, data_format=another_format
        )

    # Create another file with the right suffix.
    (tmp_path / f"raw_{DiskReactionDataset.get_filename_suffix(format, DataFold.TRAIN)}").touch()

    # Loading with an explicit format should now fail due to ambiguity.
    with pytest.raises(ValueError):
        DiskReactionDataset(data_dir=tmp_path, sample_cls=ReactionSample, data_format=format)
