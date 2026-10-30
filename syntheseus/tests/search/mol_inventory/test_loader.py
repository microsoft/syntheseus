"""Tests for `MolDataLoader`."""

from pathlib import Path

import pytest
from rdkit import Chem

from syntheseus.search.mol_inventory import MolDataLoader


def test_reads_smi_ignoring_ids(tmp_path: Path) -> None:
    path = tmp_path / "source.smi"
    path.write_text("CO mol-1\nc1ccccc1\n")
    assert [Chem.MolToSmiles(m) for m in MolDataLoader(path, "smi")] == ["CO", "c1ccccc1"]


def test_unparsable_smiles_raises(tmp_path: Path) -> None:
    path = tmp_path / "source.smi"
    path.write_text("CO\nnot-a-smiles\n")
    with pytest.raises(ValueError):
        list(MolDataLoader(path, "smi"))


def test_unsupported_format_raises(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        MolDataLoader(tmp_path / "source.sdf", "sdf")
