"""Tests for `MolInventoryBuilder` metadata."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import rdkit
from rdkit import Chem
from rdkit.Chem import inchi

from syntheseus.search.mol_inventory import (
    IdentityStandardiser,
    InChIKeyPrefix,
    MolInventoryBuilder,
)

# "CO" is repeated and "[NH4+]" shares the key of "N", so these give 2 keys.
SMILES = ["CO", "CO", "N", "[NH4+]"]


def expected_metadata(source_sha256: Optional[str]) -> dict:
    return {
        "name": "building-blocks",
        "version": "2",
        "format_version": MolInventoryBuilder.FORMAT_VERSION,
        "key": "STEREO",
        "standardiser": {"name": "identity", "version": "1"},
        "rdkit_version": rdkit.__version__,
        "inchi_version": (inchi.GetInchiVersion() if hasattr(inchi, "GetInchiVersion") else None),
        "source_sha256": source_sha256,
        "num_keys": 2,
    }


def test_build_metadata() -> None:
    inventory = MolInventoryBuilder(IdentityStandardiser(), InChIKeyPrefix.STEREO).build(
        (Chem.MolFromSmiles(s) for s in SMILES), name="building-blocks", version="2"
    )
    assert inventory.metadata == expected_metadata(source_sha256=None)


def test_build_from_file_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source.smi"
    source.write_text("\n".join(SMILES) + "\n")

    inventory = MolInventoryBuilder(IdentityStandardiser(), InChIKeyPrefix.STEREO).build_from_file(
        source, name="building-blocks", version="2", file_format="smi"
    )
    expected_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    assert inventory.metadata == expected_metadata(source_sha256=expected_sha256)
