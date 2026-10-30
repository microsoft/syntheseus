"""Tests for `MolInventory`."""

from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from syntheseus.interface.molecule import Molecule
from syntheseus.search.mol_inventory import (
    IdentityStandardiser,
    InChIKeyPrefix,
    InventoryLookupError,
    MolInventory,
    MolInventoryBuilder,
    Standardiser,
)

PURCHASABLE_SMILES = ["CC", "c1ccccc1", "CCO", "Oc1ccccn1", "N", "C[C@H](N)O"]
NON_PURCHASABLE_SMILES = ["C", "C1CCCCC1", "OCCO"]


class FragmentStandardiser(Standardiser):
    """Splits a molecule into its fragments."""

    name = "fragments"
    version = "1"

    def standardise(self, mol: Chem.Mol) -> list[Chem.Mol]:
        return list(Chem.GetMolFrags(mol, asMols=True))


class CountingStandardiser(IdentityStandardiser):
    def __init__(self) -> None:
        self.num_calls = 0

    def standardise(self, mol: Chem.Mol) -> list[Chem.Mol]:
        self.num_calls += 1
        return super().standardise(mol)


class FailingStandardiser(IdentityStandardiser):
    def standardise(self, mol: Chem.Mol) -> list[Chem.Mol]:
        raise RuntimeError("cannot standardise")


def build(
    smiles: list[str],
    standardiser: Standardiser = IdentityStandardiser(),
    key_prefix: InChIKeyPrefix = InChIKeyPrefix.CONNECTIVITY,
) -> MolInventory:
    return MolInventoryBuilder(standardiser, key_prefix).build(
        (Chem.MolFromSmiles(s) for s in smiles), name="test", version="1"
    )


@pytest.fixture
def inventory() -> MolInventory:
    return build(PURCHASABLE_SMILES)


def test_is_purchasable(inventory: MolInventory) -> None:
    for smiles in PURCHASABLE_SMILES:
        assert inventory.is_purchasable(Molecule(smiles))
    for smiles in NON_PURCHASABLE_SMILES:
        assert not inventory.is_purchasable(Molecule(smiles))


def test_fill_metadata(inventory: MolInventory) -> None:
    for smiles in PURCHASABLE_SMILES + NON_PURCHASABLE_SMILES:
        mol = Molecule(smiles)
        inventory.fill_metadata(mol)
        assert mol.metadata["is_purchasable"] == inventory.is_purchasable(mol)


@pytest.mark.parametrize(
    "variant",
    [
        "O=c1cccc[nH]1",  # tautomer of Oc1ccccn1
        "[NH4+]",  # protonated N
    ],
)
def test_variants_match(inventory: MolInventory, variant: str) -> None:
    assert inventory.is_purchasable(Molecule(variant))


@pytest.mark.parametrize(
    "key_prefix, expected",
    [
        (InChIKeyPrefix.CONNECTIVITY, {"C[C@H](N)O": True, "C[C@@H](N)O": True, "CC(N)O": True}),
        (InChIKeyPrefix.STEREO, {"C[C@H](N)O": True, "C[C@@H](N)O": False, "CC(N)O": False}),
    ],
)
def test_key_prefix_stereo(key_prefix: InChIKeyPrefix, expected: dict[str, bool]) -> None:
    inventory = build(["C[C@H](N)O"], key_prefix=key_prefix)
    assert {s: inventory.is_purchasable(Molecule(s)) for s in expected} == expected


def test_multi_structure_lookup() -> None:
    inventory = build(["CO", "CS"], standardiser=FragmentStandardiser())
    assert inventory.is_purchasable(Molecule("CO.CS"))
    assert not inventory.is_purchasable(Molecule("CO.CC"))


def test_standardiser_error_raises_lookup_error(inventory: MolInventory) -> None:
    inventory = MolInventory(inventory.keys, inventory.metadata, FailingStandardiser())
    with pytest.raises(InventoryLookupError):
        inventory.is_purchasable(Molecule("CO"))


def test_empty_inchi_raises_lookup_error(inventory: MolInventory) -> None:
    with pytest.raises(InventoryLookupError):
        inventory.is_purchasable(Molecule("*C"))


def test_lookups_are_cached() -> None:
    standardiser = CountingStandardiser()
    inventory = build(["CO"], standardiser=standardiser)
    num_build_calls = standardiser.num_calls

    for _ in range(3):
        assert inventory.is_purchasable(Molecule("CO"))
    assert standardiser.num_calls == num_build_calls + 1


def test_save_and_load_round_trip(inventory: MolInventory, tmp_path: Path) -> None:
    path = tmp_path / "inventory.molinv"
    builder = MolInventoryBuilder(IdentityStandardiser())
    builder.save(inventory, path)
    loaded = builder.load(path)

    assert loaded.metadata == inventory.metadata
    np.testing.assert_array_equal(loaded.keys, inventory.keys)
    for smiles in PURCHASABLE_SMILES + NON_PURCHASABLE_SMILES:
        mol = Molecule(smiles)
        assert loaded.is_purchasable(mol) == inventory.is_purchasable(mol)
