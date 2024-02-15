"""Tests for MolInventory objects, focusing on the provided SmilesListInventory."""

import pytest

from syntheseus.interface.molecule import Molecule
from syntheseus.search.mol_inventory import SmilesListInventory

PURCHASABLE_SMILES = ["CC", "c1ccccc1", "CCO"]
NON_PURCHASABLE_SMILES = ["C", "C1CCCCC1", "OCCO"]


@pytest.fixture
def example_inventory() -> SmilesListInventory:
    """Returns a SmilesListInventory with arbitrary molecules."""
    return SmilesListInventory(PURCHASABLE_SMILES)


def test_is_purchasable(example_inventory: SmilesListInventory) -> None:
    """
    Does the 'is_purchasable' method return true only for purchasable SMILES?
    """
    for sm in PURCHASABLE_SMILES:
        assert example_inventory.is_purchasable(Molecule(sm))

    for sm in NON_PURCHASABLE_SMILES:
        assert not example_inventory.is_purchasable(Molecule(sm))


def test_fill_metadata(example_inventory: SmilesListInventory) -> None:
    """
    Does the 'fill_metadata' method accurately fill the metadata?
    Currently it only checks that the `is_purchasable` key is filled correctly.
    At least it should add the 'is_purchasable' key.
    """

    for sm in PURCHASABLE_SMILES + NON_PURCHASABLE_SMILES:
        # Make initial molecule without any metadata
        mol = Molecule(sm)
        assert "is_purchasable" not in mol.metadata

        # Fill metadata and check that it is filled accurately.
        # To also handle the case where the metadata is filled, we run the test twice.
        for _ in range(2):
            example_inventory.fill_metadata(mol)
            assert mol.metadata["is_purchasable"] == example_inventory.is_purchasable(mol)

            # corrupt metadata so that next iteration the metadata is filled
            # and should be overwritten.
            # Type ignore is because we fill in random invalid metadata
            mol.metadata["is_purchasable"] = "abc"  # type: ignore[typeddict-item]


def test_purchasable_mols(example_inventory: SmilesListInventory) -> None:
    """
    Does the 'purchasable_mols' method work correctly? It should return a collection
    of all the purchasable molecules.
    """
    expected_set = {Molecule(sm) for sm in PURCHASABLE_SMILES}
    observed_set = set(example_inventory.purchasable_mols())
    assert expected_set == observed_set
