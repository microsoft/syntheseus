from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from typing import Any, Optional

import pytest

from syntheseus.interface.molecule import Molecule


def test_positive_equality(cocs_mol: Molecule) -> None:
    """Various tests that 2 molecules which should be equal are actually equal."""

    mol_copy = Molecule(smiles=str(cocs_mol.smiles))

    # Test 1: original and copy should be equal
    assert cocs_mol == mol_copy

    # Test 2: although equal, they should be distinct objects
    assert cocs_mol is not mol_copy

    # Test 3: differences in metadata should not affect equality
    # (type ignores are because we are adding arbitrary unrealistic metadata)
    cocs_mol.metadata["test"] = "str1"  # type: ignore[typeddict-unknown-key]
    mol_copy.metadata["test"] = "str2"  # type: ignore[typeddict-unknown-key]
    mol_copy.metadata["other_field"] = "not in other mol"  # type: ignore[typeddict-unknown-key]
    assert cocs_mol == mol_copy
    assert cocs_mol.metadata != mol_copy.metadata


def test_negative_equality(cocs_mol: Molecule) -> None:
    """Various tests that molecules which should not be equal are not equal."""

    # Test 1: SMILES strings are not mol objects
    assert cocs_mol != cocs_mol.smiles

    # Test 2: changing the identifer makes mols not equal
    mol_with_different_id = Molecule(smiles=cocs_mol.smiles, identifier="different")
    assert cocs_mol != mol_with_different_id

    # Test 3: equality should only be true if class is the same,
    # so another object with the same fields should still be not equal

    @dataclass
    class FakeMoleculeClass:
        smiles: str
        identifier: Optional[str]
        metadata: dict[str, Any]

    fake_mol = FakeMoleculeClass(smiles=cocs_mol.smiles, identifier=None, metadata=dict())
    assert fake_mol != cocs_mol

    # Test 4: same molecule but with non-canonical SMILES will still compare to False
    non_canonical_mol = Molecule(smiles="SCOC", canonicalize=False)
    assert non_canonical_mol != cocs_mol


def test_frozen(cocs_mol: Molecule) -> None:
    """Test that the fields of the Molecule cannot be modified (i.e. is actually frozen)."""
    with pytest.raises(FrozenInstanceError):
        # type ignore is because mypy complains we are modifying a frozen field, which is the point of the test
        cocs_mol.smiles = "xyz"  # type: ignore[misc]


def test_canonicalization() -> None:
    """
    Test that the `canonicalize` argument works as expected,
    canonicalizing the SMILES if True and leaving it unchanged if False.
    """
    non_canonical_smiles = "OCC"
    canonical_smiles = "CCO"

    # Test 1: canonicalize=True should canonicalize the SMILES
    mol1 = Molecule(smiles=non_canonical_smiles, canonicalize=True)
    assert mol1.smiles == canonical_smiles

    # Test 2: canonicalize=False should leave the SMILES unchanged
    mol2 = Molecule(smiles=non_canonical_smiles, canonicalize=False)
    assert mol2.smiles == non_canonical_smiles


def test_make_rdkit_mol() -> None:
    """Test that the argument `make_rdkit_mol` works as expected."""

    # Test 1: make_rdkit_mol=True
    smiles = "CCO"
    mol_with_rdkit_mol = Molecule(smiles=smiles, make_rdkit_mol=True)
    assert "rdkit_mol" in mol_with_rdkit_mol.metadata

    # Test 2: make_rdkit_mol=False
    mol_without_rdkit_mol = Molecule(smiles=smiles, make_rdkit_mol=False)
    assert "rdkit_mol" not in mol_without_rdkit_mol.metadata

    # Test 3: accessing the rdkit mol should create it
    mol_without_rdkit_mol.rdkit_mol
    assert "rdkit_mol" in mol_without_rdkit_mol.metadata


def test_sorting() -> None:
    """Test that sorting molecules works as expected: by SMILES, then by identifier."""

    # Make individual molecules
    mol1 = Molecule("CC")
    mol2 = Molecule("CCC", identifier="")
    mol3 = Molecule("CCC", identifier="abc")
    mol4 = Molecule("CCC", identifier="def")

    # Test sorting
    mol_list = [mol4, mol3, mol2, mol1]
    mol_list.sort()
    assert mol_list == [mol1, mol2, mol3, mol4]


def test_create_and_compare() -> None:
    # NOTE: this test has some redundancy with tests from above
    mol_1 = Molecule("C")
    mol_2 = Molecule("C1=CC(N)=CC=C1")
    mol_3 = Molecule("c1cccc(N)c1")

    assert mol_1 < mol_2  # Lexicographical comparison on SMILES.
    assert mol_2 == mol_3  # Should be equal after canonicalization.


def test_order_of_components() -> None:
    assert Molecule("C.CC") == Molecule("CC.C")


def test_create_invalid() -> None:
    with pytest.raises(ValueError):
        Molecule("not-a-real-SMILES")
