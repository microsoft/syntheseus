import pytest

from syntheseus.interface.molecule import Molecule


def test_create_and_compare() -> None:
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
