from __future__ import annotations

import abc
from collections.abc import Collection

from syntheseus.interface.molecule import Molecule


class BaseMolInventory(abc.ABC):
    @abc.abstractmethod
    def is_purchasable(self, mol: Molecule) -> bool:
        """Whether or not a molecule is purchasable."""
        raise NotImplementedError

    def fill_metadata(self, mol: Molecule) -> None:
        """
        Fills any/all metadata of a molecule. This method should be fast to call,
        and many algorithms will assume that it sets `is_purchasable`.
        """

        # Default just adds whether the molecule is purchasable
        mol.metadata["is_purchasable"] = self.is_purchasable(mol)


class ExplicitMolInventory(BaseMolInventory):
    """
    Base class for MolInventories which store an explicit list of purchasable molecules.
    It exposes and additional method to explore this list.

    If it is unclear how a mol inventory might *not* have an explicit list of purchasable
    molecules, imagine a toy problem where every molecule with <= 10 atoms is purchasable.
    It is easy to check if a molecule has <= 10 atoms, but it is difficult to enumerate
    all molecules with <= 10 atoms.
    """

    @abc.abstractmethod
    def purchasable_mols(self) -> Collection[Molecule]:
        """Return a collection of all purchasable molecules."""


class SmilesListInventory(ExplicitMolInventory):
    """Most common type of inventory: a list of purchasable SMILES."""

    def __init__(self, smiles_list: list[str], canonicalize: bool = True):
        all_mols = [
            Molecule(s, make_rdkit_mol=False, canonicalize=canonicalize) for s in smiles_list
        ]
        self._mol_set = set(all_mols)

    def is_purchasable(self, mol: Molecule) -> bool:
        return mol in self._mol_set

    def purchasable_mols(self) -> Collection[Molecule]:
        return self._mol_set
