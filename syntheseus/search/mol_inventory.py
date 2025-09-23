from __future__ import annotations

import abc
import warnings
from collections.abc import Collection
from pathlib import Path
from typing import Union

from rdkit import Chem

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
        if canonicalize:
            # For canonicalization we sequence `MolFromSmiles` and `MolToSmiles` to exactly match
            # the process employed in the `Molecule` class.
            smiles_list = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smiles_list]

        self._smiles_set = set(smiles_list)

    def is_purchasable(self, mol: Molecule) -> bool:
        if mol.identifier is not None:
            warnings.warn(
                f"Molecule identifier {mol.identifier} will be ignored during inventory lookup"
            )

        return mol.smiles in self._smiles_set

    def purchasable_mols(self) -> Collection[Molecule]:
        return {Molecule(s, make_rdkit_mol=False, canonicalize=False) for s in self._smiles_set}

    @classmethod
    def load_from_file(cls, path: Union[str, Path], **kwargs) -> SmilesListInventory:
        """Load the inventory SMILES from a file."""
        with open(path, "rt") as f_inventory:
            return cls([line.strip() for line in f_inventory], **kwargs)
