"""Deprecated inventories, kept for backward compatibility; use `MolInventory` instead."""

from __future__ import annotations

import abc
import warnings
from collections.abc import Collection
from pathlib import Path
from typing import Union

from rdkit import Chem

from syntheseus.interface.molecule import Molecule
from syntheseus.search.mol_inventory.base import BaseMolInventory


class ExplicitMolInventory(BaseMolInventory):
    """
    Base class for MolInventories which store an explicit list of purchasable molecules.
    It exposes and additional method to explore this list.

    Deprecated: use `MolInventory` instead.
    """

    @abc.abstractmethod
    def to_purchasable_mols(self) -> Collection[Molecule]:
        """Returns an explicit collection of all purchasable molecules.

        Likely expensive for large inventories, should be used mostly for testing or debugging.
        """

    def purchasable_mols(self) -> Collection[Molecule]:
        warnings.warn(
            "purchasable_mols is deprecated, use to_purchasable_mols instead", DeprecationWarning
        )
        return self.to_purchasable_mols()

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the number of purchasable molecules in the inventory."""


class SmilesListInventory(ExplicitMolInventory):
    """A list of purchasable SMILES.

    Deprecated: use `MolInventory` instead; this class will be removed in a future release.
    """

    def __init__(self, smiles_list: list[str], canonicalize: bool = True):
        warnings.warn(
            "SmilesListInventory is deprecated; use MolInventory instead",
            DeprecationWarning,
            stacklevel=2,
        )
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

    def to_purchasable_mols(self) -> Collection[Molecule]:
        return {Molecule(s, make_rdkit_mol=False, canonicalize=False) for s in self._smiles_set}

    def __len__(self) -> int:
        return len(self._smiles_set)

    @classmethod
    def load_from_file(cls, path: Union[str, Path], **kwargs) -> SmilesListInventory:
        """Load the inventory SMILES from a file."""
        with open(path, "rt") as f_inventory:
            return cls([line.strip() for line in f_inventory], **kwargs)
