"""`MolInventory`: an inventory of InChIKey prefixes."""

from __future__ import annotations

import json
import warnings
from collections import OrderedDict
from collections.abc import Mapping
from enum import Enum
from typing import Any, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import inchi

from syntheseus.interface.molecule import Molecule
from syntheseus.search.mol_inventory.base import BaseMolInventory
from syntheseus.search.mol_inventory.standardiser import Standardiser


class InChIKeyPrefix(Enum):
    """Which leading part of the standard InChIKey identifies a molecule in a `MolInventory`.

    Both drop the InChIKey's last character, which only records protonation, so for example
    ammonium matches ammonia.
    """

    CONNECTIVITY = 14  # first block only: also ignores stereochemistry and isotopes
    STEREO = 25  # first two blocks: distinguishes stereochemistry and isotopes


def inventory_key(mol: Chem.Mol, prefix: InChIKeyPrefix) -> str:
    """The key of a standardised molecule; building and lookup must both use this function."""
    # Stereo only affects the InChIKey's second block, so skip it (~5% faster) for the first alone.
    options = "-SNon" if prefix is InChIKeyPrefix.CONNECTIVITY else ""
    mol_inchi = inchi.MolToInchi(mol, options=options, logLevel=None)
    if not mol_inchi:
        raise ValueError(f"InChI cannot represent {Chem.MolToSmiles(mol)!r}")
    return inchi.InchiToInchiKey(mol_inchi)[: prefix.value]


class InventoryLookupError(Exception):
    """A molecule cannot be standardised or keyed during lookup; clients can catch this."""


class MolInventory(BaseMolInventory):
    """Inventory of InChIKey prefixes, built and loaded by `MolInventoryBuilder`.

    A lookup standardises the molecule with the inventory's standardiser and binary-searches the
    sorted keys for the key of each resulting molecule. Keys cannot be turned back into molecules,
    so this inventory cannot list its contents.
    """

    CACHE_SIZE = 100_000

    def __init__(
        self, keys: np.ndarray, metadata: Mapping[str, Any], standardiser: Standardiser
    ) -> None:
        """`keys` must be sorted and unique; use `MolInventoryBuilder` to build or load one."""
        self._keys = keys
        self._metadata = dict(metadata)
        self._standardiser = standardiser
        self.key_prefix = InChIKeyPrefix[metadata["key"]]
        self._cache: OrderedDict[str, bool] = OrderedDict()

    @property
    def keys(self) -> np.ndarray:
        return self._keys

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def print_metadata(self) -> None:
        print(json.dumps(self._metadata, indent=2))

    def is_purchasable(self, mol: Molecule) -> bool:
        if mol.identifier is not None:
            warnings.warn(
                f"Molecule identifier {mol.identifier} will be ignored during inventory lookup"
            )

        return self._cached_lookup(mol)

    def _cached_lookup(self, mol: Molecule) -> bool:
        if (cached := self._cached_result(mol.smiles)) is not None:
            return cached

        result = self._lookup(mol)
        self._cache_result(mol.smiles, result)
        return result

    def _cached_result(self, smiles: str) -> Optional[bool]:
        if smiles not in self._cache:
            return None
        self._cache.move_to_end(smiles)
        return self._cache[smiles]

    def _cache_result(self, smiles: str, result: bool) -> None:
        self._cache[smiles] = result
        if len(self._cache) > self.CACHE_SIZE:
            self._cache.popitem(last=False)

    def _lookup(self, mol: Molecule) -> bool:
        try:
            standardised = self._standardiser.standardise(mol.rdkit_mol)
        except Exception as e:
            raise InventoryLookupError(f"Cannot standardise {mol.smiles!r}") from e
        try:
            keys = [np.bytes_(inventory_key(m, self.key_prefix)) for m in standardised]
        except ValueError as e:
            raise InventoryLookupError(f"Cannot compute the inventory key of {mol.smiles!r}") from e

        return all(self._contains(key) for key in keys)

    def _contains(self, key: np.bytes_) -> bool:
        index = np.searchsorted(self._keys, key)
        return bool(index < len(self._keys) and self._keys[index] == key)

    def __len__(self) -> int:
        return len(self._keys)
