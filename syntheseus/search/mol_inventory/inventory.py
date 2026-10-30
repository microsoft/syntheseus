"""`MolInventory`: an inventory of InChIKey prefixes prebuilt into a binary file."""

from __future__ import annotations

import functools
import json
import warnings
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any, Union

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
    mol_inchi = inchi.MolToInchi(mol, logLevel=None)
    if not mol_inchi:
        raise ValueError(f"InChI cannot represent {Chem.MolToSmiles(mol)!r}")
    return inchi.InchiToInchiKey(mol_inchi)[: prefix.value]


class InventoryLookupError(Exception):
    """A molecule's inventory key cannot be computed; clients can catch this."""


class MolInventory(BaseMolInventory):
    """Inventory of InChIKey prefixes, loaded from a binary file prebuilt by `MolInventoryBuilder`.

    A lookup standardises the molecule with the inventory's standardiser and binary-searches its key
    among the memory-mapped, sorted keys, so loading does no chemistry and takes milliseconds. Keys
    cannot be turned back into molecules, so this inventory cannot list its contents.

    Lookups cost ~0.2 ms (mostly InChI generation) and searches repeat them for the same molecules,
    so the results of the last `CACHE_SIZE` distinct SMILES are cached.
    """

    FORMAT_VERSION = 1
    CACHE_SIZE = 100_000
    _MAGIC = b"syntheseus-mol-inventory\n"

    def __init__(
        self, keys: np.ndarray, metadata: Mapping[str, Any], standardiser: Standardiser
    ) -> None:
        """`keys` must be sorted and unique; use `load_from_file` to read a prebuilt inventory."""
        self._keys = keys
        self._metadata = dict(metadata)
        self._standardiser = standardiser
        self.key_prefix = InChIKeyPrefix[metadata["key"]]
        self._cached_lookup = functools.lru_cache(maxsize=self.CACHE_SIZE)(self._lookup)

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def print_metadata(self) -> None:
        print(json.dumps(self._metadata, indent=2))

    def save(self, path: Union[str, Path]) -> None:
        header = json.dumps(self._metadata).encode()
        with open(path, "wb") as f:
            f.write(self._MAGIC)
            f.write(len(header).to_bytes(8, "little"))
            f.write(header)
            f.write(self._keys.tobytes())

    @classmethod
    def load_from_file(cls, path: Union[str, Path], standardiser: Standardiser) -> MolInventory:
        """Memory-map a prebuilt inventory, checking it was built with `standardiser`."""
        with open(path, "rb") as f:
            if f.read(len(cls._MAGIC)) != cls._MAGIC:
                raise ValueError(
                    f"{path} is not a MolInventory file; build one with MolInventoryBuilder"
                )
            header_size = int.from_bytes(f.read(8), "little")
            metadata = json.loads(f.read(header_size))

        if metadata["format_version"] != cls.FORMAT_VERSION:
            raise ValueError(f"{path} has unsupported format version {metadata['format_version']}")
        if metadata["key"] not in InChIKeyPrefix.__members__:
            raise ValueError(f"{path} has unknown key {metadata['key']!r}")
        expected = {"name": standardiser.name, "version": standardiser.version}
        if metadata["standardiser"] != expected:
            raise ValueError(
                f"{path} was built with standardiser {metadata['standardiser']}, not {expected}"
            )

        keys = np.memmap(
            path,
            dtype=f"S{InChIKeyPrefix[metadata['key']].value}",
            mode="r",
            offset=len(cls._MAGIC) + 8 + header_size,
            shape=(metadata["num_keys"],),
        )
        return cls(keys, metadata, standardiser)

    def is_purchasable(self, mol: Molecule) -> bool:
        if mol.identifier is not None:
            warnings.warn(
                f"Molecule identifier {mol.identifier} will be ignored during inventory lookup"
            )

        return self._cached_lookup(mol.smiles)

    def _lookup(self, smiles: str) -> bool:
        try:
            standardised = self._standardiser.standardise(Chem.MolFromSmiles(smiles))
        except Exception as e:
            raise InventoryLookupError(f"Cannot standardise {smiles!r}") from e
        try:
            keys = [np.bytes_(inventory_key(m, self.key_prefix)) for m in standardised]
        except ValueError as e:
            raise InventoryLookupError(f"Cannot compute the inventory key of {smiles!r}") from e

        return all(self._contains(key) for key in keys)

    def _contains(self, key: np.bytes_) -> bool:
        index = np.searchsorted(self._keys, key)
        return bool(index < len(self._keys) and self._keys[index] == key)

    def __len__(self) -> int:
        return len(self._keys)
