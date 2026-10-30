"""Prebuilds the binary inventory files that `MolInventory.load_from_file` reads."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import inchi

from syntheseus.search.mol_inventory.inventory import InChIKeyPrefix, MolInventory, inventory_key
from syntheseus.search.mol_inventory.loader import MolDataLoader
from syntheseus.search.mol_inventory.standardiser import Standardiser


class MolInventoryBuilder:
    """Builds inventories from molecules or source files, with one standardiser and key prefix.

    Every molecule the standardiser returns for a source entry counts as purchasable. Building
    stops at the first entry that fails to load, standardise or produce a key.
    """

    def __init__(
        self,
        standardiser: Standardiser,
        key_prefix: InChIKeyPrefix = InChIKeyPrefix.CONNECTIVITY,
    ) -> None:
        self.standardiser = standardiser
        self.key_prefix = key_prefix

    def build(
        self,
        mols: Iterable[Chem.Mol],
        *,
        name: str,
        version: str,
        source_sha256: Optional[str] = None,
    ) -> MolInventory:
        """Build an inventory from `mols`; write it with `MolInventory.save`.

        `name` and `version` identify the building-block set in the inventory's metadata, together
        with `source_sha256` when the molecules come from a file.
        """
        mol_hashed_keys = [
            mol_hashed_key for mol in mols for mol_hashed_key in self.generate_mol_hashed_keys(mol)
        ]
        batch_mol_hashed_keys = np.unique(
            np.array(mol_hashed_keys, dtype=f"S{self.key_prefix.value}")
        )

        metadata = {
            "name": name,
            "version": version,
            "format_version": MolInventory.FORMAT_VERSION,
            "key": self.key_prefix.name,
            "standardiser": {"name": self.standardiser.name, "version": self.standardiser.version},
            "rdkit_version": rdkit.__version__,
            "inchi_version": inchi.GetInchiVersion(),
            "source_sha256": source_sha256,
            "num_keys": len(batch_mol_hashed_keys),
        }
        return MolInventory(batch_mol_hashed_keys, metadata, self.standardiser)

    def build_from_file(
        self,
        source: Union[str, Path],
        *,
        name: str,
        version: str,
        file_format: str,
    ) -> MolInventory:
        """Build an inventory from the molecules in `source`, recording its SHA-256."""
        return self.build(
            MolDataLoader(source, file_format),
            name=name,
            version=version,
            source_sha256=self._sha256(source),
        )

    def generate_mol_hashed_keys(self, mol: Chem.Mol) -> list[str]:
        return [inventory_key(m, self.key_prefix) for m in self.standardiser.standardise(mol)]

    @staticmethod
    def _sha256(path: Union[str, Path]) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
