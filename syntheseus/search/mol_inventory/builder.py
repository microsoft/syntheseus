"""Builds `MolInventory` objects and saves and loads their binary files."""

from __future__ import annotations

import hashlib
import json
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
    """Builds inventories with one standardiser and key prefix, and saves and loads them.

    Every molecule the standardiser returns for a source entry counts as purchasable. Building
    stops at the first entry that fails to load, standardise or produce a key.

    An inventory file holds magic bytes, a JSON metadata header and the sorted keys, which `load`
    memory-maps, so loading does no chemistry and takes milliseconds.
    """

    FORMAT_VERSION = 1
    _MAGIC = b"syntheseus-mol-inventory\n"

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
        """Build an inventory from `mols`; write it with `save`.

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
            "format_version": self.FORMAT_VERSION,
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

    def save(self, inventory: MolInventory, path: Union[str, Path]) -> None:
        header = json.dumps(inventory.metadata).encode()
        with open(path, "wb") as f:
            f.write(self._MAGIC)
            f.write(len(header).to_bytes(8, "little"))
            f.write(header)
            f.write(inventory.keys.tobytes())

    def load(self, path: Union[str, Path]) -> MolInventory:
        """Memory-map a prebuilt inventory, checking it was built with this standardiser and key."""
        with open(path, "rb") as f:
            if f.read(len(self._MAGIC)) != self._MAGIC:
                raise ValueError(f"{path} is not a MolInventory file")
            header_size = int.from_bytes(f.read(8), "little")
            metadata = json.loads(f.read(header_size))

        if metadata["format_version"] != self.FORMAT_VERSION:
            raise ValueError(f"{path} has unsupported format version {metadata['format_version']}")
        if metadata["key"] != self.key_prefix.name:
            raise ValueError(f"{path} has key {metadata['key']}, not {self.key_prefix.name}")
        expected = {"name": self.standardiser.name, "version": self.standardiser.version}
        if metadata["standardiser"] != expected:
            raise ValueError(
                f"{path} was built with standardiser {metadata['standardiser']}, not {expected}"
            )

        keys = np.memmap(
            path,
            dtype=f"S{self.key_prefix.value}",
            mode="r",
            offset=len(self._MAGIC) + 8 + header_size,
            shape=(metadata["num_keys"],),
        )
        return MolInventory(keys, metadata, self.standardiser)

    def generate_mol_hashed_keys(self, mol: Chem.Mol) -> list[str]:
        return [inventory_key(m, self.key_prefix) for m in self.standardiser.standardise(mol)]

    @staticmethod
    def _sha256(path: Union[str, Path]) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
