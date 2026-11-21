"""Reads the source files that inventories are built from, with one reader per file format."""

from __future__ import annotations

import abc
from collections.abc import Iterator
from pathlib import Path
from typing import Union

from rdkit import Chem


class MolFileReader(abc.ABC):
    @abc.abstractmethod
    def read(self, path: Path) -> Iterator[Chem.Mol]:
        """Yield the molecules in `path`."""


class SmiReader(MolFileReader):
    """One SMILES per line, optionally followed by whitespace and an ID, which is ignored."""

    def read(self, path: Path) -> Iterator[Chem.Mol]:
        with open(path) as f:
            for line in f:
                smiles = line.split()[0]
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError(f"Cannot parse SMILES {smiles!r}")
                yield mol


class MolDataLoader:
    """Streams molecules from an inventory source file, using the reader for its format."""

    _READERS: dict[str, MolFileReader] = {"smi": SmiReader()}

    def __init__(self, path: Union[str, Path], file_format: str) -> None:
        if file_format not in self._READERS:
            raise NotImplementedError(f"Unsupported inventory source format {file_format!r}")
        self.path = Path(path)
        self._reader = self._READERS[file_format]

    def __iter__(self) -> Iterator[Chem.Mol]:
        yield from self._reader.read(self.path)
