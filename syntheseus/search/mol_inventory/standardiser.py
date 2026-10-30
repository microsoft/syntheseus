"""Standardisers turn one molecule into the molecules that identify it in an inventory."""

from __future__ import annotations

import abc

from rdkit import Chem


class Standardiser(abc.ABC):
    """Maps a molecule to a list of molecules, e.g. splitting a salt into its components.

    Inventories record every molecule a standardiser returns for an entry as purchasable, and
    store the standardiser's `name` and `version` so lookups can check they use the same one.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Identifies the standardiser in inventory metadata."""

    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Changes whenever the standardiser's output can change."""

    @abc.abstractmethod
    def standardise(self, mol: Chem.Mol) -> list[Chem.Mol]:
        """Return the standardised molecules for `mol`, without modifying it."""


class IdentityStandardiser(Standardiser):
    """Returns the molecule unchanged, in a one-element list."""

    @property
    def name(self) -> str:
        return "identity"

    @property
    def version(self) -> str:
        return "1"

    def standardise(self, mol: Chem.Mol) -> list[Chem.Mol]:
        return [mol]
