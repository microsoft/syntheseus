from __future__ import annotations

import abc

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
