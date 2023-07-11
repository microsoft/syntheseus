from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type, TypeVar

from syntheseus.interface.bag import Bag
from syntheseus.interface.molecule import REACTION_SEPARATOR, SMILES_SEPARATOR, Molecule
from syntheseus.reaction_prediction.chem.utils import (
    molecule_bag_to_smiles,
    remove_atom_mapping,
)
from syntheseus.reaction_prediction.utils.misc import undictify_bag_of_molecules

ReactionType = TypeVar("ReactionType", bound="ReactionSample")


@dataclass(frozen=True, order=False)
class ReactionSample:
    """
    Wraps around a reaction.

    It is frozen since it should not need to be edited,
    and this will auto-implement __eq__ and __hash__ methods.
    """

    reactants: Bag[Molecule] = field(hash=True, compare=True)
    products: Bag[Molecule] = field(hash=True, compare=True)
    reagents: str = field(default="", hash=True, compare=True)
    identifier: Optional[str] = field(default=None, hash=True, compare=True)

    mapped_reaction_smiles: Optional[str] = field(default=None, hash=False, compare=False)
    template: Optional[str] = field(default=None, hash=False, compare=False)

    metadata: Dict[str, Any] = field(
        default_factory=lambda: dict(),
        hash=False,
        compare=False,
    )

    @property
    def reactants_combined(self) -> str:
        return molecule_bag_to_smiles(self.reactants)

    @property
    def products_combined(self) -> str:
        return molecule_bag_to_smiles(self.products)

    @property
    def reaction_smiles(self) -> str:
        return f"{self.reactants_combined}{2 * REACTION_SEPARATOR}{self.products_combined}"

    @property
    def reaction_smiles_with_reagents(self) -> str:
        return (
            f"{self.reactants_combined}{REACTION_SEPARATOR}"
            f"{self.reagents}{REACTION_SEPARATOR}"
            f"{self.products_combined}"
        )

    @classmethod
    def from_dict(cls: Type[ReactionType], data: Dict[str, Any]) -> ReactionType:
        """Creates a sample from the given arguments ignoring superfluous ones."""
        for key in ["reactants", "products"]:
            data[key] = undictify_bag_of_molecules(data[key])

        return cls(
            **{
                key: value
                for key, value in data.items()
                if key in inspect.signature(cls).parameters
            }
        )

    @classmethod
    def from_reaction_smiles_strict(
        cls: Type[ReactionType], reaction_smiles: str, mapped: bool, **kwargs
    ) -> ReactionType:
        # Split the reaction SMILES and discard the reagents.
        [reactants_smiles, reagents_smiles, products_smiles] = [
            smiles_part.split(SMILES_SEPARATOR)
            for smiles_part in reaction_smiles.split(REACTION_SEPARATOR)
        ]

        if mapped:
            assert "mapped_reaction_smiles" not in kwargs
            kwargs["mapped_reaction_smiles"] = reaction_smiles

            reactants_smiles = [remove_atom_mapping(smiles) for smiles in reactants_smiles]
            products_smiles = [remove_atom_mapping(smiles) for smiles in products_smiles]

        return cls(
            reactants=Bag(Molecule(smiles=smiles) for smiles in reactants_smiles),
            products=Bag(Molecule(smiles=smiles) for smiles in products_smiles),
            reagents=SMILES_SEPARATOR.join(sorted(reagents_smiles)),
            **kwargs,
        )

    @classmethod
    def from_reaction_smiles(cls: Type[ReactionType], *args, **kwargs) -> Optional[ReactionType]:
        try:
            return cls.from_reaction_smiles_strict(*args, **kwargs)
        except Exception:
            return None
