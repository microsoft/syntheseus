from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from syntheseus.interface.bag import Bag
from syntheseus.interface.molecule import Molecule, molecule_bag_to_smiles
from syntheseus.interface.typed_dict import TypedDict

REACTION_SEPARATOR = ">"


class ReactionMetaData(TypedDict, total=False):
    """Class to add typing to optional meta-data fields for reactions."""

    cost: float
    template: str
    source: str  # any explanation of the source of this reaction
    probability: float  # probability for this reaction (e.g. from a model)
    log_probability: float  # log probability for this reaction (should match log of above)
    score: float  # any kind of score for this reaction (e.g. softmax value, probability)
    confidence: float  # confidence (probability) that this reaction is possible
    reaction_id: int  # template id or other kind of reaction id, if applicable
    reaction_smiles: str  # reaction smiles for this reaction
    ground_truth_match: bool  # whether this reaction matches ground truth


def reaction_string(reactants_str: str, products_str: str) -> str:
    """Produces a consistent string representation of a reaction."""
    return f"{reactants_str}{2 * REACTION_SEPARATOR}{products_str}"


@dataclass(frozen=True, order=False)
class Reaction:
    reactants: Bag[Molecule] = field(hash=True, compare=True)
    products: Bag[Molecule] = field(hash=True, compare=True)
    identifier: Optional[str] = field(default=None, hash=True, compare=True)

    # Dictionary to hold additional metadata.
    metadata: ReactionMetaData = field(
        default_factory=lambda: ReactionMetaData(),
        hash=False,
        compare=False,
    )

    @property
    def unique_reactants(self) -> set[Molecule]:
        return set(self.reactants)

    @property
    def unique_products(self) -> set[Molecule]:
        return set(self.products)

    @property
    def reactants_str(self) -> str:
        return molecule_bag_to_smiles(self.reactants)

    @property
    def products_str(self) -> str:
        return molecule_bag_to_smiles(self.products)

    @property
    def reaction_smiles(self) -> str:
        return reaction_string(reactants_str=self.reactants_str, products_str=self.products_str)

    def __str__(self) -> str:
        output = self.reaction_smiles
        if self.identifier is not None:
            output += f" ({self.identifier})"
        return output


@dataclass(frozen=True, order=False)
class SingleProductReaction(Reaction):
    def __init__(self, *, reactants: Bag[Molecule], product: Molecule, **kwargs) -> None:
        super().__init__(reactants=reactants, products=Bag([product]), **kwargs)

    @property
    def product(self) -> Molecule:
        """Handle for the single product of this reaction."""
        assert len(self.products) == 1  # Guaranteed in `__init__`.
        return next(iter(self.products))
