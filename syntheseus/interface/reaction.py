from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Collection, Generic, Optional, TypeVar

from syntheseus.interface.bag import Bag
from syntheseus.interface.molecule import SMILES_SEPARATOR, Molecule
from syntheseus.interface.typed_dict import TypedDict

ReactantsType = TypeVar("ReactantsType")
ProductType = TypeVar("ProductType")

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


@dataclass(frozen=True, order=False)
class Reaction(Generic[ReactantsType]):
    """General reaction class."""

    # The molecule that the prediction is for and the predicted output:
    reactants: ReactantsType = field(hash=True, compare=True)
    identifier: Optional[str] = field(default=None, hash=True, compare=True)

    # Dictionary to hold additional metadata.
    metadata: ReactionMetaData = field(
        default_factory=lambda: ReactionMetaData(),
        hash=False,
        compare=False,
    )

    @property
    @abstractmethod
    def reaction_smiles(self) -> str:
        pass

    @property
    @abstractmethod
    def unique_reactants(self) -> set[Molecule]:
        pass

    @property
    @abstractmethod
    def unique_products(self) -> set[Molecule]:
        pass


def combine_mols_to_string(mols: Collection[Molecule]) -> str:
    """Produces a consistent string representation of a collection of molecules."""
    return SMILES_SEPARATOR.join([mol.smiles for mol in sorted(mols)])


def reaction_string(reactants_str: str, product_str: str) -> str:
    """Produces a consistent string representation of a reaction."""
    return f"{reactants_str}{2*REACTION_SEPARATOR}{product_str}"


@dataclass(frozen=True, order=False)
class _MultiProductBase:
    """Dummy class to avoid non-default argument following default argument"""

    products: Bag[Molecule] = field(hash=True, compare=True)


@dataclass(frozen=True, order=False)
class MultiProductReaction(
    Reaction[Bag[Molecule]],
    _MultiProductBase,
):
    @property
    def reaction_smiles(self) -> str:
        return reaction_string(
            reactants_str=combine_mols_to_string(self.reactants),
            product_str=combine_mols_to_string(self.products),
        )

    @property
    def unique_reactants(self) -> set[Molecule]:
        return set(self.reactants)

    @property
    def unique_products(self) -> set[Molecule]:
        return set(self.products)


@dataclass(frozen=True, order=False)
class _SingleProductBase:
    """Dummy class to avoid non-default argument following default argument"""

    product: Molecule = field(hash=True, compare=True)


@dataclass(frozen=True, order=False)
class SingleProductReaction(Reaction[Bag[Molecule]], _SingleProductBase):
    @property
    def reaction_smiles(self) -> str:
        return reaction_string(
            reactants_str=combine_mols_to_string(self.reactants), product_str=self.product.smiles
        )

    @property
    def unique_reactants(self) -> set[Molecule]:
        return set(self.reactants)

    @property
    def unique_products(self) -> set[Molecule]:
        return {self.product}
