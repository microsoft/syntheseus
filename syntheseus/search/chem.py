"""Defines molecules and reactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from syntheseus.interface.molecule import REACTION_SEPARATOR, SMILES_SEPARATOR, Molecule
from syntheseus.interface.typed_dict import TypedDict


class ReactionMetaData(TypedDict, total=False):
    """Class to add typing to optional meta-data fields for reactions."""

    cost: float
    template: str
    source: str  # any explanation of the source of this reaction
    probability: float  # probability for this reaction, used for multi-step search
    score: float  # any kind of score for this reaction (e.g. softmax value, probability)
    confidence: float  # confidence (probability) that this reaction is possible
    template_id: int  # index of the template used to generate this reaction


@dataclass(frozen=True, order=False)
class BackwardReaction:
    """
    A backward reaction mapping a single product molecule to a set of reactant molecules.
    "identifier" is an optional string to disambiguate between reactions with identical product/reactants.

    It is frozen since it should not need to be edited,
    and this will auto-implement __eq__ and __hash__ methods.
    """

    reactants: frozenset[Molecule] = field(hash=True, compare=True)
    product: Molecule = field(hash=True, compare=True)
    identifier: Optional[str] = field(default=None, hash=True, compare=True)

    metadata: ReactionMetaData = field(
        default_factory=lambda: ReactionMetaData(),
        hash=False,
        compare=False,
    )

    @property
    def reactants_combined(self) -> str:
        return SMILES_SEPARATOR.join([reactant.smiles for reactant in sorted(self.reactants)])

    @property
    def reaction_smiles(self) -> str:
        return f"{self.reactants_combined}{2 * REACTION_SEPARATOR}{self.product.smiles}"
