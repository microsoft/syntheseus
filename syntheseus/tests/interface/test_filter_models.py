"""Tests for the `ReactionFilterModel` base class using a dummy filter."""
from __future__ import annotations

from typing import Sequence

from syntheseus.interface.models import ReactionFilterModel
from syntheseus.interface.reaction import SingleProductReaction


class ReactantListFilterModel(ReactionFilterModel):
    """Dummy filter that accepts a reaction if every reactant is in the allow list."""

    def __init__(self, allowed_smiles, **kwargs) -> None:
        super().__init__(**kwargs)
        self._allowed = set(allowed_smiles)

    def _get_acceptance(self, reactions: Sequence[SingleProductReaction]) -> list[bool]:
        return [all(r.smiles in self._allowed for r in rxn.reactants) for rxn in reactions]


def test_filter_model_basic() -> None:
    """Filter returns one boolean per reaction, in input order."""
    rxn_accept = SingleProductReaction.from_reaction_smiles("C.C>>CC")
    rxn_reject = SingleProductReaction.from_reaction_smiles("CO>>CO")

    filter = ReactantListFilterModel(["C"])
    assert filter([rxn_accept, rxn_reject]) == [True, False]


def test_filter_model_caching() -> None:
    """With caching enabled, repeated reactions should not re-invoke the underlying filter."""
    rxn = SingleProductReaction.from_reaction_smiles("C.C>>CC")
    other = SingleProductReaction.from_reaction_smiles("CO>>CO")

    filter = ReactantListFilterModel(["C"], use_cache=True)
    filter([rxn])
    filter([rxn, other, rxn])

    # Two unique reactions seen across calls -> exactly two cache misses.
    assert filter.num_calls() == 2
    assert filter.num_calls(count_cache=True) == 4

    filter.reset()
    assert filter.num_calls() == 0
    assert filter.cache_size == 0
