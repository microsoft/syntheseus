"""Tests for `ForwardReactionFilterModel`."""
from __future__ import annotations

from typing import Sequence

from syntheseus.interface.bag import Bag
from syntheseus.interface.models import ForwardReactionModel
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import Reaction, SingleProductReaction
from syntheseus.reaction_prediction.filters.forward import ForwardReactionFilterModel


class DictForwardModel(ForwardReactionModel):
    """Returns a hardcoded list of single-product predictions per reactant bag."""

    def __init__(self, predictions: dict[Bag[Molecule], list[Molecule]], **kwargs) -> None:
        super().__init__(**kwargs)
        self._predictions = predictions

    def _get_reactions(
        self, inputs: list[Bag[Molecule]], num_results: int
    ) -> list[Sequence[Reaction]]:
        return [
            [
                SingleProductReaction(reactants=reactants, product=p)
                for p in self._predictions[reactants][:num_results]
            ]
            for reactants in inputs
        ]

    def get_parameters(self):
        return []


def test_forward_filter_top_k() -> None:
    reactants = Bag([Molecule("C"), Molecule("C")])
    forward_model = DictForwardModel(
        {reactants: [Molecule("CC"), Molecule("CCO"), Molecule("CCS")]}
    )

    rxn_top1 = SingleProductReaction(reactants=reactants, product=Molecule("CC"))
    rxn_top3 = SingleProductReaction(reactants=reactants, product=Molecule("CCS"))
    rxn_miss = SingleProductReaction(reactants=reactants, product=Molecule("N"))

    filter_top1 = ForwardReactionFilterModel(forward_model=forward_model, top_k=1)
    assert filter_top1([rxn_top1, rxn_top3, rxn_miss]) == [True, False, False]

    filter_top3 = ForwardReactionFilterModel(forward_model=forward_model, top_k=3)
    assert filter_top3([rxn_top1, rxn_top3, rxn_miss]) == [True, True, False]
