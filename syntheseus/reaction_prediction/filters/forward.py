from __future__ import annotations

from typing import Sequence

from syntheseus.interface.models import ForwardReactionModel, ReactionFilterModel
from syntheseus.interface.reaction import Reaction


class ForwardReactionFilterModel(ReactionFilterModel):
    """Accepts a reaction if its product is among top-k forward predictions."""

    def __init__(self, *, forward_model: ForwardReactionModel, top_k: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.forward_model = forward_model
        self.top_k = top_k

    def _get_acceptance(self, reactions: list[Reaction]) -> Sequence[bool]:
        predictions = self.forward_model(
            [rxn.reactants for rxn in reactions], num_results=self.top_k
        )
        return [
            any(pred.product in rxn.products for pred in preds)
            for rxn, preds in zip(reactions, predictions)
        ]
