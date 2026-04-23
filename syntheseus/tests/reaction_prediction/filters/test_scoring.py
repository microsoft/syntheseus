"""Tests for `ScoringReactionFilterModel`."""
from __future__ import annotations

from typing import Sequence

from syntheseus.interface.models import ReactionScoringModel
from syntheseus.interface.reaction import Reaction
from syntheseus.reaction_prediction.filters.scoring import ScoringReactionFilterModel


class DictScoringModel(ReactionScoringModel):
    """Returns a hardcoded score per reaction SMILES."""

    def __init__(self, scores: dict[str, float], **kwargs) -> None:
        super().__init__(**kwargs)
        self._scores = scores

    def _get_scores(self, reactions: list[Reaction]) -> Sequence[float]:
        return [self._scores[rxn.reaction_smiles] for rxn in reactions]


def test_scoring_filter_threshold() -> None:
    rxn_lo = Reaction.from_reaction_smiles("C.C>>CC")
    rxn_hi = Reaction.from_reaction_smiles("CO>>CO")
    scoring_model = DictScoringModel({rxn_lo.reaction_smiles: 0.2, rxn_hi.reaction_smiles: 0.8})

    filter = ScoringReactionFilterModel(scoring_model=scoring_model, min_score_threshold=0.1)
    assert filter([rxn_lo, rxn_hi]) == [True, True]

    filter = ScoringReactionFilterModel(scoring_model=scoring_model, min_score_threshold=0.5)
    assert filter([rxn_lo, rxn_hi]) == [False, True]

    filter = ScoringReactionFilterModel(scoring_model=scoring_model, min_score_threshold=0.9)
    assert filter([rxn_lo, rxn_hi]) == [False, False]
