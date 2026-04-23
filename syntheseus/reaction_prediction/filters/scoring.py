from __future__ import annotations

from typing import Sequence

from syntheseus.interface.models import ReactionFilterModel, ReactionScoringModel
from syntheseus.interface.reaction import Reaction


class ScoringReactionFilterModel(ReactionFilterModel):
    """Accepts a reaction if its score is at least a given threshold."""

    def __init__(
        self, *, scoring_model: ReactionScoringModel, min_score_threshold: float, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.scoring_model = scoring_model
        self.min_score_threshold = min_score_threshold

    def _get_acceptance(self, reactions: list[Reaction]) -> Sequence[bool]:
        return [score >= self.min_score_threshold for score in self.scoring_model(reactions)]
