from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from syntheseus.interface.models import BackwardReactionModel, ReactionFilterModel
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction

logger = logging.getLogger(__file__)


class FilteredBackwardReactionModel(BackwardReactionModel):
    """Backward model wrapper that filters candidates using a sequence of filter models."""

    def __init__(
        self,
        *,
        backward_model: BackwardReactionModel,
        filter_models: dict[str, ReactionFilterModel],
        **kwargs,
    ) -> None:
        self.backward_model = backward_model
        self.filter_models = filter_models

        self.reset_acceptance_rates()

        default_kwargs: dict[str, Any] = {  # By default match the backward model's settings
            "remove_duplicates": backward_model._remove_duplicates,
            "use_cache": backward_model._use_cache,
            "count_cache_in_num_calls": backward_model.count_cache_in_num_calls,
            "max_cache_size": backward_model._max_cache_size,
            "default_num_results": backward_model.default_num_results,
        }

        super().__init__(**{**default_kwargs, **kwargs})

    def reset(self, use_cache: Optional[bool] = None) -> None:
        super().reset(use_cache=use_cache)

        self.backward_model.reset(use_cache=use_cache)
        for filter_model in self.filter_models.values():
            filter_model.reset(use_cache=use_cache)

        self.reset_acceptance_rates()

    def reset_acceptance_rates(self) -> None:
        self._num_accepted = 0
        self._num_seen = 0
        self._num_accepted_per_filter = {name: 0 for name in self.filter_models}
        self._num_seen_per_filter = {name: 0 for name in self.filter_models}

    def _get_reactions(
        self, inputs: list[Molecule], num_results: int
    ) -> list[Sequence[SingleProductReaction]]:
        return self.backward_model(inputs=inputs, num_results=num_results)

    def get_parameters(self):
        return self.backward_model.get_parameters()

    def filter_reactions(
        self, reactions: Sequence[SingleProductReaction]
    ) -> Sequence[SingleProductReaction]:
        reactions = list(super().filter_reactions(reactions))

        if not reactions:
            return reactions  # do not update acceptance rate if there were no reactions to filter

        reactions_passed: list[SingleProductReaction] = reactions

        for filter_name, filter_model in self.filter_models.items():
            if not reactions_passed:
                break

            num_before = len(reactions_passed)
            passed_flags = filter_model(reactions_passed)
            reactions_passed = [
                reaction for reaction, passed in zip(reactions_passed, passed_flags) if passed
            ]

            # Update per-filter acceptance counts.
            self._num_seen_per_filter[filter_name] += num_before
            self._num_accepted_per_filter[filter_name] += len(reactions_passed)

        # Update overall acceptance counts.
        self._num_seen += len(reactions)
        self._num_accepted += len(reactions_passed)

        return reactions_passed

    @property
    def acceptance_rate(self) -> float:
        return self._num_accepted / max(self._num_seen, 1)

    @property
    def acceptance_rate_per_filter(self) -> dict[str, float]:
        return {
            name: self._num_accepted_per_filter[name] / max(self._num_seen_per_filter[name], 1)
            for name in self.filter_models
        }
