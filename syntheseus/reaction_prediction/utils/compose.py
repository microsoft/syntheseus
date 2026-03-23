from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from syntheseus.interface.models import BackwardReactionModel
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction


logger = logging.getLogger(__file__)


class FilteredBackwardReactionModel(BackwardReactionModel):
    """Backward model wrapper that filters candidates via a dict of filter models."""

    def __init__(
        self,
        *,
        backward_model: BackwardReactionModel,
        filter_models: dict[str, Any],
        **kwargs,
    ) -> None:
        self.backward_model = backward_model
        self.filter_models = filter_models

        self.reset_acceptance_rates()

        default_kwargs = {  # By default match the backward model's settings
            "remove_duplicates": backward_model._remove_duplicates,
            "use_cache": backward_model._use_cache,
            "count_cache_in_num_calls": backward_model.count_cache_in_num_calls,
            "max_cache_size": backward_model._max_cache_size,
            "default_num_results": backward_model.default_num_results,
        }

        super().__init__(**(default_kwargs | kwargs))

    def reset(self, use_cache: Optional[bool] = None) -> None:
        super().reset(use_cache=use_cache)

        self.backward_model.reset(use_cache=use_cache)
        # TODO: Reset filter models

        self.reset_acceptance_rates()

    def reset_acceptance_rates(self) -> None:
        self._acceptance_rate_avg = 0.0
        self._acceptance_rate_count = 0
        self._acceptance_rate_per_filter_avg = {name: 0.0 for name in self.filter_models}
        self._acceptance_rate_per_filter_count = {name: 0 for name in self.filter_models}

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
                reaction
                for reaction, passed in zip(reactions_passed, passed_flags)
                if passed
            ]

            # Update per-filter acceptance rate
            filter_acceptance_rate = len(reactions_passed) / num_before
            self._acceptance_rate_per_filter_avg[filter_name] = (
                self._acceptance_rate_per_filter_avg[filter_name] * self._acceptance_rate_per_filter_count[filter_name]
                + filter_acceptance_rate
            ) / (self._acceptance_rate_per_filter_count[filter_name] + 1)
            self._acceptance_rate_per_filter_count[filter_name] += 1

        # Update overall acceptance rate
        acceptance_rate = len(reactions_passed) / len(reactions)
        self._acceptance_rate_avg = (
            self._acceptance_rate_avg * self._acceptance_rate_count + acceptance_rate
        ) / (self._acceptance_rate_count + 1)
        self._acceptance_rate_count += 1

        return reactions_passed

    @property
    def acceptance_rate(self) -> float:
        return self._acceptance_rate_avg

    @property
    def acceptance_rate_per_filter(self) -> dict[str, float]:
        return self._acceptance_rate_per_filter_avg.copy()
