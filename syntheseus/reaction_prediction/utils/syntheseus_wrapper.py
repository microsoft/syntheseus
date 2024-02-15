from __future__ import annotations

from typing import Sequence

import syntheseus.search.reaction_models
from syntheseus.interface.models import BackwardReactionModel
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction


class SyntheseusBackwardReactionModel(syntheseus.search.reaction_models.BackwardReactionModel):
    """
    A syntheseus backward reaction model which wraps a single-step model from this repo.
    The resulting model can be used in search.
    """

    def __init__(self, model: BackwardReactionModel, num_results: int, **kwargs):
        super().__init__(**kwargs)

        # These properties should not be modified since they will affect caching
        self._model = model
        self._num_results = num_results

    def _get_backward_reactions(
        self, mols: list[Molecule]
    ) -> list[Sequence[SingleProductReaction]]:
        # Call the underlying model
        model_outputs = self._model(mols, self._num_results)
        return model_outputs
