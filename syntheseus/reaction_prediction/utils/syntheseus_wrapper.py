from __future__ import annotations

import syntheseus.search.reaction_models
from syntheseus.interface.models import BackwardPredictionList, BackwardReactionModel
from syntheseus.interface.molecule import Molecule
from syntheseus.search.chem import ReactionMetaData


class SyntheseusBackwardReactionModel(syntheseus.search.reaction_models.BackwardReactionModel):
    """
    A syntheseus backward reaction model which wraps a single-step model from this repo.
    The resulting model can be used in search.

    NOTE: should be deleted if interfaces are combined.
    """

    def __init__(self, model: BackwardReactionModel, num_results: int, **kwargs):
        super().__init__(**kwargs)

        # These properties should not be modified since they will affect caching
        self._model = model
        self._num_results = num_results

    def _get_prediction_list(
        self, inputs: list[Molecule], num_results: int
    ) -> list[BackwardPredictionList]:
        # Call the underlying model
        # TODO: ignores `num_results`. Won't bother fixing though since this PR makes the class obsolete.
        model_outputs = self._model(inputs, self._num_results)

        # Transfer the metadata
        for pred_list in model_outputs:
            for pred in pred_list.predictions:
                # Read metadata
                metadata = ReactionMetaData()

                try:
                    # will raise ValueError if probability is not present
                    metadata["probability"] = pred.get_prob()  # type: ignore[typeddict-unknown-key]
                except ValueError:
                    pass

                if pred.score is not None:
                    metadata["score"] = pred.score

                if pred.rxnid is not None:
                    metadata["template"] = str(pred.rxnid)

                if pred.metadata is not None:
                    metadata["other_metadata"] = pred.metadata  # type: ignore[typeddict-unknown-key]

        return model_outputs
