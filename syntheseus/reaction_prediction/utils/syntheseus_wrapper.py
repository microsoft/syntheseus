from __future__ import annotations

import syntheseus.search.reaction_models
from syntheseus.interface.models import BackwardReactionModel
from syntheseus.interface.molecule import Molecule
from syntheseus.search.chem import BackwardReaction, ReactionMetaData


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

    def _get_backward_reactions(self, mols: list[Molecule]) -> list[list[BackwardReaction]]:
        # Call the underlying model
        model_outputs = self._model(mols, self._num_results)

        # Convert the outputs to backward reactions
        reaction_outputs: list[list[BackwardReaction]] = []
        for pred_list in model_outputs:
            reaction_outputs.append([])  # Initialize the list
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

                rxn = BackwardReaction(
                    product=pred.input, reactants=frozenset(pred.output), metadata=metadata
                )
                reaction_outputs[-1].append(rxn)
        return reaction_outputs
