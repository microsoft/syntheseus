"""Inference wrapper for the Graph2Edits model.

Paper: https://www.nature.com/articles/s41467-023-38851-5
Code: https://github.com/Jamson-Zhong/Graph2Edits

The original Graph2Edits code is released under the MIT license.
Parts of this file are based on code from the GitHub repository above.
"""

from __future__ import annotations

import sys
from typing import Sequence

from rdkit import Chem, RDLogger

from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.reaction_prediction.inference.base import ExternalBackwardReactionModel
from syntheseus.reaction_prediction.utils.inference import (
    get_module_path,
    get_unique_file_in_dir,
    process_raw_smiles_outputs_backwards,
)
from syntheseus.reaction_prediction.utils.misc import suppress_outputs


class Graph2EditsModel(ExternalBackwardReactionModel):
    def __init__(self, *args, max_edit_steps: int = 9, **kwargs) -> None:
        """Initializes the Graph2Edits model wrapper.

        Assumed format of the model directory:
        - `model_dir` contains the model checkpoint as the only `*.pt` file
        """
        super().__init__(*args, **kwargs)

        import graph2edits
        import torch

        sys.path.insert(0, str(get_module_path(graph2edits)))

        from graph2edits.models import BeamSearch, Graph2Edits

        checkpoint = torch.load(
            get_unique_file_in_dir(self.model_dir, pattern="*.pt"), map_location=self.device
        )

        model = Graph2Edits(**checkpoint["saveables"], device=self.device)
        model.load_state_dict(checkpoint["state"])
        model.to(self.device)
        model.eval()

        # We set the beam size to a placeholder value for now and override it in `_get_reactions`.
        self.model = BeamSearch(model=model, step_beam_size=10, beam_size=None, use_rxn_class=False)
        self._max_edit_steps = max_edit_steps

        RDLogger.DisableLog("rdApp.*")

    def get_parameters(self):
        return self.model.model.parameters()

    def _get_reactions(
        self, inputs: list[Molecule], num_results: int
    ) -> list[Sequence[SingleProductReaction]]:
        import torch

        self.model.beam_size = num_results

        batch_predictions = []
        for input in inputs:
            # Copy the `rdkit` molecule as below we modify it in-place.
            mol = Chem.Mol(input.rdkit_mol)

            # Assign a dummy atom mapping as Graph2Edits depends on it. This has no connection to
            # the ground-truth atom mapping, which we do not have access to.
            for idx, atom in enumerate(mol.GetAtoms()):
                atom.SetAtomMapNum(idx + 1)

            with torch.no_grad(), suppress_outputs():
                try:
                    raw_results = self.model.run_search(
                        prod_smi=Chem.MolToSmiles(mol),
                        max_steps=self._max_edit_steps,
                        rxn_class=None,
                    )
                except IndexError:
                    # This can happen in some rare edge cases (e.g. "OBr").
                    raw_results = []

            # Errors are returned as a string "final_smi_unmapped"; we get rid of those here.
            raw_results = [
                raw_result
                for raw_result in raw_results
                if raw_result["final_smi"] != "final_smi_unmapped"
            ]

            batch_predictions.append(
                process_raw_smiles_outputs_backwards(
                    input=input,
                    output_list=[raw_result["final_smi"] for raw_result in raw_results],
                    metadata_list=[
                        {"probability": raw_result["prob"]} for raw_result in raw_results
                    ],
                )
            )

        return batch_predictions
