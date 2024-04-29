"""Inference wrapper for the Graph Logic Network (GLN) model.

Paper: https://arxiv.org/abs/2001.01408
Code: https://github.com/Hanjun-Dai/GLN

The original GLN code is released under the MIT license.
"""

import sys
from pathlib import Path
from typing import List, Sequence

from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.reaction_prediction.inference.base import ExternalBackwardReactionModel
from syntheseus.reaction_prediction.utils.inference import process_raw_smiles_outputs_backwards
from syntheseus.reaction_prediction.utils.misc import suppress_outputs


class GLNModel(ExternalBackwardReactionModel):
    def __init__(self, *args, dataset_name: str = "schneider50k", **kwargs) -> None:
        """Initializes the GLN model wrapper.

        Assumed format of the model directory:
        - `model_dir` contains files necessary to build `RetroGLN`
        - `model_dir/{dataset_name}.ckpt` is the model checkpoint
        - `model_dir/cooked_{dataset_name}/atom_list.txt` is the atom type list
        """
        super().__init__(*args, **kwargs)

        import torch

        chkpt_path = Path(self.model_dir) / f"{dataset_name}.ckpt"
        gln_args = {
            "dropbox": self.model_dir,
            "data_name": dataset_name,
            "model_for_test": chkpt_path,
            "tpl_name": "default",
            "f_atoms": Path(self.model_dir) / f"cooked_{dataset_name}" / "atom_list.txt",
            "gpu": torch.device(self.device).index,
        }

        # Suppress most of the prints from GLN's internals. This only works on messages that
        # originate from Python, so the C++-based ones slip through.
        with suppress_outputs():
            # GLN makes heavy use of global state (saved either in `gln.common.cmd_args` or `sys.argv`),
            # so we have to hack both of these sources below.
            from gln.common.cmd_args import cmd_args

            sys.argv = []
            for name, value in gln_args.items():
                setattr(cmd_args, name, value)
                sys.argv += [f"-{name}", str(value)]

            # The global state hackery has to happen before this.
            from gln.test.model_inference import RetroGLN

            self.model = RetroGLN(self.model_dir, chkpt_path)

    @property
    def name(self) -> str:
        return "GLN"

    def get_parameters(self):
        return self.model.gln.parameters()

    def _get_model_predictions(
        self, input: Molecule, num_results: int
    ) -> Sequence[SingleProductReaction]:
        with suppress_outputs():
            result = self.model.run(input.smiles, num_results, num_results)

        if result is None:
            return []
        else:
            # `scores` are actually probabilities (produced by running `softmax`).
            return process_raw_smiles_outputs_backwards(
                input=input,
                output_list=result["reactants"],
                metadata_list=[
                    {"probability": probability.item()} for probability in result["scores"]
                ],
            )

    def _get_reactions(
        self, inputs: List[Molecule], num_results: int
    ) -> List[Sequence[SingleProductReaction]]:
        return [self._get_model_predictions(input, num_results=num_results) for input in inputs]
