"""Inference wrapper for the Graph Logic Network (GLN) model.

Paper: https://arxiv.org/abs/2001.01408
Code: https://github.com/Hanjun-Dai/GLN

The original GLN code is released under the MIT license.
"""

import sys
from pathlib import Path
from typing import List, Union

from syntheseus.interface.models import BackwardPredictionList, BackwardReactionModel
from syntheseus.interface.molecule import Molecule
from syntheseus.reaction_prediction.utils.inference import process_raw_smiles_outputs
from syntheseus.reaction_prediction.utils.misc import suppress_outputs


class GLNModel(BackwardReactionModel):
    def __init__(
        self,
        model_dir: Union[str, Path],
        device: str = "cuda:0",
        dataset_name: str = "schneider50k",
    ) -> None:
        """Initializes the GLN model wrapper.

        Assumed format of the model directory:
        - `model_dir` contains files necessary to build `RetroGLN`
        - `model_dir/{dataset_name}.ckpt` is the model checkpoint
        - `model_dir/cooked_{dataset_name}/atom_list.txt` is the atom type list
        """
        import torch

        chkpt_path = Path(model_dir) / f"{dataset_name}.ckpt"
        args = {
            "dropbox": model_dir,
            "data_name": dataset_name,
            "model_for_test": chkpt_path,
            "tpl_name": "default",
            "f_atoms": Path(model_dir) / f"cooked_{dataset_name}" / "atom_list.txt",
            "gpu": torch.device(device).index,
        }

        # Suppress most of the prints from GLN's internals. This only works on messages that
        # originate from Python, so the C++-based ones slip through.
        with suppress_outputs():
            # GLN makes heavy use of global state (saved either in `gln.common.cmd_args` or `sys.argv`),
            # so we have to hack both of these sources below.
            from gln.common.cmd_args import cmd_args

            sys.argv = []
            for name, value in args.items():
                setattr(cmd_args, name, value)
                sys.argv += [f"-{name}", str(value)]

            # The global state hackery has to happen before this.
            from gln.test.model_inference import RetroGLN

            self.model = RetroGLN(model_dir, chkpt_path)

    def get_parameters(self):
        return self.model.gln.parameters()

    def _get_model_predictions(self, input: Molecule, num_results: int) -> BackwardPredictionList:
        with suppress_outputs():
            result = self.model.run(input.smiles, num_results, num_results)

        if result is None:
            return BackwardPredictionList(input=input, predictions=[])
        else:
            # `scores` are actually probabilities (produced by running `softmax`).
            return process_raw_smiles_outputs(
                input=input,
                output_list=result["reactants"],
                kwargs_list=[{"probability": probability} for probability in result["scores"]],
            )

    def __call__(self, inputs: List[Molecule], num_results: int) -> List[BackwardPredictionList]:
        return [self._get_model_predictions(input, num_results=num_results) for input in inputs]
