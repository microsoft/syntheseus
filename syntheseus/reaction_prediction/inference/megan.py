"""Inference wrapper for the MEGAN model.

Paper: https://arxiv.org/abs/2006.15426
Code: https://github.com/molecule-one/megan

The original MEGAN code is released under the MIT license.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional, Sequence

from rdkit import Chem

from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.reaction_prediction.inference.base import ExternalBackwardReactionModel
from syntheseus.reaction_prediction.utils.inference import (
    get_module_path,
    get_unique_file_in_dir,
    process_raw_smiles_outputs_backwards,
)
from syntheseus.reaction_prediction.utils.misc import suppress_outputs


class MEGANModel(ExternalBackwardReactionModel):
    def __init__(
        self,
        *args,
        n_max_atoms: int = 200,
        max_gen_steps: int = 16,
        beam_batch_size: int = 10,
        **kwargs,
    ) -> None:
        """Initializes the MEGAN model wrapper.

        Assumed format of the model directory:
        - `model_dir` contains the config as the only `*.gin` file
        - `model_dir/model_best.pt` is the model checkpoint
        - `model_dir/{featurizer_key}` contains files needed to build MEGAN's featurizer
        """
        super().__init__(*args, **kwargs)

        # Silence tensorflow's warnings.
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        import gin
        import megan
        import torch

        if self.device == "cpu" and torch.cuda.is_available():
            raise ValueError(
                "MEGAN currently does not support running on CPU if a GPU is available"
            )

        # Extract the path to the `megan` module.
        module_path = Path(get_module_path(megan))

        os.environ["PROJECT_ROOT"] = str(module_path.parent)
        sys.path.insert(0, str(module_path))

        def _get_root_logger_log_file_paths() -> set[Path]:
            return set(
                Path(handler.baseFilename)
                for handler in logging.root.handlers
                if isinstance(handler, RotatingFileHandler)
            )

        # The (seemingly unused) import below is needed for `gin` configurables to get registered.
        with suppress_outputs():
            preexisting_paths = _get_root_logger_log_file_paths()

            from bin.train import train_megan  # noqa: F401
            from src.config import get_featurizer
            from src.feat.megan_graph import MeganTrainingSamplesFeaturizer
            from src.model.megan import Megan as MeganModel
            from src.model.megan_utils import RdkitCache, get_base_action_masks
            from src.utils import load_state_dict

            # Delete logging files created by MEGAN.
            new_paths = _get_root_logger_log_file_paths() - preexisting_paths

            for path in new_paths:
                path.unlink()

            for path in new_paths:
                if path.parent.exists():
                    path.parent.rmdir()

        self.n_max_atoms = n_max_atoms
        self.max_gen_steps = max_gen_steps
        self.beam_batch_size = beam_batch_size

        # Get the model config using `gin`.
        gin.parse_config_file(get_unique_file_in_dir(self.model_dir, pattern="*.gin"))

        # Set up the data featurizer.
        featurizer_key = gin.query_parameter("train_megan.featurizer_key")
        featurizer = get_featurizer(featurizer_key)

        # Get the action vocab and masks.
        assert isinstance(featurizer, MeganTrainingSamplesFeaturizer)
        self.action_vocab = featurizer.get_actions_vocabulary(self.model_dir)
        self.base_action_masks = get_base_action_masks(
            n_max_atoms + 1, action_vocab=self.action_vocab
        )
        self.rdkit_cache = RdkitCache(props=self.action_vocab["props"])

        # Load the MEGAN model.
        checkpoint = load_state_dict(Path(self.model_dir) / "model_best.pt")
        self.model = MeganModel(
            n_atom_actions=self.action_vocab["n_atom_actions"],
            n_bond_actions=self.action_vocab["n_bond_actions"],
            prop2oh=self.action_vocab["prop2oh"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    def get_parameters(self):
        return self.model.parameters()

    def _mols_to_batch(self, inputs: list[Molecule]) -> list[Optional[Chem.Mol]]:
        from src.feat.utils import fix_explicit_hs

        # Inputs to the model are list of `rdkit` molecules.
        input_batch = []
        for input_mol in inputs:
            # Copy the `rdkit` molecule as below we modify it in-place.
            mol = Chem.Mol(input_mol.rdkit_mol)

            for i, a in enumerate(mol.GetAtoms()):
                a.SetAtomMapNum(i + 1)

            try:
                input_batch.append(fix_explicit_hs(mol))
            except Exception:
                # MEGAN sometimes produces broken molecules containing C+ atoms which pass `rdkit`
                # sanitization but fail in `fix_explicit_hs`. We block these here to avoid making
                # predictions for them.
                input_batch.append(None)

        return input_batch

    def _get_reactions(
        self, inputs: list[Molecule], num_results: int
    ) -> list[Sequence[SingleProductReaction]]:
        import torch
        from src.model.beam_search import beam_search

        # Get the inputs into the right form to call the underlying model.
        batch = self._mols_to_batch(inputs)
        batch_valid = [mol for mol in batch if mol is not None]
        batch_valid_idxs = [idx for idx, mol in enumerate(batch) if mol is not None]

        if batch_valid:
            with torch.no_grad():
                beam_search_results = beam_search(
                    [self.model],
                    batch_valid,
                    rdkit_cache=self.rdkit_cache,
                    max_steps=self.max_gen_steps,
                    beam_size=num_results,
                    batch_size=self.beam_batch_size,
                    base_action_masks=self.base_action_masks,
                    max_atoms=self.n_max_atoms,
                    reaction_types=None,
                    action_vocab=self.action_vocab,
                )  # returns a list of `beam_size` results for each input molecule
        else:
            beam_search_results = []

        assert len(batch_valid_idxs) == len(beam_search_results)

        all_outputs: list[list[dict[str, Any]]] = [[] for _ in batch]
        for idx, raw_outputs in zip(batch_valid_idxs, beam_search_results):
            all_outputs[idx] = raw_outputs

        return [
            process_raw_smiles_outputs_backwards(
                input=input,
                output_list=[prediction["final_smi_unmapped"] for prediction in raw_outputs],
                metadata_list=[{"probability": prediction["prob"]} for prediction in raw_outputs],
            )
            for input, raw_outputs in zip(inputs, all_outputs)
        ]
