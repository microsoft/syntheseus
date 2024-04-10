"""Inference wrapper for the Chemformer model.

Paper: https://chemrxiv.org/engage/chemrxiv/article-details/60ee8a3eb95bdd06d062074b
Code: https://github.com/MolecularAI/Chemformer

The original Chemformer code is released under the Apache 2.0 license.
"""

import math
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple, cast

from syntheseus.interface.bag import Bag
from syntheseus.interface.models import InputType, ReactionType
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import ReactionMetaData
from syntheseus.reaction_prediction.inference.base import ExternalReactionModel
from syntheseus.reaction_prediction.utils.inference import (
    get_module_path,
    get_unique_file_in_dir,
    process_raw_smiles_outputs_backwards,
    process_raw_smiles_outputs_forwards,
)
from syntheseus.reaction_prediction.utils.misc import suppress_outputs


class ChemformerModel(ExternalReactionModel[InputType, ReactionType]):
    def __init__(self, *args, is_forward: bool = False, **kwargs) -> None:
        """Initializes the Chemformer model wrapper.

        Assumed format of the model directory:
        - `model_dir` contains the model checkpoint as the only `*.ckpt` file
        """
        self._is_forward = is_forward
        super().__init__(*args, **kwargs)

        # There should be exaclty one `*.ckpt` file under `model_dir`.
        chkpt_path = get_unique_file_in_dir(self.model_dir, pattern="*.ckpt")

        import chemformer

        # Fix for Chemformer's relative imports.
        chemformer_root_dir = get_module_path(chemformer)
        sys.path.insert(0, chemformer_root_dir)

        import chemformer.molbart.util as util
        from chemformer.molbart.decoder import DecodeSampler
        from chemformer.molbart.models.pre_train import BARTModel

        # Vocab path for the tokenizer is relative from Chemformer dir.
        self.tokenizer = util.load_tokeniser(
            Path(chemformer_root_dir) / util.DEFAULT_VOCAB_PATH, util.DEFAULT_CHEM_TOKEN_START
        )

        self.sampler = DecodeSampler(self.tokenizer, util.DEFAULT_MAX_SEQ_LEN)

        with suppress_outputs():
            self.model = BARTModel.load_from_checkpoint(chkpt_path, decode_sampler=self.sampler)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.sampler.max_seq_len = self.model.max_seq_len  # following Chemformer's codebase

    def get_parameters(self):
        return self.model.parameters()

    def _get_token_ids_and_mask(self, smiles) -> Tuple[Any, Any]:
        """Call the model tokeniser to get token ids and masks."""

        output = self.tokenizer.tokenise(smiles, pad=True)

        tokens = output["original_tokens"]
        mask = output["original_pad_masks"]

        # Truncate if we happened to exceed `max_seq_len`.
        if any(len(t) > self.model.max_seq_len for t in tokens):
            mask = [m[: self.model.max_seq_len] for m in mask]
            tokens = [t[: self.model.max_seq_len - 1] for t in tokens]
            for token_list in tokens:
                # We want to ensure that every truncated sequence ends with the `end_token`,
                # possibly followed by some `pad_tokens`:
                if token_list[-1] in [self.tokenizer.pad_token, self.tokenizer.end_token]:
                    token_list.append(self.tokenizer.pad_token)
                else:
                    token_list.append(self.tokenizer.end_token)

        return self.tokenizer.convert_tokens_to_ids(tokens), mask

    def _mols_to_batch(self, inputs: List[InputType]) -> Dict[str, Any]:
        import torch

        # Depending on direction we may need to concatenate bags of molecules into a single SMILES.
        if self.is_forward():
            smiles = [
                ".".join([x.smiles for x in input]) for input in cast(List[Bag[Molecule]], inputs)
            ]
        else:
            smiles = [mol.smiles for mol in cast(List[Molecule], inputs)]

        token_ids, mask = self._get_token_ids_and_mask(smiles)

        # Convert inputs to the model to tensors.
        return {
            "encoder_input": torch.tensor(token_ids).transpose(0, 1),
            "encoder_pad_mask": torch.tensor(mask, dtype=torch.bool).transpose(0, 1),
        }

    def _get_reactions(
        self, inputs: List[InputType], num_results: int
    ) -> List[Sequence[ReactionType]]:
        import torch

        # Get the data in to the right form to call the sampling method on the model.
        batch = self._mols_to_batch(inputs)

        device_batch = {
            key: val.to(self.device) if type(val) == torch.Tensor else val
            for key, val in batch.items()
        }

        # We have to set `num_beams` as an attribute of the model.
        self.model.num_beams = num_results

        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="__floordiv__ is deprecated")

            smiles_batch, batch_log_likelihoods = self.model.sample_molecules(
                device_batch, sampling_alg="beam"
            )

        # Choose processing function (controls reaction types).
        # `type: ignore[assignment]` statements are because link between is_forward/is_backward
        # and [InputType, ReactionType] is not visible to mypy.
        if self.is_forward():
            process_fn: Callable[
                [InputType, List[str], List[ReactionMetaData]], Sequence[ReactionType]
            ] = process_raw_smiles_outputs_forwards  # type: ignore[assignment]
        else:
            process_fn = process_raw_smiles_outputs_backwards  # type: ignore[assignment]

        return [
            process_fn(
                input,
                outputs,
                [
                    {"log_probability": log_prob, "probability": math.exp(log_prob)}
                    for log_prob in log_probs
                ],
            )
            for input, outputs, log_probs in zip(inputs, smiles_batch, batch_log_likelihoods)
        ]

    def is_forward(self):
        return self._is_forward
