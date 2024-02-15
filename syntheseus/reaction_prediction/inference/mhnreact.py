"""Inference wrapper for the MHNreact model.

Paper: https://arxiv.org/abs/2104.03279
Code: https://github.com/ml-jku/mhn-react

The original MHNreact code is released under the BSD-2-Clause license.
"""

import json
from collections import defaultdict
from functools import partial
from typing import List, Sequence

from tqdm.contrib import concurrent

from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.reaction_prediction.inference.base import ExternalBackwardReactionModel
from syntheseus.reaction_prediction.utils.inference import (
    get_unique_file_in_dir,
    process_raw_smiles_outputs_backwards,
)
from syntheseus.reaction_prediction.utils.misc import cpu_count, suppress_outputs


class MHNreactModel(ExternalBackwardReactionModel):
    def __init__(
        self,
        *args,
        use_FPF: bool = True,
        num_processes: int = cpu_count(),
        chunksize: int = 64,
        num_additional_templates_to_run: int = 1000,
        **kwargs,
    ) -> None:
        """Initializes the MHNreact model wrapper.

        Assumed format of the model directory:
        - `model_dir` contains the model checkpoint as the only `*.pt` file
        - `model_dir` contains the config as the only `*.json` file
        - `model_dir` contains the data file as the only `*.csv.gz` file
        """
        super().__init__(*args, **kwargs)

        import torch
        from mhnreact import data, model

        with open(get_unique_file_in_dir(self.model_dir, pattern="*.json"), "r") as conf:
            conf_dict = json.load(conf)
        conf_dict["device"] = self.device

        conf = model.ModelConfig(**conf_dict)
        self.model = model.MHN(config=conf)

        self.use_FPF = use_FPF
        self.num_processes = num_processes
        self.chunksize = chunksize
        self.num_additional_templates_to_run = num_additional_templates_to_run

        params = torch.load(
            get_unique_file_in_dir(self.model_dir, pattern="*.pt"), map_location=self.device
        )

        with suppress_outputs():
            # Load templates.
            _, _, template_list, _ = data.load_dataset_from_csv(
                get_unique_file_in_dir(self.model_dir, pattern="*.csv.gz"), ssretroeval=True
            )

        self.model.template_list = list(template_list.values())
        self.template_product_smarts = [str(s).split(">")[0] for s in self.model.template_list]

        self.model.load_state_dict(params, strict=False)
        if "templates+noise" in params.keys():
            self.model.templates = params["templates+noise"]
        else:
            assert (
                conf_dict["concat_rand_template_thresh"] == -1
            ), "No templates+noise in checkpoint, but concat_rand_template_thresh is not -1"
            assert (
                conf_dict["template_fp_type2"] is None
            ), "currently no support for template_fp_type2"
            self.model.set_templates(
                self.model.template_list,
                which=conf_dict.get("template_fp_type"),
                fp_size=conf_dict.get("fp_size"),
                radius=conf_dict.get("fp_radius"),
                learnable=False,
                njobs=conf_dict.get("njobs", self.num_processes),
                only_templates_in_batch=conf_dict.get("only_templates_in_batch", False),
            )
        self.model.eval()
        self.model.X = self.model.template_encoder(self.model.templates)

    def get_parameters(self):
        return self.model.parameters()

    def _get_reactions(
        self, inputs: List[Molecule], num_results: int
    ) -> List[Sequence[SingleProductReaction]]:
        import pandas as pd
        import torch

        input_smiles_list = [inp.smiles for inp in inputs]

        with torch.no_grad():
            # Compute probabilities ranking the templates for each molecule.
            template_scores = self.model.forward_smiles(input_smiles_list)
            template_scores = self.model.softmax(template_scores)

            top_k_templates = (
                torch.topk(
                    template_scores,
                    num_results + self.num_additional_templates_to_run,
                    dim=1,
                    largest=True,
                )[1]
                .reshape(len(inputs), -1)
                .tolist()
            )
            template_scores = template_scores.detach().cpu().numpy()

        # Run templates.
        from mhnreact.molutils import smarts2appl
        from mhnreact.retroeval import run_templates

        if self.use_FPF:
            appl = smarts2appl(
                input_smiles_list, self.template_product_smarts, njobs=self.num_processes
            )

        batch_idxs = []
        templates_to_apply = []
        for batch_idx, top_templates in enumerate(top_k_templates):
            for template in top_templates:
                # If `use_FPF` perform an extra filter to weed out most inapplicable templates.
                if (not self.use_FPF) or ((appl[1][appl[0] == batch_idx] == template).any()):
                    batch_idxs.append(batch_idx)
                    templates_to_apply.append(template)

        # Temporarily disable tqdm (just the particular function used in MHNreact).
        process_map_orig = concurrent.process_map
        concurrent.process_map = partial(concurrent.process_map, disable=True)

        prod_idx_reactants, _ = run_templates(
            input_smiles_list,
            templates=self.model.template_list,
            appl=[batch_idxs, templates_to_apply],
            njobs=self.num_processes,
            chunksize=self.chunksize,
        )

        concurrent.process_map = process_map_orig

        # Now aggregate over same outcome (parts copied from `utils.sort_by_template_and_flatten`,
        # which does not expose the summed probabilities) and build the prediction objects.

        batch_predictions: List[Sequence[SingleProductReaction]] = []
        for idx in range(len(template_scores)):
            idx_prod_reactants = defaultdict(list)
            for k, v in prod_idx_reactants[idx].items():
                for iv in v:
                    idx_prod_reactants[iv].append(template_scores[idx, k])
            d2 = {k: sum(v) for k, v in idx_prod_reactants.items()}

            if len(d2) == 0:
                results = []
                probs = []
            else:
                df_sorted = pd.DataFrame.from_dict(d2, orient="index").sort_values(
                    0, ascending=False
                )

                # Limit to `num_results` results.
                df_sorted = df_sorted.iloc[:num_results, :]

                results = df_sorted.index.tolist()
                probs = df_sorted.values.ravel().tolist()

            batch_predictions.append(
                process_raw_smiles_outputs_backwards(
                    input=inputs[idx],
                    output_list=results,
                    metadata_list=[{"probability": probability} for probability in probs],
                )
            )

        return batch_predictions
