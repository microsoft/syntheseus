"""Inference wrapper for the LocalRetro model.

Paper: https://pubs.acs.org/doi/10.1021/jacsau.1c00246
Code: https://github.com/kaist-amsg/LocalRetro

The original LocalRetro code is released under the Apache 2.0 license.
Parts of this file are based on code from the GitHub repository above.
"""

from pathlib import Path
from typing import Any, List, Sequence

from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.reaction_prediction.inference.base import ExternalBackwardReactionModel
from syntheseus.reaction_prediction.utils.inference import (
    get_unique_file_in_dir,
    process_raw_smiles_outputs_backwards,
)
from syntheseus.reaction_prediction.utils.misc import suppress_outputs


class LocalRetroModel(ExternalBackwardReactionModel):
    def __init__(self, *args, **kwargs) -> None:
        """Initializes the LocalRetro model wrapper.

        Assumed format of the model directory:
        - `model_dir` contains the model checkpoint as the only `*.pth` file
        - `model_dir` contains the config as the only `*.json` file
        - `model_dir/data` contains `*.csv` data files needed by LocalRetro
        """
        super().__init__(*args, **kwargs)

        from local_retro.Retrosynthesis import load_templates
        from local_retro.scripts.utils import init_featurizer, load_model

        data_dir = Path(self.model_dir) / "data"
        self.args = init_featurizer(
            {
                "mode": "test",
                "device": self.device,
                "model_path": get_unique_file_in_dir(self.model_dir, pattern="*.pth"),
                "config_path": get_unique_file_in_dir(self.model_dir, pattern="*.json"),
                "data_dir": data_dir,
                "rxn_class_given": False,
            }
        )

        with suppress_outputs():
            self.model = load_model(self.args)

        [
            self.args["atom_templates"],
            self.args["bond_templates"],
            self.args["template_infos"],
        ] = load_templates(self.args)

    def get_parameters(self):
        return self.model.parameters()

    def _mols_to_batch(self, mols: List[Molecule]) -> Any:
        from dgllife.utils import smiles_to_bigraph
        from local_retro.scripts.utils import collate_molgraphs_test

        graphs = [
            smiles_to_bigraph(
                mol.smiles,
                node_featurizer=self.args["node_featurizer"],
                edge_featurizer=self.args["edge_featurizer"],
                add_self_loop=True,
                canonical_atom_order=False,
            )
            for mol in mols
        ]

        return collate_molgraphs_test([(None, graph, None) for graph in graphs])[1]

    def _build_batch_predictions(
        self, batch, num_results: int, inputs: List[Molecule], batch_atom_logits, batch_bond_logits
    ) -> List[Sequence[SingleProductReaction]]:
        from local_retro.scripts.Decode_predictions import get_k_predictions
        from local_retro.scripts.get_edit import combined_edit, get_bg_partition

        graphs, nodes_sep, edges_sep = get_bg_partition(batch)
        start_node = 0
        start_edge = 0

        self.args["top_k"] = num_results
        self.args["raw_predictions"] = []

        for input, graph, end_node, end_edge in zip(inputs, graphs, nodes_sep, edges_sep):
            pred_types, pred_sites, pred_scores = combined_edit(
                graph,
                batch_atom_logits[start_node:end_node],
                batch_bond_logits[start_edge:end_edge],
                num_results,
            )
            start_node, start_edge = end_node, end_edge

            raw_predictions = [
                f"({pred_types[i]}, {pred_sites[i][0]}, {pred_sites[i][1]}, {pred_scores[i]:.3f})"
                for i in range(num_results)
            ]

            self.args["raw_predictions"].append([input.smiles] + raw_predictions)

        batch_predictions = []
        for idx, input in enumerate(inputs):
            try:
                raw_str_results = get_k_predictions(test_id=idx, args=self.args)[1][0]
            except RuntimeError:
                # In very rare cases we may get `rdkit` errors.
                raw_str_results = []

            # We have to `eval` the predictions as they come rendered into strings. Second tuple
            # component is empirically (on USPTO-50K test set) in [0, 1], resembling a probability,
            # but does not sum up to 1.0 (usually to something in [0.5, 2.0]).
            raw_results = [eval(str_result) for str_result in raw_str_results]

            if raw_results:
                raw_outputs, probabilities = zip(*raw_results)
            else:
                raw_outputs = probabilities = []

            batch_predictions.append(
                process_raw_smiles_outputs_backwards(
                    input=input,
                    output_list=raw_outputs,
                    metadata_list=[{"probability": probability} for probability in probabilities],
                )
            )

        return batch_predictions

    def _get_reactions(
        self, inputs: List[Molecule], num_results: int
    ) -> List[Sequence[SingleProductReaction]]:
        import torch
        from local_retro.scripts.utils import predict

        batch = self._mols_to_batch(inputs)
        batch_atom_logits, batch_bond_logits, _ = predict(self.args, self.model, batch)

        batch_atom_logits = torch.nn.Softmax(dim=1)(batch_atom_logits)
        batch_bond_logits = torch.nn.Softmax(dim=1)(batch_bond_logits)

        return self._build_batch_predictions(
            batch, num_results, inputs, batch_atom_logits, batch_bond_logits
        )
