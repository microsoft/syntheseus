"""Inference wrapper for the RetroKNN model.

Paper: https://arxiv.org/abs/2306.04123
"""

from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.reaction_prediction.inference.local_retro import LocalRetroModel
from syntheseus.reaction_prediction.utils.inference import get_unique_file_in_dir


class RetroKNNModel(LocalRetroModel):
    """Warpper for RetroKNN model."""

    def __init__(self, model_dir: Optional[Union[str, Path]] = None, *args, **kwargs) -> None:
        """Initializes the RetroKNN model wrapper.

        Assumed format of the model directory:
        - `model_dir/local_retro` contains the files needed to load the LocalRetro wrapper
        - `model_dir/knn/` contains the adapter checkpoint as the only `*.pt` file
        - `model_dir/knn/datastore` contains the data store files
        """
        model_dir = Path(model_dir or self.get_default_model_dir())
        super().__init__(model_dir / "local_retro", *args, **kwargs)

        import torch

        from syntheseus.reaction_prediction.models.retro_knn import Adapter

        adapter_chkpt_path = get_unique_file_in_dir(Path(model_dir) / "knn", pattern="*.pt")
        datastore_path = Path(model_dir) / "knn" / "datastore"

        import faiss
        import faiss.contrib.torch_utils  # make faiss available for torch tensors

        def load_data_store(path: Path, device: str):
            index = faiss.read_index(str(path), faiss.IO_FLAG_ONDISK_SAME_DIR)

            if device == "cpu":
                return index
            else:
                res = faiss.StandardGpuResources()
                co = faiss.GpuClonerOptions()
                co.useFloat16 = True
                return faiss.index_cpu_to_gpu(res, 0, index, co)

        self.atom_store = load_data_store(datastore_path / "data.atom_idx", device=self.device)
        self.bond_store = load_data_store(datastore_path / "data.bond_idx", device=self.device)
        self.raw_data = np.load(datastore_path / "data.npz")

        self.adapter = Adapter(self.model.linearB.weight.shape[0], k=32).to(self.device)
        self.adapter.load_state_dict(torch.load(adapter_chkpt_path, map_location=self.device))
        self.adapter.eval()

    def _forward_localretro(self, bg):
        from local_retro.scripts.model_utils import pair_atom_feats, unbatch_feats, unbatch_mask

        bg = bg.to(self.device)
        node_feats = bg.ndata.pop("h").to(self.device)
        edge_feats = bg.edata.pop("e").to(self.device)

        node_feats = self.model.mpnn(bg, node_feats, edge_feats)
        atom_feats = node_feats
        bond_feats = self.model.linearB(pair_atom_feats(bg, node_feats))
        edit_feats, mask = unbatch_mask(bg, atom_feats, bond_feats)
        _, edit_feats = self.model.att(edit_feats, mask)

        atom_feats, bond_feats = unbatch_feats(bg, edit_feats)
        atom_outs = self.model.atom_linear(atom_feats)
        bond_outs = self.model.bond_linear(bond_feats)

        return atom_outs, bond_outs, atom_feats, bond_feats

    def _get_reactions(
        self, inputs: List[Molecule], num_results: int
    ) -> List[Sequence[SingleProductReaction]]:
        import torch

        from syntheseus.reaction_prediction.models.retro_knn import knn_prob

        batch = self._mols_to_batch(inputs)
        (
            batch_atom_logits,
            batch_bond_logits,
            atom_feats,
            bond_feats,
        ) = self._forward_localretro(batch)
        sg = batch.remove_self_loop().to(self.device)

        node_dis, _ = self.atom_store.search(atom_feats, k=32)
        edge_dis, _ = self.bond_store.search(bond_feats, k=32)

        node_t, node_p, edge_t, edge_p = self.adapter(
            sg, atom_feats, bond_feats, node_dis, edge_dis
        )

        batch_atom_prob_nn = torch.nn.Softmax(dim=1)(batch_atom_logits)
        batch_bond_prob_nn = torch.nn.Softmax(dim=1)(batch_bond_logits)

        atom_output_label = torch.from_numpy(self.raw_data["atom_output_label"]).to(self.device)
        bond_output_label = torch.from_numpy(self.raw_data["bond_output_label"]).to(self.device)

        batch_atom_prob_knn = knn_prob(
            atom_feats, self.atom_store, atom_output_label, batch_atom_logits.shape[1], 32, node_t
        )
        batch_bond_prob_knn = knn_prob(
            bond_feats, self.bond_store, bond_output_label, batch_bond_logits.shape[1], 32, edge_t
        )

        batch_atom_logits = node_p * batch_atom_prob_nn + (1 - node_p) * batch_atom_prob_knn
        batch_bond_logits = edge_p * batch_bond_prob_nn + (1 - edge_p) * batch_bond_prob_knn

        return self._build_batch_predictions(
            batch, num_results, inputs, batch_atom_logits, batch_bond_logits
        )
