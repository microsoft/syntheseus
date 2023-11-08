import torch
import torch.nn.functional as F
from torch import nn


class Adapter(nn.Module):
    def __init__(self, dim, k=32):
        from dgl.nn import GINEConv

        super().__init__()
        self.gnn = GINEConv(nn.Linear(dim, dim))
        self.node_proj = nn.Linear(dim, 2)  # [tmp, p]
        self.edge_proj = nn.Linear(dim, 2)  # [tmp, p]

        self.node_dist_in_proj = nn.Linear(k, k)
        self.edge_dist_in_proj = nn.Linear(k, k)

        self.node_ffn = nn.Linear(dim + k, dim)
        self.edge_ffn = nn.Linear(dim * 2 + k, dim)

        nn.init.kaiming_uniform_(self.node_dist_in_proj.weight, a=100)
        nn.init.kaiming_uniform_(self.edge_dist_in_proj.weight, a=100)

        nn.init.zeros_(self.node_proj.weight)
        nn.init.zeros_(self.edge_proj.weight)

        nn.init.constant_(self.node_proj.bias[0], 10.0)
        nn.init.constant_(self.edge_proj.bias[0], 10.0)

    def forward(self, g, nfeat, efeat, ndist, edist):
        from local_retro.scripts.model_utils import pair_atom_feats

        x = self.gnn(g, nfeat, efeat)
        x = F.relu(x)

        ndist = F.relu(self.node_dist_in_proj(ndist))
        edist = F.relu(self.edge_dist_in_proj(edist))

        node_x = torch.cat((x, ndist), dim=-1)
        node_x = self.node_ffn(node_x)
        node_x = F.relu(node_x)
        node_x = self.node_proj(node_x)

        edge_x = pair_atom_feats(g, x)
        edge_x = F.relu(edge_x)
        edge_x = torch.cat((edge_x, edist), dim=-1)
        edge_x = self.edge_ffn(edge_x)
        edge_x = F.relu(edge_x)
        edge_x = self.edge_proj(edge_x)

        node_t = torch.clamp(node_x[:, 0], 1, 100)
        node_p = torch.sigmoid(node_x[:, 1])

        edge_t = torch.clamp(edge_x[:, 0], 1, 100)
        edge_p = torch.sigmoid(edge_x[:, 1])

        return (r.unsqueeze(-1) for r in (node_t, node_p, edge_t, edge_p))


def knn_prob(feats, store, lables, max_idx, k=32, temperature=5):
    from torch_scatter import scatter

    dis, idx = store.search(feats, k)  # [B, K]
    pred = lables[idx].unsqueeze(-1)  # [B, K, 1]

    re_compute_dists = -1 * dis
    knn_weight = torch.softmax(re_compute_dists / temperature, dim=-1).unsqueeze(-1)  # [B, K, 1]

    bsz = feats.shape[0]
    output = torch.zeros(bsz, k, max_idx).to(feats)

    scatter(src=knn_weight, out=output, index=pred, dim=-1)

    return output.sum(dim=1)
