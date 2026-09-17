"""A small graph attention network for corners, dense and masked.

Why dense? A corner graph is complete and has <= 22 nodes. Batched [B, N, N]
attention is then *exactly* message passing on a complete graph (what TacticAI
used), without sparse-graph machinery. The whole model is ~50 lines of PyTorch.

Layer (GATv2-style, with edge features):
    e_ij   = edge MLP input: [dx, dy, distance, same_team]   (computed from node positions)
    score  = a^T LeakyReLU(W_q h_i + W_k h_j + W_e e_ij)     (per head)
    alpha  = softmax over valid j
    h_i   <- h_i + W_o concat_heads( sum_j alpha_ij W_v h_j )     then LayerNorm + FFN

GATv2 vs GAT: the original GAT applies the LeakyReLU *after* the dot product with
`a`, which makes the ranking of neighbours the same for every query node ("static
attention"). GATv2 moves the nonlinearity inside, so each player can attend to a
different set of others (dynamic attention). That's the variant TacticAI used.

Heads on top of the shared trunk:
  * receiver: one logit per node, non-candidates masked to -inf, softmax over the corner
  * shot:     masked mean+max pooling of node states + graph features -> one logit

Multi-task training shares what the trunk learns about marking and space between both
tasks. That matters with ~1-2k corners.
"""

from __future__ import annotations

import torch
from torch import nn

from phds.features.corner_graph import NODE_FEATURES

X_IDX, Y_IDX, ATT_IDX = (NODE_FEATURES.index(k) for k in ("x", "y", "attacker"))
EDGE_DIM = 4


def edge_features(x: torch.Tensor) -> torch.Tensor:
    """[B, N, F] node features -> [B, N, N, 4] pairwise geometry (in pitch-scaled units)."""
    pos = torch.stack([x[..., X_IDX] * 120.0, x[..., Y_IDX] * 80.0], dim=-1) / 20.0
    delta = pos[:, None, :, :] - pos[:, :, None, :]  # j - i
    dist = delta.norm(dim=-1, keepdim=True)
    same = (x[..., ATT_IDX][:, :, None] == x[..., ATT_IDX][:, None, :]).float().unsqueeze(-1)
    return torch.cat([delta, dist, same], dim=-1)


class DenseGATv2Layer(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.hd = heads, dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.e = nn.Linear(EDGE_DIM, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.a = nn.Parameter(torch.randn(heads, self.hd) * 0.1)
        self.out = nn.Linear(dim, dim)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, h, e, mask):
        b, n, _ = h.shape
        split = lambda t: t.view(*t.shape[:-1], self.heads, self.hd)
        q, k, v = split(self.q(h)), split(self.k(h)), split(self.v(h))  # [B, N, H, hd]
        z = q[:, :, None] + k[:, None, :] + split(self.e(e))  # [B, N, N, H, hd]
        score = (nn.functional.leaky_relu(z, 0.2) * self.a).sum(-1)  # [B, N, N, H]
        score = score.masked_fill(~mask[:, None, :, None], float("-inf"))
        alpha = self.drop(torch.softmax(score, dim=2))
        msg = torch.einsum("bijh,bjhd->bihd", alpha, v).reshape(b, n, -1)
        h = self.norm1(h + self.drop(self.out(msg)))
        h = self.norm2(h + self.drop(self.ffn(h)))
        return h * mask[..., None]


class CornerGNN(nn.Module):
    def __init__(self, node_dim: int, graph_dim: int, hidden: int = 64, layers: int = 3,
                 heads: int = 4, dropout: float = 0.1):  # fmt: skip
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(node_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.layers = nn.ModuleList(DenseGATv2Layer(hidden, heads, dropout) for _ in range(layers))
        self.receiver_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.shot_head = nn.Sequential(
            nn.Linear(2 * hidden + graph_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )  # fmt: skip

    def forward(self, x, mask, cand, g):
        e = edge_features(x)
        h = self.embed(x) * mask[..., None]
        for layer in self.layers:
            h = layer(h, e, mask)
        recv_logits = self.receiver_head(h).squeeze(-1).masked_fill(~cand, float("-inf"))
        m = mask[..., None].float()
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        mx = h.masked_fill(~mask[..., None], float("-inf")).amax(1)
        shot_logit = self.shot_head(torch.cat([mean, mx, g], dim=-1)).squeeze(-1)
        return recv_logits, shot_logit


def receiver_loss(logits, target_dist, has_label):
    """Cross-entropy against a (possibly soft) target distribution over nodes."""
    if not has_label.any():
        return logits.new_zeros(())
    logp = torch.log_softmax(logits[has_label], dim=-1)
    t = target_dist[has_label]
    return -(t * logp.masked_fill(t == 0, 0.0)).sum(-1).mean()
