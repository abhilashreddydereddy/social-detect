"""
Metadata encoder — Graph Attention Network over post propagation / account graph.

Paper: GAT → 256-d metadata feature vector.

We implement a self-contained multi-head GAT layer so we do not require
torch_geometric. Node features default to a fixed schema:
  [platform_onehot(8), account_age_norm, follower_log, sharing_velocity,
   geo_spread, engagement_rate, hour_sin, hour_cos] → 16-d per node.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        assert out_dim % heads == 0
        self.head_dim = out_dim // heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.empty(heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a)
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x:   (B, N, F)
        adj: (B, N, N) binary / weighted adjacency (include self-loops)
        """
        b, n, _ = x.shape
        h = self.W(x).view(b, n, self.heads, self.head_dim)  # (B,N,H,D)
        # Attention scores
        h_i = h.unsqueeze(2).expand(-1, -1, n, -1, -1)
        h_j = h.unsqueeze(1).expand(-1, n, -1, -1, -1)
        cat = torch.cat([h_i, h_j], dim=-1)  # (B,N,N,H,2D)
        e = self.leaky((cat * self.a.view(1, 1, 1, self.heads, 2 * self.head_dim)).sum(dim=-1))
        # Mask non-edges
        mask = (adj <= 0).unsqueeze(-1)  # (B,N,N,1)
        e = e.masked_fill(mask, -1e9)
        alpha = F.softmax(e, dim=2)
        alpha = self.dropout(alpha)
        # Aggregate
        out = torch.einsum("bnmh,bmhd->bnhd", alpha, h)
        return out.reshape(b, n, self.out_dim)


class MetadataEncoder(nn.Module):
    def __init__(self, in_dim: int = 16, hidden: int = 128, out_dim: int = 256, heads: int = 4) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.gat1 = GraphAttentionLayer(in_dim, hidden, heads=heads)
        self.gat2 = GraphAttentionLayer(hidden, out_dim, heads=heads)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, node_features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        node_features: (B, N, F)
        adj: (B, N, N)
        Returns graph-level embedding (B, 256) via mean pooling.
        """
        # Ensure self-loops
        eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype).unsqueeze(0)
        adj = torch.clamp(adj + eye, max=1.0)
        h = F.elu(self.gat1(node_features, adj))
        h = self.gat2(h, adj)
        h = self.norm(h)
        return h.mean(dim=1)


def build_trivial_graph(meta_vec: torch.Tensor, n_nodes: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand a flat metadata vector into a small star-graph for GAT.

    meta_vec: (B, F)  — if F < 16, zero-pad; if >, truncate.
    """
    b, f = meta_vec.shape
    feat = torch.zeros(b, 16, device=meta_vec.device, dtype=meta_vec.dtype)
    feat[:, : min(16, f)] = meta_vec[:, : min(16, f)]

    nodes = feat.unsqueeze(1).repeat(1, n_nodes, 1)  # (B, N, 16)
    # Perturb neighbor nodes slightly so the graph isn't identical copies
    noise = torch.randn_like(nodes) * 0.05
    noise[:, 0] = 0  # root node = exact metadata
    nodes = nodes + noise

    adj = torch.zeros(b, n_nodes, n_nodes, device=meta_vec.device, dtype=meta_vec.dtype)
    adj[:, 0, :] = 1.0
    adj[:, :, 0] = 1.0
    return nodes, adj
