"""
Cross-Modal Attention Fusion (CMAF).

Projects visual (1792), audio (808), metadata (256) → 512-d each, then a
shared 4-head Transformer encoder over the 3 modality tokens.

Auxiliary heads:
  - audio-visual sync score (lip-sync inconsistency)
  - metadata anomaly score
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalAttentionFusion(nn.Module):
    def __init__(
        self,
        visual_dim: int = 1792,
        audio_dim: int = 808,
        meta_dim: int = 256,
        d_model: int = 512,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.proj_v = nn.Sequential(nn.Linear(visual_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.proj_a = nn.Sequential(nn.Linear(audio_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.proj_m = nn.Sequential(nn.Linear(meta_dim, d_model), nn.LayerNorm(d_model), nn.GELU())

        self.modality_embed = nn.Embedding(3, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.av_sync_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.meta_anomaly_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        f_visual: torch.Tensor,
        f_audio: torch.Tensor,
        f_meta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        v = self.proj_v(f_visual)
        a = self.proj_a(f_audio)
        m = self.proj_m(f_meta)

        tokens = torch.stack([v, a, m], dim=1)  # (B, 3, 512)
        ids = torch.arange(3, device=tokens.device).unsqueeze(0).expand(tokens.size(0), -1)
        tokens = tokens + self.modality_embed(ids)

        fused_tokens = self.transformer(tokens)  # (B, 3, 512)
        f_fused = fused_tokens.mean(dim=1)  # (B, 512)

        score_av = self.av_sync_head(torch.cat([v, a], dim=-1)).squeeze(-1)
        score_meta = self.meta_anomaly_head(m).squeeze(-1)

        return {
            "fused": f_fused,
            "tokens": fused_tokens,
            "v_proj": v,
            "a_proj": a,
            "m_proj": m,
            "score_av": score_av,
            "score_meta": score_meta,
        }
