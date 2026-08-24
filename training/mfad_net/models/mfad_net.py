"""
MFAD-Net — full assembled model (paper §5–§6).

INPUT:  frames (B,T,3,224,224) or (B,3,224,224), wav (B,L), metadata graph
OUTPUT: logits, P(fake), aux scores, fused embedding, optional drift info
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio_encoder import AudioEncoder
from .cmaf import CrossModalAttentionFusion
from .meta_encoder import MetadataEncoder, build_trivial_graph
from .tsdd import TemporalSemanticDriftDetector
from .visual_encoder import VisualEncoder


@dataclass
class MFADNetConfig:
    lite: bool = True
    pretrained: bool = False
    d_model: int = 512
    nhead: int = 4
    cmaf_layers: int = 2
    drift_threshold: float = 0.15
    window_size: int = 500
    mc_dropout_passes: int = 0  # set >0 at inference for CI
    num_classes: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MFADNetConfig":
        fields = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**fields)


class MFADNet(nn.Module):
    def __init__(self, config: Optional[MFADNetConfig] = None) -> None:
        super().__init__()
        self.config = config or MFADNetConfig()
        cfg = self.config

        self.visual = VisualEncoder(lite=cfg.lite, pretrained=cfg.pretrained)
        self.audio = AudioEncoder(lite=cfg.lite, pretrained=cfg.pretrained)
        self.meta = MetadataEncoder(out_dim=256)
        self.cmaf = CrossModalAttentionFusion(
            visual_dim=self.visual.out_dim,
            audio_dim=self.audio.out_dim,
            meta_dim=self.meta.out_dim,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.cmaf_layers,
        )
        self.tsdd = TemporalSemanticDriftDetector(
            dim=cfg.d_model,
            window_size=cfg.window_size,
            drift_threshold=cfg.drift_threshold,
        )
        # Classifier(F_fused + drift_signal)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(cfg.d_model + 1, cfg.d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(cfg.d_model // 2, cfg.num_classes),
        )

    def encode(
        self,
        frames: torch.Tensor,
        wav: torch.Tensor,
        meta_vec: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        f_visual = self.visual(frames)
        f_audio = self.audio(wav)
        if node_features is None or adj is None:
            node_features, adj = build_trivial_graph(meta_vec)
        f_meta = self.meta(node_features, adj)
        fusion = self.cmaf(f_visual, f_audio, f_meta)
        return {
            "f_visual": f_visual,
            "f_audio": f_audio,
            "f_meta": f_meta,
            **fusion,
        }

    def forward(
        self,
        frames: torch.Tensor,
        wav: torch.Tensor,
        meta_vec: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
        update_drift: bool = False,
    ) -> dict[str, torch.Tensor]:
        enc = self.encode(frames, wav, meta_vec, node_features, adj)
        fused = enc["fused"]
        drift_sig = self.tsdd.drift_signal(fused)
        logits = self.classifier(torch.cat([fused, drift_sig], dim=-1))
        probs = F.softmax(logits, dim=-1)
        p_fake = probs[:, 1]

        drift_info = None
        if update_drift:
            drift_info = self.tsdd.check_drift(fused)

        out = {
            "logits": logits,
            "probs": probs,
            "p_fake": p_fake,
            "fused": fused,
            "score_av": enc["score_av"],
            "score_meta": enc["score_meta"],
            "drift_signal": drift_sig,
        }
        if drift_info is not None:
            out["drift_flag"] = torch.tensor(float(drift_info["drift_flag"]), device=fused.device)
            out["mmd_score"] = torch.tensor(drift_info["mmd_score"], device=fused.device)
            out["tool_match"] = drift_info["tool_match"]
        return out

    @torch.no_grad()
    def predict(
        self,
        frames: torch.Tensor,
        wav: torch.Tensor,
        meta_vec: torch.Tensor,
        update_drift: bool = True,
    ) -> dict[str, Any]:
        self.eval()
        out = self.forward(frames, wav, meta_vec, update_drift=update_drift)
        p_fake = float(out["p_fake"][0].item())
        if p_fake > 0.75:
            verdict = "FAKE"
        elif p_fake < 0.35:
            verdict = "REAL"
        else:
            verdict = "UNCERTAIN"

        ci = None
        if self.config.mc_dropout_passes and self.config.mc_dropout_passes > 1:
            ci = self.monte_carlo_ci(frames, wav, meta_vec, passes=self.config.mc_dropout_passes)

        return {
            "verdict": verdict,
            "p_fake": p_fake,
            "confidence_interval": ci,
            "score_av": float(out["score_av"][0].item()),
            "score_meta": float(out["score_meta"][0].item()),
            "drift_flag": bool(out.get("drift_flag", torch.tensor(0.0)).item()) if "drift_flag" in out else False,
            "mmd_score": float(out["mmd_score"].item()) if "mmd_score" in out else 0.0,
            "tool_match": out.get("tool_match"),
            "fused": out["fused"].cpu(),
        }

    def monte_carlo_ci(
        self,
        frames: torch.Tensor,
        wav: torch.Tensor,
        meta_vec: torch.Tensor,
        passes: int = 50,
    ) -> tuple[float, float]:
        """MC dropout confidence interval over P(fake)."""
        was_training = self.training
        self.train()  # enable dropout
        samples = []
        with torch.no_grad():
            for _ in range(passes):
                out = self.forward(frames, wav, meta_vec, update_drift=False)
                samples.append(out["p_fake"].detach())
        if not was_training:
            self.eval()
        stacked = torch.cat(samples, dim=0)
        lo = float(stacked.quantile(0.025).item())
        hi = float(stacked.quantile(0.975).item())
        return lo, hi


def verdict_from_probability(p_fake: float) -> str:
    if p_fake > 0.75:
        return "FAKE"
    if p_fake < 0.35:
        return "REAL"
    return "UNCERTAIN"
