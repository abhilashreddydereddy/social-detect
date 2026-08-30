"""
Inference copy of the image-branch classifier.

Keep the architecture in sync with
`training/image_branch/models/image_classifier.py`. Training writes the
checkpoint; this module loads it inside the API process without requiring
the `training` package on PYTHONPATH (Docker / `cd backend` uvicorn).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFTFrequencyBranch(nn.Module):
    """2D FFT magnitude → CNN → embedding (GAN / diffusion fingerprints)."""

    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        gray = images.mean(dim=1, keepdim=True)
        spec = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1))
        mag = torch.log1p(spec.abs())
        mag = mag / (mag.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        return self.net(mag)


class TinySpatialCNN(nn.Module):
    """CPU-friendly stand-in when timm EfficientNet is not desired."""

    def __init__(self, out_dim: int = 512) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.features(x))


@dataclass
class ImageClassifierConfig:
    backbone: str = "efficientnet_b0"  # efficientnet_b0 | efficientnet_b4 | tiny_cnn
    pretrained: bool = True
    use_fft: bool = True
    fft_dim: int = 256
    dropout: float = 0.2
    num_classes: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ImageClassifierConfig":
        data = data or {}
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


class ImageClassifier(nn.Module):
    def __init__(self, config: ImageClassifierConfig | None = None) -> None:
        super().__init__()
        self.config = config or ImageClassifierConfig()
        self.use_fft = bool(self.config.use_fft)

        backbone = self.config.backbone.lower()
        if backbone == "tiny_cnn":
            self.spatial = TinySpatialCNN(out_dim=512)
            spatial_dim = 512
            self.backbone_name = "tiny_cnn"
        else:
            import timm

            self.spatial = timm.create_model(
                backbone,
                pretrained=bool(self.config.pretrained),
                num_classes=0,
            )
            spatial_dim = int(getattr(self.spatial, "num_features", 1280))
            self.backbone_name = backbone

        fft_dim = int(self.config.fft_dim) if self.use_fft else 0
        self.freq = FFTFrequencyBranch(out_dim=fft_dim) if self.use_fft else None
        fused_dim = spatial_dim + fft_dim

        self.head = nn.Sequential(
            nn.Dropout(p=float(self.config.dropout)),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(self.config.dropout) * 0.5),
            nn.Linear(fused_dim // 2, int(self.config.num_classes)),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        spat = self.spatial(images)
        if self.freq is not None:
            return torch.cat([spat, self.freq(images)], dim=-1)
        return spat

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.encode(images)
        logits = self.head(feats)
        probs = F.softmax(logits, dim=-1)
        p_fake = probs[:, 1] if logits.size(-1) > 1 else probs[:, 0]
        return {
            "logits": logits,
            "probs": probs,
            "p_fake": p_fake,
            "features": feats,
        }
