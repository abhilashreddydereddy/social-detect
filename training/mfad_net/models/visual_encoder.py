"""
Visual encoder — EfficientNet spatial backbone + parallel FFT frequency branch.

Paper dims:
  F_spatial = EfficientNet-B4(frames)   # 1280-d (B4 num_features)
  F_freq    = FFT_branch(frames)       # 512-d
  F_visual  = concat(...)              # 1792-d

`lite=True` swaps EfficientNet-B4 for a tiny CNN so CPU smoke training works
without downloading ImageNet weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFTFrequencyBranch(nn.Module):
    """2D FFT magnitude → CNN → 512-d spectral embedding (GAN fingerprints)."""

    def __init__(self, out_dim: int = 512) -> None:
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

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, C, H, W) in [0, 1] or ImageNet-normalized
        gray = frames.mean(dim=1, keepdim=True)
        # Centered FFT magnitude (log1p for dynamic range)
        spec = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1))
        mag = torch.log1p(spec.abs())
        # Per-sample normalize
        mag = mag / (mag.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        return self.net(mag)


class TinySpatialCNN(nn.Module):
    """CPU-friendly stand-in for EfficientNet-B4 (outputs 1280-d)."""

    def __init__(self, out_dim: int = 1280) -> None:
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


class VisualEncoder(nn.Module):
    def __init__(self, lite: bool = True, pretrained: bool = False) -> None:
        super().__init__()
        self.lite = lite
        self.freq = FFTFrequencyBranch(out_dim=512)

        if lite:
            self.spatial = TinySpatialCNN(out_dim=1280)
            self.backbone_name = "tiny_cnn"
        else:
            import timm

            self.spatial = timm.create_model(
                "efficientnet_b4",
                pretrained=pretrained,
                num_classes=0,  # feature extractor → 1792 for B4? Actually B4 is 1792 features
            )
            # EfficientNet-B4 num_features is 1792 in timm; paper says 1280.
            # Project to 1280 to match paper concat → 1792 visual.
            feat_dim = getattr(self.spatial, "num_features", 1792)
            self.spatial_proj = nn.Identity() if feat_dim == 1280 else nn.Linear(feat_dim, 1280)
            self.backbone_name = "efficientnet_b4"

        self.out_dim = 1792  # 1280 + 512

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (B, 3, 224, 224) or (B, T, 3, 224, 224).
        For multi-frame, mean-pool temporal axis after encoding each frame.
        """
        if frames.dim() == 5:
            b, t, c, h, w = frames.shape
            flat = frames.reshape(b * t, c, h, w)
            spat = self._spatial(flat).reshape(b, t, -1).mean(dim=1)
            freq = self.freq(flat).reshape(b, t, -1).mean(dim=1)
        else:
            spat = self._spatial(frames)
            freq = self.freq(frames)
        return torch.cat([spat, freq], dim=-1)

    def _spatial(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.spatial(x)
        if not self.lite:
            feats = self.spatial_proj(feats)
        return feats
