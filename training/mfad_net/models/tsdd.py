"""
Temporal-Semantic Drift Detector (TSDD) — unique MFAD-Net module.

Maintains a rolling window of fused embeddings, computes MMD against a
prototype memory of known real/fake content, and maintains a GAN fingerprint
memory (cluster centroids) for tool attribution.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Gaussian RBF kernel matrix between x (N,D) and y (M,D)."""
    x_norm = (x * x).sum(dim=1, keepdim=True)
    y_norm = (y * y).sum(dim=1, keepdim=True).T
    dist = x_norm + y_norm - 2.0 * x @ y.T
    return torch.exp(-gamma * dist.clamp_min(0.0))


def mmd2(x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Unbiased-ish squared MMD with RBF kernel (paper §6.2)."""
    if x.size(0) < 2 or y.size(0) < 2:
        return torch.tensor(0.0, device=x.device)
    k_xx = rbf_kernel(x, x, gamma)
    k_yy = rbf_kernel(y, y, gamma)
    k_xy = rbf_kernel(x, y, gamma)
    n = x.size(0)
    m = y.size(0)
    # Exclude diagonal for unbiased estimate
    mmd = (k_xx.sum() - k_xx.diag().sum()) / (n * (n - 1))
    mmd = mmd + (k_yy.sum() - k_yy.diag().sum()) / (m * (m - 1))
    mmd = mmd - 2.0 * k_xy.mean()
    return mmd.clamp_min(0.0)


class TemporalSemanticDriftDetector(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        window_size: int = 500,
        drift_threshold: float = 0.15,
        n_prototypes: int = 64,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.gamma = gamma

        # Learnable prototype bank (initialized from data later)
        self.register_buffer("prototypes", torch.randn(n_prototypes, dim) * 0.01)
        self.register_buffer("prototype_initialized", torch.tensor(False))

        # GAN fingerprint memory: list of centroids (stored as buffer stack)
        self.register_buffer("gan_memory", torch.zeros(0, dim))
        self.register_buffer("gan_memory_labels", torch.zeros(0, dtype=torch.long))

        self._window: Deque[torch.Tensor] = deque(maxlen=window_size)

    @torch.no_grad()
    def update_window(self, fused: torch.Tensor) -> None:
        """Push batch embeddings into the rolling window (CPU/GPU agnostic)."""
        for row in fused.detach():
            self._window.append(row.cpu())

    @torch.no_grad()
    def initialize_prototypes(self, embeddings: torch.Tensor) -> None:
        """Seed prototype memory from a set of training embeddings."""
        if embeddings.size(0) == 0:
            return
        n = min(self.prototypes.size(0), embeddings.size(0))
        # K-means-ish: take random subset as centroids (simple, stable)
        idx = torch.randperm(embeddings.size(0))[:n]
        self.prototypes[:n] = embeddings[idx].detach()
        self.prototype_initialized.fill_(True)

    @torch.no_grad()
    def update_prototypes_from_window(self) -> None:
        if len(self._window) < 8:
            return
        stacked = torch.stack(list(self._window), dim=0)
        self.initialize_prototypes(stacked)

    def window_tensor(self, device=None) -> torch.Tensor:
        if not self._window:
            return torch.zeros(0, self.dim, device=device)
        t = torch.stack(list(self._window), dim=0)
        return t.to(device) if device is not None else t

    @torch.no_grad()
    def compute_mmd(self, device=None) -> float:
        window = self.window_tensor(device=device or self.prototypes.device)
        if window.size(0) < 8 or not bool(self.prototype_initialized):
            return 0.0
        score = mmd2(window, self.prototypes.to(window.device), gamma=self.gamma)
        return float(score.item())

    @torch.no_grad()
    def check_drift(self, fused_batch: torch.Tensor) -> dict:
        self.update_window(fused_batch)
        mmd_score = self.compute_mmd(device=fused_batch.device)
        drift_flag = mmd_score > self.drift_threshold
        tool_match = self.nearest_gan_tool(fused_batch.mean(dim=0, keepdim=True))
        result = {
            "mmd_score": mmd_score,
            "drift_flag": drift_flag,
            "tool_match": tool_match,
        }
        if drift_flag:
            # Cluster new artifacts into fingerprint memory
            centroid = fused_batch.mean(dim=0, keepdim=True).detach().cpu()
            self.append_gan_fingerprint(centroid, label_id=int(self.gan_memory.size(0)))
            self.update_prototypes_from_window()
        return result

    @torch.no_grad()
    def append_gan_fingerprint(self, centroid: torch.Tensor, label_id: int = 0) -> None:
        if centroid.dim() == 1:
            centroid = centroid.unsqueeze(0)
        self.gan_memory = torch.cat([self.gan_memory, centroid.cpu()], dim=0)
        label = torch.tensor([label_id], dtype=torch.long)
        self.gan_memory_labels = torch.cat([self.gan_memory_labels, label], dim=0)

    @torch.no_grad()
    def nearest_gan_tool(self, fused: torch.Tensor) -> Optional[int]:
        if self.gan_memory.numel() == 0:
            return None
        # fused: (1, D) or (D,)
        if fused.dim() == 1:
            fused = fused.unsqueeze(0)
        mem = self.gan_memory.to(fused.device)
        dists = torch.cdist(fused, mem)  # (1, K)
        idx = int(dists.argmin(dim=1).item())
        return int(self.gan_memory_labels[idx].item())

    def drift_signal(self, fused: torch.Tensor) -> torch.Tensor:
        """Scalar drift signal concatenated into classifier (paper step 5)."""
        # Non-differentiable MMD as a detached feature; keeps graph clean.
        with torch.no_grad():
            score = self.compute_mmd(device=fused.device)
        return torch.full((fused.size(0), 1), score, device=fused.device, dtype=fused.dtype)
