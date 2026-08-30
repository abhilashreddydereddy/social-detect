from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AudioClassifierConfig:
    sample_rate: int = 16000
    clip_seconds: float = 4.0
    n_fft: int = 512
    hop_length: int = 160
    n_mels: int = 80
    dropout: float = 0.25
    num_classes: int = 2

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AudioClassifierConfig":
        data = data or {}
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


class AudioClassifier(nn.Module):
    def __init__(self, config: AudioClassifierConfig | None = None) -> None:
        super().__init__()
        self.config = config or AudioClassifierConfig()
        self.register_buffer("window", torch.hann_window(self.config.n_fft), persistent=False)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(self.config.dropout), nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(self.config.dropout), nn.Linear(64, self.config.num_classes))

    def _log_mel(self, wav: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(wav, n_fft=self.config.n_fft, hop_length=self.config.hop_length, win_length=self.config.n_fft, window=self.window, return_complex=True, center=True).abs().pow(2)
        freqs = torch.linspace(0, self.config.sample_rate / 2, spec.size(1), device=wav.device)
        mel_points = torch.linspace(1127.0 * torch.log1p(torch.tensor(20.0 / 700.0)), 1127.0 * torch.log1p(torch.tensor((self.config.sample_rate / 2) / 700.0)), self.config.n_mels + 2, device=wav.device)
        hz_points = 700.0 * torch.expm1(mel_points / 1127.0)
        filters = []
        for i in range(self.config.n_mels):
            left, center, right = hz_points[i:i + 3]
            filters.append(torch.clamp(torch.minimum((freqs - left) / (center - left + 1e-6), (right - freqs) / (right - center + 1e-6)), min=0.0))
        return torch.log(torch.stack(filters).to(spec.dtype) @ spec + 1e-6).unsqueeze(1)

    def forward(self, wav: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.head(self.features(self._log_mel(wav)))
        probs = F.softmax(logits, dim=-1)
        return {"logits": logits, "probs": probs, "p_fake": probs[:, 1]}