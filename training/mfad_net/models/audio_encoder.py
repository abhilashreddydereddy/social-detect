"""
Audio encoder — Wav2Vec 2.0 voice embeddings + MFCC spectral features.

Paper dims:
  F_voice = Wav2Vec2(wav)     # 768-d
  F_mfcc  = MFCC(wav, 40)     # 40-d
  F_audio = concat(...)       # 808-d

`lite=True` replaces Wav2Vec2 with a small 1D CNN over the waveform / MFCC
stack so training works offline without downloading Facebook weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_mfcc(wav: torch.Tensor, sample_rate: int = 16000, n_mfcc: int = 40) -> torch.Tensor:
    """Differentiable-enough MFCC approximation via log-mel filterbank + DCT.

    wav: (B, T) float mono at `sample_rate`.
    Returns (B, n_mfcc) mean-pooled coefficients.
    """
    # Short-time Fourier
    n_fft = 512
    hop = 160
    window = torch.hann_window(n_fft, device=wav.device, dtype=wav.dtype)
    # stft → (B, freq, time)
    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
    power = spec.abs().pow(2).clamp_min(1e-10)

    n_mels = 64
    # Triangular mel filterbank (static, rebuilt on device)
    mel_f = _mel_filterbank(n_fft, n_mels, sample_rate, device=wav.device, dtype=wav.dtype)
    mel = torch.matmul(mel_f, power)  # (B, n_mels, time)
    log_mel = torch.log(mel.clamp_min(1e-10))

    # DCT-II → MFCC
    mfcc = _dct(log_mel, n_mfcc)  # (B, n_mfcc, time)
    return mfcc.mean(dim=-1)


def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_fft: int, n_mels: int, sample_rate: int, device, dtype) -> torch.Tensor:
    f_max = sample_rate / 2.0
    m_min, m_max = _hz_to_mel(torch.tensor(0.0)), _hz_to_mel(torch.tensor(f_max))
    m_pts = torch.linspace(m_min.item(), m_max.item(), n_mels + 2, device=device, dtype=dtype)
    hz_pts = _mel_to_hz(m_pts)
    bins = torch.floor((n_fft + 1) * hz_pts / sample_rate).long()
    fb = torch.zeros(n_mels, n_fft // 2 + 1, device=device, dtype=dtype)
    for i in range(n_mels):
        left, center, right = bins[i].item(), bins[i + 1].item(), bins[i + 2].item()
        if center == left:
            center += 1
        if right == center:
            right += 1
        for j in range(left, center):
            if 0 <= j < fb.shape[1]:
                fb[i, j] = (j - left) / max(center - left, 1)
        for j in range(center, right):
            if 0 <= j < fb.shape[1]:
                fb[i, j] = (right - j) / max(right - center, 1)
    return fb


def _dct(x: torch.Tensor, n_mfcc: int) -> torch.Tensor:
    # x: (B, n_mels, time)
    b, n_mels, t = x.shape
    n = torch.arange(n_mels, device=x.device, dtype=x.dtype)
    k = torch.arange(n_mfcc, device=x.device, dtype=x.dtype).unsqueeze(1)
    basis = torch.cos(torch.pi / n_mels * (n + 0.5) * k)  # (n_mfcc, n_mels)
    return torch.matmul(basis, x)


class TinyVoiceEncoder(nn.Module):
    """1D CNN stand-in for Wav2Vec2 → 768-d."""

    def __init__(self, out_dim: int = 768) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=10, stride=5, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        x = wav.unsqueeze(1)  # (B, 1, T)
        return self.net(x)


class AudioEncoder(nn.Module):
    def __init__(self, lite: bool = True, sample_rate: int = 16000, pretrained: bool = False) -> None:
        super().__init__()
        self.lite = lite
        self.sample_rate = sample_rate
        self.out_dim = 808  # 768 + 40

        if lite:
            self.voice = TinyVoiceEncoder(768)
            self.voice_name = "tiny_1d_cnn"
        else:
            from transformers import Wav2Vec2Model

            self.voice = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            self.voice_name = "wav2vec2-base"
            # Freeze feature extractor early; fine-tune later if desired
            if not pretrained:
                pass

        self.mfcc_proj = nn.Identity()

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: (B, T) mono float32 at 16 kHz.
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        if self.lite:
            voice = self.voice(wav)
        else:
            out = self.voice(wav)
            # mean pool last hidden state
            voice = out.last_hidden_state.mean(dim=1)  # (B, 768)

        mfcc = compute_mfcc(wav, sample_rate=self.sample_rate, n_mfcc=40)
        return torch.cat([voice, mfcc], dim=-1)
