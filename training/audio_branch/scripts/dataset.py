from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset


class AudioManifestDataset(Dataset):
    def __init__(self, manifest: str | Path, sample_rate: int = 16000, clip_seconds: float = 4.0, train: bool = False) -> None:
        self.df = pd.read_csv(manifest)
        if self.df.empty or not {"path", "label"}.issubset(self.df.columns):
            raise ValueError(f"Manifest must contain path and label rows: {manifest}")
        self.samples = int(sample_rate * clip_seconds)
        self.sample_rate = sample_rate
        self.train = train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        waveform, rate = sf.read(str(row["path"]), dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        waveform = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
        if rate != self.sample_rate:
            import scipy.signal
            length = round(waveform.numel() * self.sample_rate / rate)
            waveform = torch.from_numpy(scipy.signal.resample(waveform.numpy(), length).astype(np.float32))
        if waveform.numel() >= self.samples:
            start = random.randint(0, waveform.numel() - self.samples) if self.train else (waveform.numel() - self.samples) // 2
            waveform = waveform[start:start + self.samples]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, self.samples - waveform.numel()))
        if self.train:
            if random.random() < 0.5:
                waveform = waveform * random.uniform(0.75, 1.25)
            if random.random() < 0.35:
                noise = torch.randn_like(waveform) * random.uniform(0.001, 0.01)
                waveform = waveform + noise
        peak = waveform.abs().max().clamp_min(1e-5)
        return {"waveform": (waveform / peak).clamp(-1, 1), "label": torch.tensor(int(row["label"]), dtype=torch.long), "path": str(row["path"])}


def collate_audio(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {"waveform": torch.stack([item["waveform"] for item in batch]), "label": torch.stack([item["label"] for item in batch]), "path": [item["path"] for item in batch]}