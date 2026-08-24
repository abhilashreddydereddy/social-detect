"""Dataset + collate for MFAD-Net trimodal samples."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class MFADMultimodalDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        frames_per_clip: int = 4,
        image_size: int = 224,
        audio_samples: int = 16000,
    ) -> None:
        self.df = pd.read_csv(manifest)
        self.frames_per_clip = frames_per_clip
        self.image_size = image_size
        self.audio_samples = audio_samples

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        sample_dir = Path(row["path"])
        label = int(row["label"])

        frame_paths = sorted(sample_dir.glob("frame_*.jpg"))
        if not frame_paths:
            raise FileNotFoundError(f"No frames in {sample_dir}")

        # Uniform sample / pad
        if len(frame_paths) >= self.frames_per_clip:
            idxs = np.linspace(0, len(frame_paths) - 1, self.frames_per_clip).astype(int)
            chosen = [frame_paths[i] for i in idxs]
        else:
            chosen = list(frame_paths) + [frame_paths[-1]] * (self.frames_per_clip - len(frame_paths))

        frames = []
        for p in chosen:
            img = Image.open(p).convert("RGB").resize((self.image_size, self.image_size))
            arr = np.asarray(img).astype(np.float32) / 255.0
            # ImageNet-ish normalize
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arr = (arr - mean) / std
            frames.append(torch.from_numpy(arr).permute(2, 0, 1))
        frames_t = torch.stack(frames, dim=0)  # (T,3,H,W)

        wav = np.load(sample_dir / "audio.npy").astype(np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        if len(wav) < self.audio_samples:
            wav = np.pad(wav, (0, self.audio_samples - len(wav)))
        else:
            wav = wav[: self.audio_samples]
        wav_t = torch.from_numpy(wav)

        meta = np.load(sample_dir / "meta.npy").astype(np.float32)
        if meta.shape[0] < 16:
            meta = np.pad(meta, (0, 16 - meta.shape[0]))
        meta_t = torch.from_numpy(meta[:16])

        return {
            "frames": frames_t,
            "wav": wav_t,
            "meta": meta_t,
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": row["sample_id"],
        }


def collate_mfad(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "frames": torch.stack([b["frames"] for b in batch], dim=0),
        "wav": torch.stack([b["wav"] for b in batch], dim=0),
        "meta": torch.stack([b["meta"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "sample_id": [b["sample_id"] for b in batch],
    }
