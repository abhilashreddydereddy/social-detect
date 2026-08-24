#!/usr/bin/env python3
"""Generate a synthetic trimodal dataset for MFAD-Net smoke training.

Real samples: natural-looking noise faces + harmonic speech-like audio + calm metadata.
Fake samples: periodic spectral grid (GAN-like) + flat vocoder-ish audio + anomalous metadata.

This does NOT replace FF++ / DFDC / WildDeepfake — it only validates the
architecture and produces a deployable lite checkpoint on CPU.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def _make_real_frame(rng: np.random.Generator, size: int = 224) -> np.ndarray:
    base = rng.integers(40, 200, (size, size, 3), dtype=np.uint8).astype(np.float32)
    # Soft spatial structure
    yy, xx = np.mgrid[0:size, 0:size]
    blob = np.exp(-((yy - size / 2) ** 2 + (xx - size / 2) ** 2) / (2 * (size / 4) ** 2))
    skin = np.stack([blob * 180, blob * 140, blob * 120], axis=-1)
    img = np.clip(0.55 * base + 0.45 * skin + rng.normal(0, 8, base.shape), 0, 255)
    return img.astype(np.uint8)


def _make_fake_frame(rng: np.random.Generator, size: int = 224) -> np.ndarray:
    img = _make_real_frame(rng, size).astype(np.float32)
    # Periodic grid artifact (frequency fingerprint)
    yy, xx = np.mgrid[0:size, 0:size]
    grid = (np.sin(xx / 3.0) * np.sin(yy / 3.0)) * 25.0
    img = np.clip(img + grid[..., None], 0, 255)
    # Over-smoothed patches
    img[40:120, 40:120] = img[40:120, 40:120] * 0.4 + 120
    return img.astype(np.uint8)


def _make_real_wav(rng: np.random.Generator, sr: int = 16000, seconds: float = 1.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    # Variable pitch + ambient floor
    f0 = 140 + 40 * np.sin(2 * np.pi * 2.5 * t)
    phase = np.cumsum(2 * np.pi * f0 / sr)
    tone = 0.35 * np.sin(phase)
    ambient = 0.03 * rng.standard_normal(t.shape[0])
    return np.clip(tone + ambient, -1, 1).astype(np.float32)


def _make_fake_wav(rng: np.random.Generator, sr: int = 16000, seconds: float = 1.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    # Unnaturally stable pitch + digital silence floor
    tone = 0.4 * np.sin(2 * np.pi * 180 * t)
    # Vocoder buzz
    buzz = 0.08 * np.sin(2 * np.pi * 3000 * t)
    return np.clip(tone + buzz, -1, 1).astype(np.float32)


def _make_meta(label: int, rng: np.random.Generator) -> np.ndarray:
    # 16-d vector: platform one-hot-ish + account stats
    v = rng.normal(0, 0.3, size=16).astype(np.float32)
    if label == 0:  # real
        v[0] = 1.0
        v[8] = abs(rng.normal(0.6, 0.1))   # account age
        v[9] = abs(rng.normal(0.4, 0.1))   # followers
        v[10] = abs(rng.normal(0.2, 0.05)) # sharing velocity
    else:
        v[1] = 1.0
        v[8] = abs(rng.normal(0.05, 0.02))
        v[9] = abs(rng.normal(0.9, 0.05))
        v[10] = abs(rng.normal(0.95, 0.05))  # viral burst
    return v


def generate_split(out_dir: Path, split: str, n: int, frames: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        label = int(i % 2)  # balanced
        sample_id = f"{split}_{i:05d}_y{label}"
        sample_dir = split_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        for fidx in range(frames):
            img = _make_fake_frame(rng) if label == 1 else _make_real_frame(rng)
            Image.fromarray(img).save(sample_dir / f"frame_{fidx:02d}.jpg", quality=90)

        wav = _make_fake_wav(rng) if label == 1 else _make_real_wav(rng)
        np.save(sample_dir / "audio.npy", wav)

        meta = _make_meta(label, rng)
        np.save(sample_dir / "meta.npy", meta)

        rows.append({
            "sample_id": sample_id,
            "path": str(sample_dir.as_posix()),
            "label": label,
            "source": "synthetic",
            "split": split,
        })

    csv_path = out_dir.parent / "splits" / f"{split}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "path", "label", "source", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="training/data/mfad_net/processed")
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-val", type=int, default=64)
    parser.add_argument("--n-test", type=int, default=64)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    generate_split(out, "train", args.n_train, args.frames, args.seed)
    generate_split(out, "val", args.n_val, args.frames, args.seed + 1)
    generate_split(out, "test", args.n_test, args.frames, args.seed + 2)
    print(f"Synthetic MFAD-Net dataset written under {out}")
    print("Manifests: training/data/mfad_net/splits/{train,val,test}.csv")


if __name__ == "__main__":
    main()
