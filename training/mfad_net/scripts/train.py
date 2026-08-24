#!/usr/bin/env python3
"""
MFAD-Net training loop (paper Phase 4).

Usage:
  python -m training.mfad_net.scripts.prepare_synthetic
  python -m training.mfad_net.scripts.train --config training/mfad_net/configs/smoke.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow `python training/mfad_net/scripts/train.py` from repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mfad_net.models.mfad_net import MFADNet, MFADNetConfig
from training.mfad_net.scripts.dataset import MFADMultimodalDataset, collate_mfad


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def ensure_synthetic(config: dict) -> None:
    syn = config.get("synthetic") or {}
    if not syn.get("enabled", False):
        return

    processed = Path(config["data"]["processed_dir"])
    train_m = Path(config["data"]["train_manifest"])
    # CSVs may be committed while processed crops are gitignored — regenerate
    # whenever the first train sample directory has no frames.
    needs_build = not train_m.exists()
    if not needs_build:
        try:
            import pandas as pd

            df = pd.read_csv(train_m)
            if df.empty:
                needs_build = True
            else:
                sample_dir = Path(df.iloc[0]["path"])
                needs_build = not sample_dir.exists() or not any(sample_dir.glob("frame_*.jpg"))
        except Exception:
            needs_build = True

    if not needs_build:
        return

    from training.mfad_net.scripts.prepare_synthetic import generate_split

    print(f"Generating synthetic MFAD-Net dataset under {processed} …")
    frames = int(config.get("frames_per_clip", 4))
    seed = int(config.get("seed", 42))
    generate_split(processed, "train", int(syn.get("n_train", 256)), frames, seed)
    generate_split(processed, "val", int(syn.get("n_val", 64)), frames, seed + 1)
    generate_split(processed, "test", int(syn.get("n_test", 64)), frames, seed + 2)


@torch.no_grad()
def evaluate(model: MFADNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    for batch in loader:
        frames = batch["frames"].to(device)
        wav = batch["wav"].to(device)
        meta = batch["meta"].to(device)
        labels = batch["label"].to(device)
        out = model(frames, wav, meta, update_drift=False)
        loss = F.cross_entropy(out["logits"], labels)
        total_loss += float(loss.item()) * labels.size(0)
        preds = out["logits"].argmax(dim=-1)
        correct += int((preds == labels).sum().item())
        total += labels.size(0)
        all_probs.extend(out["p_fake"].detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    # Simple AUC via Mann-Whitney
    auc = _auc(all_labels, all_probs)
    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "auc": auc,
    }


def _auc(labels: list[int], probs: list[float]) -> float:
    pairs = sorted(zip(probs, labels), key=lambda x: x[0])
    pos = sum(1 for _, y in pairs if y == 1)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return 0.5
    rank_sum = 0.0
    for i, (_, y) in enumerate(pairs, start=1):
        if y == 1:
            rank_sum += i
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def train_one_epoch(
    model: MFADNet,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    aux_av: float,
    aux_meta: float,
) -> float:
    model.train()
    running = 0.0
    n = 0
    for batch in tqdm(loader, desc="train", leave=False):
        frames = batch["frames"].to(device)
        wav = batch["wav"].to(device)
        meta = batch["meta"].to(device)
        labels = batch["label"].to(device)

        out = model(frames, wav, meta, update_drift=False)
        loss = F.cross_entropy(out["logits"], labels)
        # Aux: high AV desync / meta anomaly should correlate with fake
        target = labels.float()
        loss = loss + aux_av * F.binary_cross_entropy(out["score_av"], target)
        loss = loss + aux_meta * F.binary_cross_entropy(out["score_meta"], target)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()

        running += float(loss.item()) * labels.size(0)
        n += labels.size(0)
    return running / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MFAD-Net")
    parser.add_argument("--config", default="training/mfad_net/configs/smoke.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(int(config.get("seed", 42)))
    ensure_synthetic(config)

    device = resolve_device(str(config.get("device", "auto")))
    print(f"Device: {device}")

    audio_samples = int(config.get("sample_rate", 16000) * float(config.get("audio_seconds", 1.0)))
    train_ds = MFADMultimodalDataset(
        config["data"]["train_manifest"],
        frames_per_clip=int(config.get("frames_per_clip", 4)),
        image_size=int(config.get("image_size", 224)),
        audio_samples=audio_samples,
    )
    val_ds = MFADMultimodalDataset(
        config["data"]["val_manifest"],
        frames_per_clip=int(config.get("frames_per_clip", 4)),
        image_size=int(config.get("image_size", 224)),
        audio_samples=audio_samples,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=collate_mfad,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=collate_mfad,
    )

    model_cfg = MFADNetConfig(
        lite=bool(config.get("lite", True)),
        pretrained=bool(config.get("pretrained", False)),
        drift_threshold=float(config.get("drift_threshold", 0.15)),
        window_size=int(config.get("window_size", 500)),
    )
    model = MFADNet(model_cfg).to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )

    out_dir = Path(config.get("output_dir", "training/exports/mfad_net"))
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / config.get("checkpoint_name", "mfad_net_best.pth")
    history = []
    best_auc = -1.0

    epochs = int(config.get("epochs", 5))
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optim,
            device,
            aux_av=float(config.get("aux_av_weight", 0.1)),
            aux_meta=float(config.get("aux_meta_weight", 0.05)),
        )
        metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(row)
        print(
            f"epoch {epoch:02d}  train_loss={train_loss:.4f}  "
            f"val_loss={metrics['loss']:.4f}  acc={metrics['acc']:.3f}  auc={metrics['auc']:.3f}"
        )

        if metrics["auc"] >= best_auc:
            best_auc = metrics["auc"]
            # Seed TSDD prototypes from a validation batch
            with torch.no_grad():
                batch = next(iter(val_loader))
                out = model(
                    batch["frames"].to(device),
                    batch["wav"].to(device),
                    batch["meta"].to(device),
                    update_drift=False,
                )
                model.tsdd.initialize_prototypes(out["fused"].detach().cpu())

            payload = {
                "model_config": model_cfg.to_dict(),
                "state_dict": model.state_dict(),
                "val_metrics": metrics,
                "epoch": epoch,
                "paper": "MFAD-Net",
            }
            torch.save(payload, best_path)
            print(f"  saved {best_path}")

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Done. Best AUC={best_auc:.3f} → {best_path}")


if __name__ == "__main__":
    main()
