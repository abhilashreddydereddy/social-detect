#!/usr/bin/env python3
"""
Train the image-branch classifier (CIFake and similar image-only datasets).

Usage (from repo root):
  python -m training.image_branch.scripts.prepare_manifest ...
  python -m training.image_branch.scripts.train --config training/image_branch/configs/cifake.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.image_branch.models.image_classifier import ImageClassifier, ImageClassifierConfig
from training.image_branch.scripts.dataset import (
    ImageManifestDataset,
    build_eval_transform,
    build_train_transform,
    collate_images,
)


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


def _f1(labels: list[int], preds: list[int]) -> float:
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


@torch.no_grad()
def evaluate(
    model: ImageClassifier,
    loader: DataLoader,
    device: torch.device,
    collect_paths: bool = False,
) -> tuple[dict[str, float], list[dict[str, object]] | None]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs: list[float] = []
    all_labels: list[int] = []
    all_preds: list[int] = []
    rows: list[dict[str, object]] = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        out = model(images)
        loss = F.cross_entropy(out["logits"], labels)
        total_loss += float(loss.item()) * labels.size(0)
        preds = out["logits"].argmax(dim=-1)
        correct += int((preds == labels).sum().item())
        total += labels.size(0)

        probs = out["p_fake"].detach().cpu().tolist()
        lab_list = labels.detach().cpu().tolist()
        pred_list = preds.detach().cpu().tolist()
        all_probs.extend(probs)
        all_labels.extend(lab_list)
        all_preds.extend(pred_list)

        if collect_paths:
            for path, y, p, pred in zip(batch["path"], lab_list, probs, pred_list):
                rows.append({"path": path, "label": y, "p_fake": p, "pred": pred})

    metrics = {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "auc": _auc(all_labels, all_probs),
        "f1": _f1(all_labels, all_preds),
    }
    return metrics, (rows if collect_paths else None)


def train_one_epoch(
    model: ImageClassifier,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running = 0.0
    n = 0
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        out = model(images)
        loss = F.cross_entropy(out["logits"], labels)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()
        running += float(loss.item()) * labels.size(0)
        n += labels.size(0)
    return running / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train image-branch classifier (CIFake)")
    parser.add_argument("--config", default="training/image_branch/configs/cifake.yaml")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate an existing checkpoint and exit")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "auto")))
    print(f"Device: {device}")

    image_size = int(config.get("image_size", 224))
    path_col = str(config.get("path_column", "path"))
    label_col = str(config.get("label_column", "label"))
    jpeg_aug = bool(config.get("jpeg_aug", True))
    filter_aug = bool(config.get("filter_aug", True))

    train_ds = ImageManifestDataset(
        config["train_manifest"],
        path_column=path_col,
        label_column=label_col,
        transform=build_train_transform(
            image_size,
            jpeg_aug=jpeg_aug,
            filter_aug=filter_aug,
        ),
    )
    val_ds = ImageManifestDataset(
        config["val_manifest"],
        path_column=path_col,
        label_column=label_col,
        transform=build_eval_transform(image_size),
    )

    num_workers = int(config.get("num_workers", 0))
    pin_memory = device.type == "cuda"
    loader_kwargs: dict = {
        "batch_size": int(config.get("batch_size", 32)),
        "num_workers": num_workers,
        "collate_fn": collate_images,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 2))

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model_cfg = ImageClassifierConfig(
        backbone=str(config.get("backbone", "efficientnet_b0")),
        pretrained=bool(config.get("pretrained", True)),
        use_fft=bool(config.get("use_fft", True)),
        fft_dim=int(config.get("fft_dim", 256)),
        dropout=float(config.get("dropout", 0.2)),
        num_classes=2,
    )
    model = ImageClassifier(model_cfg).to(device)

    out_dir = Path(config.get("output_dir", "training/exports/image_branch/cifake"))
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / config.get("checkpoint_name", "best_model.pth")

    if args.eval_only:
        if not best_path.exists():
            raise SystemExit(f"Checkpoint not found: {best_path}")
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        metrics, rows = evaluate(model, val_loader, device, collect_paths=True)
        print(f"val  acc={metrics['acc']:.4f}  auc={metrics['auc']:.4f}  f1={metrics['f1']:.4f}")
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / "val_predictions.csv", index=False)
        return

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", config.get("lr", 1e-4))),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    epochs = int(config.get("epochs", 12))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(epochs, 1))

    patience = int(config.get("early_stopping_patience", 4))
    history: list[dict] = []
    best_auc = -1.0
    stale = 0

    print(f"Train rows: {len(train_ds)}  Val rows: {len(val_ds)}")
    print(f"Backbone: {model_cfg.backbone}  use_fft={model_cfg.use_fft}  pretrained={model_cfg.pretrained}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optim, device)
        metrics, _ = evaluate(model, val_loader, device, collect_paths=False)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, **metrics, "lr": float(scheduler.get_last_lr()[0])}
        history.append(row)
        print(
            f"epoch {epoch:02d}  train_loss={train_loss:.4f}  "
            f"val_loss={metrics['loss']:.4f}  acc={metrics['acc']:.3f}  "
            f"auc={metrics['auc']:.3f}  f1={metrics['f1']:.3f}"
        )

        if metrics["auc"] >= best_auc:
            best_auc = metrics["auc"]
            stale = 0
            payload = {
                "model_config": model_cfg.to_dict(),
                "state_dict": model.state_dict(),
                "val_metrics": metrics,
                "epoch": epoch,
                "config_name": config.get("name", "image-branch"),
                "image_size": image_size,
            }
            torch.save(payload, best_path)
            print(f"  saved {best_path}")
            # Refresh val predictions for best checkpoint
            _, rows = evaluate(model, val_loader, device, collect_paths=True)
            if rows:
                pd.DataFrame(rows).to_csv(out_dir / "val_predictions.csv", index=False)
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Done. Best AUC={best_auc:.3f} -> {best_path}")
    print("Copy to backend when ready:")
    print(f"  cp {best_path} backend/models/image_branch/cifake_best.pth")


if __name__ == "__main__":
    main()
