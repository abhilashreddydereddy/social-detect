#!/usr/bin/env python3
"""Evaluate a trained MFAD-Net checkpoint on a manifest split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mfad_net.models.mfad_net import MFADNet, MFADNetConfig
from training.mfad_net.scripts.dataset import MFADMultimodalDataset, collate_mfad
from training.mfad_net.scripts.train import _auc, resolve_device


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/mfad_net/configs/smoke.yaml")
    parser.add_argument("--checkpoint", default="training/exports/mfad_net/mfad_net_best.pth")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = resolve_device(str(config.get("device", "auto")))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = MFADNet(MFADNetConfig.from_dict(ckpt.get("model_config", {}))).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    key = f"{args.split}_manifest" if f"{args.split}_manifest" in config["data"] else None
    manifest = config["data"].get(key) or config["data"].get(f"{args.split}_manifest") or config["data"]["test_manifest"]
    if args.split == "train":
        manifest = config["data"]["train_manifest"]
    elif args.split == "val":
        manifest = config["data"]["val_manifest"]
    else:
        manifest = config["data"]["test_manifest"]

    audio_samples = int(config.get("sample_rate", 16000) * float(config.get("audio_seconds", 1.0)))
    ds = MFADMultimodalDataset(
        manifest,
        frames_per_clip=int(config.get("frames_per_clip", 4)),
        image_size=int(config.get("image_size", 224)),
        audio_samples=audio_samples,
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_mfad)

    correct = 0
    total = 0
    probs, labels = [], []
    for batch in loader:
        out = model(batch["frames"].to(device), batch["wav"].to(device), batch["meta"].to(device))
        pred = out["logits"].argmax(dim=-1)
        y = batch["label"].to(device)
        correct += int((pred == y).sum().item())
        total += y.size(0)
        probs.extend(out["p_fake"].cpu().tolist())
        labels.extend(y.cpu().tolist())

    metrics = {
        "split": args.split,
        "n": total,
        "accuracy": correct / max(total, 1),
        "auc": _auc(labels, probs),
        "checkpoint": str(args.checkpoint),
    }
    print(json.dumps(metrics, indent=2))
    out_path = Path(config.get("output_dir", "training/exports/mfad_net")) / f"eval_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
