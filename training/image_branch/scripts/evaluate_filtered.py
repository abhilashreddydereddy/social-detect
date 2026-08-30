#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.image_branch.models.image_classifier import ImageClassifier, ImageClassifierConfig
from training.image_branch.scripts.dataset import _load_rgb, _to_tensor_normalized, collate_images
from training.image_branch.scripts.train import _auc, _f1, resolve_device, set_seed


class DeterministicFilteredTransform:
    def __init__(self, image_size: int, filter_name: str) -> None:
        self.image_size = image_size
        self.filter_name = filter_name

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
        name = self.filter_name
        if name == "warm":
            arr = np.asarray(img).astype(np.float32)
            arr[..., 0] *= 1.10
            arr[..., 2] *= 0.92
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        elif name == "cool":
            arr = np.asarray(img).astype(np.float32)
            arr[..., 0] *= 0.92
            arr[..., 2] *= 1.10
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        elif name == "blur":
            img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
        elif name == "sharp":
            img = ImageEnhance.Sharpness(img).enhance(1.7)
        elif name == "mono":
            img = ImageEnhance.Color(img).enhance(0.05)
        elif name == "jpeg":
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=55)
            buf.seek(0)
            img = Image.open(buf).convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
        else:
            img = ImageEnhance.Color(img).enhance(1.25)
        return _to_tensor_normalized(img)


class FilteredManifestDataset(Dataset):
    def __init__(self, manifest: str | Path, image_size: int, filter_name: str) -> None:
        self.df = pd.read_csv(manifest)
        self.transform = DeterministicFilteredTransform(image_size=image_size, filter_name=filter_name)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.df.iloc[idx]
        path = Path(str(row["path"]))
        label = int(row["label"])
        img = _load_rgb(path)
        tensor = self.transform(img)
        return {
            "image": tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(path),
        }


@torch.no_grad()
def evaluate(model: ImageClassifier, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    probs: list[float] = []
    labels: list[int] = []
    preds: list[int] = []
    rows: list[dict[str, object]] = []
    for batch in loader:
        images = batch["image"].to(device)
        y = batch["label"].to(device)
        out = model(images)
        loss = F.cross_entropy(out["logits"], y)
        total_loss += float(loss.item()) * y.size(0)
        pred = out["logits"].argmax(dim=-1)
        prob_list = out["p_fake"].detach().cpu().tolist()
        label_list = y.detach().cpu().tolist()
        pred_list = pred.detach().cpu().tolist()
        correct += int((pred == y).sum().item())
        total += y.size(0)
        probs.extend(prob_list)
        labels.extend(label_list)
        preds.extend(pred_list)
        for path, label, prob, pred_label in zip(batch["path"], label_list, prob_list, pred_list):
            rows.append({"path": path, "label": label, "p_fake": prob, "pred": pred_label})
    metrics = {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "auc": _auc(labels, probs),
        "f1": _f1(labels, preds),
        "n": total,
    }
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    metrics["precision"] = tp / max(tp + fp, 1)
    metrics["recall"] = tp / max(tp + fn, 1)
    metrics["confusion_matrix"] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    return metrics, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CIFake image branch on a deterministic filtered split")
    parser.add_argument("--config", default="training/image_branch/configs/cifake.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--filter", default="jpeg", choices=["color", "warm", "cool", "blur", "sharp", "mono", "jpeg"])
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "auto")))
    image_size = int(config.get("image_size", 224))
    manifest = args.manifest or config["val_manifest"]
    checkpoint = args.checkpoint or str(Path(config.get("output_dir", "training/exports/image_branch/cifake")) / config.get("checkpoint_name", "best_model.pth"))

    ds = FilteredManifestDataset(manifest, image_size=image_size, filter_name=args.filter)
    loader = DataLoader(
        ds,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=collate_images,
        pin_memory=device.type == "cuda",
    )

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = ImageClassifier(ImageClassifierConfig.from_dict(ckpt.get("model_config", {}))).to(device)
    model.load_state_dict(ckpt["state_dict"])

    metrics, rows = evaluate(model, loader, device)
    output_dir = Path(config.get("output_dir", "training/exports/image_branch/cifake"))
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_name or f"filtered_{args.filter}"
    (output_dir / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / f"{stem}_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "p_fake", "pred"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
