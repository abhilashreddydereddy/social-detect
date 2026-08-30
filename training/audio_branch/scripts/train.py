from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from training.audio_branch.models.audio_classifier import AudioClassifier, AudioClassifierConfig
from training.audio_branch.scripts.dataset import AudioManifestDataset, collate_audio


def auc(labels: list[int], probs: list[float]) -> float:
    pairs = sorted(zip(probs, labels), key=lambda pair: pair[0])
    positive = sum(label == 1 for _, label in pairs)
    negative = len(pairs) - positive
    if not positive or not negative:
        return 0.5
    rank_sum = sum(index for index, (_, label) in enumerate(pairs, 1) if label == 1)
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


@torch.no_grad()
def evaluate(model: AudioClassifier, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = total = 0
    labels: list[int] = []
    probs: list[float] = []
    loss_sum = 0.0
    for batch in loader:
        output = model(batch["waveform"].to(device))
        target = batch["label"].to(device)
        loss_sum += float(F.cross_entropy(output["logits"], target)) * target.size(0)
        correct += int((output["logits"].argmax(1) == target).sum())
        total += target.size(0)
        labels.extend(target.cpu().tolist())
        probs.extend(output["p_fake"].cpu().tolist())
    return {"loss": loss_sum / max(total, 1), "accuracy": correct / max(total, 1), "auc": auc(labels, probs)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ASVspoof 5 audio classifier")
    parser.add_argument("--config", default="training/audio_branch/configs/asvspoof5.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device(config.get("device", "cuda") if config.get("device", "cuda") != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    sample_rate = int(config.get("sample_rate", 16000))
    clip_seconds = float(config.get("clip_seconds", 4.0))
    train_ds = AudioManifestDataset(config["train_manifest"], sample_rate, clip_seconds, train=True)
    val_ds = AudioManifestDataset(config["val_manifest"], sample_rate, clip_seconds)
    train_loader = DataLoader(train_ds, batch_size=int(config.get("batch_size", 32)), shuffle=True, num_workers=int(config.get("num_workers", 0)), collate_fn=collate_audio, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=int(config.get("batch_size", 32)), shuffle=False, num_workers=int(config.get("num_workers", 0)), collate_fn=collate_audio, pin_memory=device.type == "cuda")
    model_config = AudioClassifierConfig(sample_rate=sample_rate, clip_seconds=clip_seconds, n_fft=int(config.get("n_fft", 512)), hop_length=int(config.get("hop_length", 160)), n_mels=int(config.get("n_mels", 80)))
    model = AudioClassifier(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-3)), weight_decay=float(config.get("weight_decay", 1e-4)))
    output_dir = Path(config.get("output_dir", "training/exports/audio_branch/asvspoof5"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / config.get("checkpoint_name", "asvspoof5_best.pth")
    best_auc = -1.0
    history = []
    print(f"Device: {device} | train={len(train_ds)} val={len(val_ds)}")
    for epoch in range(1, int(config.get("epochs", 15)) + 1):
        model.train()
        for batch in train_loader:
            output = model(batch["waveform"].to(device))
            loss = F.cross_entropy(output["logits"], batch["label"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, **metrics})
        print(f"epoch {epoch:02d} loss={metrics['loss']:.4f} acc={metrics['accuracy']:.3f} auc={metrics['auc']:.3f}")
        if metrics["auc"] >= best_auc:
            best_auc = metrics["auc"]
            torch.save({"model_config": model_config.to_dict(), "state_dict": model.state_dict(), "val_metrics": metrics, "dataset": "ASVspoof 5"}, checkpoint)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Done. Best AUC={best_auc:.3f} -> {checkpoint}")


if __name__ == "__main__":
    main()