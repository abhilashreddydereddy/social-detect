#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "Paper"
FIG_DIR = PAPER_DIR / "figures"


def read_probs(csv_path: Path) -> tuple[list[int], list[float]]:
    labels: list[int] = []
    probs: list[float] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row["label"]))
            probs.append(float(row["p_fake"]))
    return labels, probs


def roc_points(labels: list[int], probs: list[float]) -> list[tuple[float, float]]:
    thresholds = sorted(set(probs), reverse=True)
    thresholds = [1.01] + thresholds + [-0.01]
    pos = sum(labels)
    neg = len(labels) - pos
    points: list[tuple[float, float]] = []
    for t in thresholds:
        tp = fp = 0
        for y, p in zip(labels, probs):
            pred = 1 if p >= t else 0
            if pred == 1 and y == 1:
                tp += 1
            elif pred == 1 and y == 0:
                fp += 1
        tpr = tp / pos if pos else 0.0
        fpr = fp / neg if neg else 0.0
        points.append((fpr, tpr))
    points.sort()
    return points


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def new_canvas(width: int = 1200, height: int = 760) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(img)
    return img, draw


def font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def draw_roc() -> Path:
    cifake_labels, cifake_probs = read_probs(ROOT / "training/exports/image_branch/cifake/val_predictions.csv")
    mfad_labels, mfad_probs = read_probs(ROOT / "training/exports/mfad_net/faceforensics/eval_test_predictions.csv")
    cifake_metrics = load_metrics(ROOT / "training/exports/image_branch/cifake/history.json")[-1]
    mfad_metrics = load_metrics(ROOT / "training/exports/mfad_net/faceforensics/eval_test.json")
    cifake_pts = roc_points(cifake_labels, cifake_probs)
    mfad_pts = roc_points(mfad_labels, mfad_probs)

    img, draw = new_canvas()
    title_font = font(28, bold=True)
    label_font = font(20)
    small_font = font(18)

    draw.text((70, 40), "ROC Comparison", fill="#111111", font=title_font)
    left, top, right, bottom = 110, 120, 1020, 650
    draw.rectangle((left, top, right, bottom), outline="#222222", width=2)

    for i in range(6):
        x = left + (right - left) * i / 5
        y = bottom - (bottom - top) * i / 5
        draw.line((x, top, x, bottom), fill="#dddddd", width=1)
        draw.line((left, y, right, y), fill="#dddddd", width=1)
        draw.text((x - 10, bottom + 10), f"{i/5:.1f}", fill="#333333", font=small_font)
        draw.text((left - 48, y - 8), f"{i/5:.1f}", fill="#333333", font=small_font)

    draw.line((left, bottom, right, top), fill="#aaaaaa", width=2)

    def plot(points: list[tuple[float, float]], color: str):
        scaled = []
        for fpr, tpr in points:
            x = left + fpr * (right - left)
            y = bottom - tpr * (bottom - top)
            scaled.append((x, y))
        if len(scaled) > 1:
            draw.line(scaled, fill=color, width=4)

    plot(cifake_pts, "#0072B2")
    plot(mfad_pts, "#D55E00")

    draw.text((450, 695), "False Positive Rate", fill="#111111", font=label_font)
    draw.text((10, 360), "True Positive Rate", fill="#111111", font=label_font)
    draw.text((735, 150), f"CIFake val AUC: {cifake_metrics['auc']*100:.2f}%", fill="#0072B2", font=label_font)
    draw.text((735, 185), f"MFAD-Net FF++ test AUC: {mfad_metrics['auc']*100:.2f}%", fill="#D55E00", font=label_font)

    out = FIG_DIR / "roc_comparison.png"
    img.save(out)
    return out


def draw_filtered_bar() -> Path:
    raw = {
        "accuracy": 0.97275,
        "precision": 0.975046055937029,
        "recall": 0.9703333333333334,
        "f1": 0.9726839863002256,
        "auc": 0.9967227222222222,
    }
    filtered = load_metrics(ROOT / "training/exports/image_branch/cifake/filtered_jpeg_metrics.json")
    keys = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("auc", "ROC-AUC")]

    img, draw = new_canvas()
    title_font = font(28, bold=True)
    label_font = font(18)
    small_font = font(16)
    draw.text((70, 40), "CIFake Raw vs Filtered Robustness", fill="#111111", font=title_font)

    chart_left, chart_top, chart_right, chart_bottom = 90, 140, 1080, 650
    draw.rectangle((chart_left, chart_top, chart_right, chart_bottom), outline="#222222", width=2)
    for i in range(6):
        y = chart_bottom - (chart_bottom - chart_top) * i / 5
        draw.line((chart_left, y, chart_right, y), fill="#dddddd", width=1)
        draw.text((35, y - 8), f"{20*i}", fill="#333333", font=small_font)

    group_width = (chart_right - chart_left) / len(keys)
    bar_width = 52
    for idx, (key, label) in enumerate(keys):
        cx = chart_left + group_width * idx + group_width / 2
        raw_h = raw[key] * (chart_bottom - chart_top)
        filt_h = filtered[key] * (chart_bottom - chart_top)
        draw.rectangle((cx - 62, chart_bottom - raw_h, cx - 10, chart_bottom), fill="#0072B2")
        draw.rectangle((cx + 10, chart_bottom - filt_h, cx + 62, chart_bottom), fill="#E69F00")
        draw.text((cx - 38, chart_bottom + 15), label, fill="#111111", font=label_font, anchor="ma")
        draw.text((cx - 36, chart_bottom - raw_h - 22), f"{raw[key]*100:.1f}", fill="#0072B2", font=small_font)
        draw.text((cx + 12, chart_bottom - filt_h - 22), f"{filtered[key]*100:.1f}", fill="#E69F00", font=small_font)

    draw.rectangle((760, 110, 790, 130), fill="#0072B2")
    draw.text((800, 107), "Raw CIFake validation", fill="#111111", font=label_font)
    draw.rectangle((760, 145, 790, 165), fill="#E69F00")
    draw.text((800, 142), "JPEG-filtered evaluation", fill="#111111", font=label_font)

    out = FIG_DIR / "filtered_robustness.png"
    img.save(out)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    roc = draw_roc()
    filt = draw_filtered_bar()
    print(json.dumps({"roc": str(roc), "filtered": str(filt)}, indent=2))


if __name__ == "__main__":
    main()
