from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


METHODS = ("original", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures", "DeepFakeDetection")


def source_group(path: Path) -> str:
    """Group a manipulated clip with its source identity for split isolation."""
    stem = path.stem
    return stem.split("_")[0]


def split_for_group(group: str) -> str:
    value = int(hashlib.sha1(group.encode("utf-8")).hexdigest()[:8], 16) % 100
    if value < 70:
        return "train"
    if value < 85:
        return "val"
    return "test"


def extract_frames(video: Path, output: Path, count: int, size: int) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Video has no readable frames: {video}")
    indices = np.linspace(0, total - 1, count).astype(int)
    for frame_index, source_index in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(source_index))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame).resize((size, size), Image.Resampling.BICUBIC)
        image.save(output / f"frame_{frame_index:02d}.jpg", quality=90)
    cap.release()
    if not list(output.glob("frame_*.jpg")):
        raise RuntimeError(f"No frames extracted: {video}")


def extract_audio(video: Path, output: Path, sample_rate: int, seconds: float) -> None:
    samples = int(sample_rate * seconds)
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as handle:
        raw_path = Path(handle.name)
    try:
        command = [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn",
            "-ac", "1", "-ar", str(sample_rate), "-t", str(seconds),
            "-f", "f32le", str(raw_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        if result.returncode != 0:
            waveform = np.zeros(samples, dtype=np.float32)
        else:
            waveform = np.fromfile(raw_path, dtype=np.float32)[:samples]
            if waveform.size < samples:
                waveform = np.pad(waveform, (0, samples - waveform.size))
        np.save(output / "audio.npy", waveform.astype(np.float32))
    finally:
        raw_path.unlink(missing_ok=True)


def prepare_sample(video: Path, output_root: Path, label: int, frames: int, image_size: int, sample_rate: int, seconds: float) -> dict[str, object]:
    sample_id = f"{video.parent.name}_{video.stem}"
    sample_dir = output_root / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    extract_frames(video, sample_dir, frames, image_size)
    extract_audio(video, sample_dir, sample_rate, seconds)
    # FF++ has no social-account metadata; zeros represent unavailable context.
    np.save(sample_dir / "meta.npy", np.zeros(16, dtype=np.float32))
    return {"sample_id": sample_id, "path": str(sample_dir.resolve()), "label": label, "source": "faceforensics++", "split": split_for_group(source_group(video))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert FaceForensics++ videos for MFAD-Net")
    parser.add_argument("--input-root", type=Path, default=Path("training/data/faceforensics/FaceForensics++_C23"))
    parser.add_argument("--output-root", type=Path, default=Path("training/data/mfad_net/faceforensics_processed"))
    parser.add_argument("--manifests-dir", type=Path, default=Path("training/data/mfad_net/splits"))
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--max-per-method", type=int, default=0, help="Limit each method for a quick smoke run; 0 means all")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for method in METHODS:
        method_dir = args.input_root / method
        videos = sorted(method_dir.glob("*.mp4"))
        if args.max_per_method:
            videos = videos[:args.max_per_method]
        label = 0 if method == "original" else 1
        print(f"{method}: {len(videos)} videos")
        for index, video in enumerate(videos, 1):
            try:
                rows.append(prepare_sample(video, args.output_root, label, args.frames, args.image_size, args.sample_rate, args.seconds))
            except Exception as exc:  # noqa: BLE001
                print(f"skip {video}: {exc}")
            if index % 25 == 0:
                print(f"  processed {index}/{len(videos)}")

    if not rows:
        raise SystemExit("No samples were prepared. Check --input-root and FFmpeg installation.")
    args.manifests_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        with (args.manifests_dir / f"faceforensics_{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "path", "label", "source", "split"])
            writer.writeheader()
            writer.writerows(split_rows)
        print(f"{split}: {len(split_rows)} samples")


if __name__ == "__main__":
    main()