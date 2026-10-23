"""Download a size-capped, label-balanced ASVspoof 5 evaluation subset.

The source dataset is streamed from Hugging Face; it is never materialised in
full. Each selected audio payload is stored as its original FLAC bytes and a
JSONL manifest records the binary label and ASVspoof metadata.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import Audio, load_dataset

DATASET_ID = "SpeechAntiSpoofingBenchmarks/ASVspoof5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("training/data/asvspoof5_subset"))
    parser.add_argument("--max-gb", type=float, default=11.0, help="Maximum stored audio size in GiB")
    parser.add_argument("--max-samples", type=int, default=50_000, help="Hard cap on stored examples")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_gb <= 0 or args.max_samples <= 0:
        raise ValueError("--max-gb and --max-samples must both be positive")

    output_dir = args.output_dir.resolve()
    audio_dir = output_dir / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "README.json"
    if manifest_path.exists():
        if manifest_path.stat().st_size:
            raise FileExistsError(f"{manifest_path} already exists; choose a new --output-dir to avoid mixing subsets")
        # A failed connection can leave an empty manifest; it is safe to retry.
        manifest_path.unlink()

    dataset = load_dataset(DATASET_ID, split="test", streaming=True)
    dataset = dataset.cast_column("audio", Audio(decode=False)).shuffle(seed=args.seed, buffer_size=20_000)
    max_bytes = int(args.max_gb * 1024 ** 3)
    target_per_label = args.max_samples // 2
    counts: Counter[int] = Counter()
    written_bytes = 0

    with manifest_path.open("x", encoding="utf-8") as manifest:
        for row in dataset:
            label = int(row["label"])
            if counts[label] >= target_per_label:
                continue
            payload = row["audio"].get("bytes")
            if not payload or written_bytes + len(payload) > max_bytes:
                continue

            filename = Path(row["path"]).name
            destination = audio_dir / filename
            destination.write_bytes(payload)
            notes = row.get("notes")
            manifest.write(json.dumps({
                "path": str(destination.relative_to(output_dir)).replace("\\\\", "/"),
                "label": label,
                "label_name": "spoof" if label else "bonafide",
                "notes": json.loads(notes) if isinstance(notes, str) else notes,
            }) + "\n")
            counts[label] += 1
            written_bytes += len(payload)
            if sum(counts.values()) >= args.max_samples or written_bytes >= max_bytes:
                break

    summary_path.write_text(json.dumps({
        "source": DATASET_ID, "split": "test", "samples": sum(counts.values()),
        "bonafide": counts[0], "spoof": counts[1], "audio_bytes": written_bytes,
        "audio_gib": round(written_bytes / 1024 ** 3, 3), "max_gib": args.max_gb,
        "seed": args.seed,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Downloaded {sum(counts.values())} samples ({written_bytes / 1024 ** 3:.2f} GiB) to {output_dir}")


if __name__ == "__main__":
    main()
