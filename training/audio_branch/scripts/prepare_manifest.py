from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _label(value: str) -> int | None:
    value = value.lower()
    if value in {"bonafide", "genuine", "real", "target"}:
        return 0
    if value in {"spoof", "fake", "synthetic", "attack"}:
        return 1
    return None


def _audio_path(token: str, audio_root: Path) -> Path | None:
    candidate = Path(token)
    options = [candidate, audio_root / candidate, audio_root / f"{token}.flac", audio_root / f"{token}.wav"]
    direct = next((p.resolve() for p in options if p.is_file()), None)
    if direct is not None:
        return direct
    for suffix in (".flac", ".wav"):
        matches = list(audio_root.rglob(f"{token}{suffix}"))
        if matches:
            return matches[0].resolve()
    return None


def parse_protocol(protocol: Path, audio_root: Path, split: str) -> list[dict[str, object]]:
    rows = []
    for line in protocol.read_text(encoding="utf-8", errors="ignore").splitlines():
        tokens = line.split()
        if not tokens or tokens[0].startswith("#"):
            continue
        label = next((_label(token) for token in reversed(tokens)), None)
        if label is None:
            continue
        path = next((_audio_path(token, audio_root) for token in tokens), None)
        if path is None:
            continue
        rows.append({"path": str(path), "label": label, "split": split, "source": "asvspoof5"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an ASVspoof protocol CSV")
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--audio-root", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = parse_protocol(args.protocol, args.audio_root, args.split)
    if not rows:
        raise SystemExit("No audio rows found. Check ASVspoof protocol format and --audio-root.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()