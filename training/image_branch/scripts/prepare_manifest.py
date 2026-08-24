from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a simple image manifest from labeled folders.")
    parser.add_argument("--real-dir", required=True, help="Folder containing real images")
    parser.add_argument("--fake-dir", required=True, help="Folder containing fake images")
    parser.add_argument("--split", default="train", help="Split label to write into the manifest")
    parser.add_argument("--out-csv", required=True, help="Destination CSV path")
    args = parser.parse_args()

    rows = []
    for label, root in [(0, Path(args.real_dir)), (1, Path(args.fake_dir))]:
        for path in root.rglob("*"):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            rows.append({
                "path": str(path),
                "label": label,
                "split": args.split,
                "source": root.name,
                "platform": "unknown",
            })

    df = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
