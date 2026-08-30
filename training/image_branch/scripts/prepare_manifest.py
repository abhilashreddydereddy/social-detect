from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _iter_images(root: Path):
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def build_rows(real_dir: Path, fake_dir: Path, split: str, source: str) -> list[dict]:
    rows = []
    for label, folder in [(0, real_dir), (1, fake_dir)]:
        for path in _iter_images(folder):
            rows.append({
                "path": str(path.resolve()),
                "label": label,
                "split": split,
                "source": source,
                "platform": "unknown",
            })
    return rows


def write_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows to {out_csv}")


def prepare_from_dirs(args: argparse.Namespace) -> None:
    rows = build_rows(Path(args.real_dir), Path(args.fake_dir), args.split, source=Path(args.real_dir).name)
    write_csv(rows, Path(args.out_csv))


def _find_class_dir(split_root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = split_root / name
        if candidate.is_dir():
            return candidate
    # Case-insensitive match
    lower_map = {p.name.lower(): p for p in split_root.iterdir() if p.is_dir()}
    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def prepare_cifake_tree(args: argparse.Namespace) -> None:
    """
    Accept either:
      root/train/{real|REAL}, root/train/{fake|FAKE}, and same for val/test
    or Kaggle CIFake:
      root/train/{REAL,FAKE}, root/test/{REAL,FAKE}
    """
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    manifests_dir = Path(args.manifests_dir)

    # Discover splits
    train_src = None
    for name in ("train", "Train"):
        if (root / name).is_dir():
            train_src = root / name
            break
    test_src = None
    for name in ("test", "Test", "val", "Val", "validation"):
        if (root / name).is_dir():
            test_src = root / name
            break

    if train_src is None:
        raise SystemExit(f"No train/ folder under {root}")

    # Materialize into project layout training/data/cifake/{train,val}/{real,fake}
    mapping = [("train", train_src)]
    if test_src is not None:
        mapping.append(("val", test_src))

    for dest_split, src_split in mapping:
        real_src = _find_class_dir(src_split, ("real", "REAL", "Real"))
        fake_src = _find_class_dir(src_split, ("fake", "FAKE", "Fake"))
        if real_src is None or fake_src is None:
            raise SystemExit(f"Expected REAL/FAKE (or real/fake) under {src_split}")

        real_dst = out_dir / dest_split / "real"
        fake_dst = out_dir / dest_split / "fake"
        if args.copy:
            if real_dst.exists():
                shutil.rmtree(real_dst)
            if fake_dst.exists():
                shutil.rmtree(fake_dst)
            shutil.copytree(real_src, real_dst)
            shutil.copytree(fake_src, fake_dst)
            real_for_manifest, fake_for_manifest = real_dst, fake_dst
        else:
            # Manifest can point at original Kaggle paths without copying
            real_for_manifest, fake_for_manifest = real_src, fake_src
            # Still create empty marker dirs for docs consistency
            real_dst.mkdir(parents=True, exist_ok=True)
            fake_dst.mkdir(parents=True, exist_ok=True)

        rows = build_rows(real_for_manifest, fake_for_manifest, dest_split, source="cifake")
        write_csv(rows, manifests_dir / f"cifake_{dest_split}.csv")

    if test_src is None:
        print("Warning: no test/val split found; only cifake_train.csv was written.")
        print("Hold out a validation set from train before training.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build image manifests for CIFake / folder datasets.")
    sub = parser.add_subparsers(dest="command")

    # Backward-compatible flat flags (no subcommand)
    parser.add_argument("--real-dir", help="Folder containing real images")
    parser.add_argument("--fake-dir", help="Folder containing fake images")
    parser.add_argument("--split", default="train", help="Split label to write into the manifest")
    parser.add_argument("--out-csv", help="Destination CSV path")

    kaggle = sub.add_parser("from-tree", help="Import CIFake/Kaggle tree into manifests (and optional copy)")
    kaggle.add_argument("--root", required=True, help="Root containing train/ and test/ (or val/)")
    kaggle.add_argument("--out-dir", default="training/data/cifake", help="Normalized CIFake layout root")
    kaggle.add_argument("--manifests-dir", default="training/data/manifests", help="Where to write CSVs")
    kaggle.add_argument("--copy", action="store_true", help="Copy images into out-dir (default: manifest-only)")

    args = parser.parse_args()

    if args.command == "from-tree":
        prepare_cifake_tree(args)
        return

    if not args.real_dir or not args.fake_dir or not args.out_csv:
        parser.error("Provide --real-dir/--fake-dir/--out-csv, or use: from-tree --root ...")
    prepare_from_dirs(args)


if __name__ == "__main__":
    main()
