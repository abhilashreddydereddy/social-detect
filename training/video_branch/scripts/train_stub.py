from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Stub entrypoint for the video-branch training loop.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_df = pd.read_csv(config["train_manifest"])
    val_df = pd.read_csv(config["val_manifest"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Video branch training stub")
    print(f"Framework: {config['framework']}")
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows: {len(val_df)}")
    print(f"Output dir: {output_dir}")
    print("Next implementation step: call DeepfakeBench training/eval here.")


if __name__ == "__main__":
    main()
