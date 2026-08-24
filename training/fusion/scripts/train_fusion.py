from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline fusion model.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_df = pd.read_csv(config["train_features_csv"])
    val_df = pd.read_csv(config["val_features_csv"])
    target_column = config.get("target_column", "label")

    feature_columns = [c for c in train_df.columns if c != target_column]
    x_train = train_df[feature_columns]
    y_train = train_df[target_column]
    x_val = val_df[feature_columns]
    y_val = val_df[target_column]

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    val_scores = model.predict_proba(x_val)[:, 1]
    auc = roc_auc_score(y_val, val_scores)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_dir / "fusion.pkl")
    with open(output_dir / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump({"feature_columns": feature_columns}, f, indent=2)

    print(f"Validation AUC: {auc:.4f}")
    print(f"Saved fusion model to {output_dir / 'fusion.pkl'}")


if __name__ == "__main__":
    main()
