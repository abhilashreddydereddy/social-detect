#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.main import app


@dataclass
class EvalRow:
    path: Path
    label: int
    media_type: str
    platform: str
    source: str


def _auc(labels: list[int], probs: list[float]) -> float:
    pairs = sorted(zip(probs, labels), key=lambda x: x[0])
    pos = sum(1 for _, y in pairs if y == 1)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return 0.5
    rank_sum = 0.0
    for i, (_, y) in enumerate(pairs, start=1):
        if y == 1:
            rank_sum += i
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def _compute_metrics(labels: list[int], probs: list[float], threshold: float) -> dict[str, Any]:
    preds = [1 if p >= threshold else 0 for p in probs]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    total = len(labels)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "n": total,
        "accuracy": (tp + tn) / max(total, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": _auc(labels, probs),
        "threshold": threshold,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _guess_content_type(path: Path, media_type: str) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed:
        return guessed
    if media_type == "image":
        return "image/jpeg"
    if media_type == "video":
        return "video/mp4"
    return "application/octet-stream"


def _endpoint_for(media_type: str) -> str:
    if media_type == "image":
        return "/analyze/image"
    if media_type == "video":
        return "/analyze/video"
    return "/analyze/media"


def load_manifest(path: Path, default_platform: str) -> list[EvalRow]:
    rows: list[EvalRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"path", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        for idx, raw in enumerate(reader, start=2):
            sample_path = Path(str(raw["path"]).strip())
            if not sample_path.is_absolute():
                sample_path = (REPO_ROOT / sample_path).resolve()
            media_type = str(raw.get("media_type") or "auto").strip().lower()
            if media_type not in {"image", "video", "auto"}:
                raise ValueError(f"Line {idx}: unsupported media_type '{media_type}'")
            label_text = str(raw["label"]).strip()
            if label_text not in {"0", "1"}:
                raise ValueError(f"Line {idx}: label must be 0 or 1")
            rows.append(
                EvalRow(
                    path=sample_path,
                    label=int(label_text),
                    media_type=media_type,
                    platform=str(raw.get("platform") or default_platform).strip() or default_platform,
                    source=str(raw.get("source") or sample_path.name).strip() or sample_path.name,
                )
            )
    return rows


def post_sample(
    client: Any,
    row: EvalRow,
) -> tuple[float, int, dict[str, Any]]:
    body = row.path.read_bytes()
    content_type = _guess_content_type(row.path, row.media_type)
    endpoint = _endpoint_for(row.media_type)
    started = time.perf_counter()
    resp = client.post(
        endpoint,
        files={"file": (row.path.name, body, content_type)},
        data={"platform": row.platform},
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    payload: dict[str, Any]
    try:
        payload = resp.json()
    except Exception:  # noqa: BLE001
        payload = {"detail": resp.text}
    return elapsed_ms, resp.status_code, payload


def build_client(base_url: str | None) -> Any:
    if base_url:
        import httpx

        return httpx.Client(base_url=base_url.rstrip("/"), timeout=120.0)
    return TestClient(app)


def get_status_snapshot(client: Any) -> dict[str, Any] | None:
    try:
        resp = client.get("/status")
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate labeled 'realtime' media against the current Social Detect backend."
    )
    parser.add_argument("--manifest", required=True, help="CSV with columns: path,label[,media_type,platform,source]")
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional live backend URL, e.g. http://127.0.0.1:8000. Defaults to in-process TestClient.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--platform", default="unknown", help="Default platform label when not present in manifest")
    parser.add_argument(
        "--output-dir",
        default="backend/eval_outputs/realtime",
        help="Directory for metrics JSON and detailed CSV",
    )
    parser.add_argument("--run-name", default=None, help="Optional run name. Defaults to manifest stem + timestamp.")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = (REPO_ROOT / manifest).resolve()
    rows = load_manifest(manifest, default_platform=args.platform)
    if not rows:
        raise SystemExit("Manifest contains no rows")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or f"{manifest.stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    detail_path = output_dir / f"{run_name}_predictions.csv"
    summary_path = output_dir / f"{run_name}_summary.json"

    client = build_client(args.base_url)
    close = getattr(client, "close", None)
    try:
        status_snapshot = get_status_snapshot(client)
        labels: list[int] = []
        probs: list[float] = []
        wall_times: list[float] = []
        backend_times: list[float] = []
        successful = 0
        failed = 0
        detail_rows: list[dict[str, Any]] = []

        for row in rows:
            if not row.path.exists():
                failed += 1
                detail_rows.append(
                    {
                        "path": str(row.path),
                        "source": row.source,
                        "label": row.label,
                        "media_type": row.media_type,
                        "platform": row.platform,
                        "status_code": 0,
                        "ok": False,
                        "error": "File not found",
                    }
                )
                continue

            elapsed_ms, status_code, payload = post_sample(client, row)
            ok = status_code == 200 and isinstance(payload, dict) and "ai_probability" in payload
            detail = {
                "path": str(row.path),
                "source": row.source,
                "label": row.label,
                "media_type": row.media_type,
                "platform": row.platform,
                "status_code": status_code,
                "ok": ok,
                "wall_time_ms": round(elapsed_ms, 3),
            }
            if ok:
                successful += 1
                prob = float(payload["ai_probability"])
                labels.append(row.label)
                probs.append(prob)
                wall_times.append(elapsed_ms)
                processing_time_ms = payload.get("processing_time_ms")
                if processing_time_ms is not None:
                    backend_times.append(float(processing_time_ms))
                detail.update(
                    {
                        "request_id": payload.get("request_id"),
                        "ai_probability": prob,
                        "pred_label": 1 if prob >= args.threshold else 0,
                        "confidence": payload.get("confidence"),
                        "classification": payload.get("classification"),
                        "processing_time_ms": processing_time_ms,
                        "audio_available": (
                            payload.get("audio_result", {}) or {}
                        ).get("available"),
                        "audio_error": (payload.get("audio_result", {}) or {}).get("error"),
                        "primary_model": (payload.get("metadata", {}) or {}).get("primary_model"),
                        "pipeline": (payload.get("metadata", {}) or {}).get("pipeline"),
                    }
                )
            else:
                failed += 1
                detail["error"] = payload.get("detail") if isinstance(payload, dict) else str(payload)
            detail_rows.append(detail)

        metrics = _compute_metrics(labels, probs, threshold=args.threshold) if successful else None
        summary = {
            "run_name": run_name,
            "manifest": str(manifest),
            "base_url": args.base_url,
            "attempted": len(rows),
            "successful": successful,
            "failed": failed,
            "default_platform": args.platform,
            "status_snapshot": status_snapshot,
            "latency_ms": {
                "wall_time_mean": _mean(wall_times),
                "processing_time_mean": _mean(backend_times),
                "wall_time_max": max(wall_times) if wall_times else None,
                "processing_time_max": max(backend_times) if backend_times else None,
            },
            "metrics": metrics,
            "notes": [
                "Metrics are computed only across successful labeled requests.",
                "This evaluator uses /analyze/image, /analyze/video, or /analyze/media based on manifest media_type.",
                "For a true full-multimodal benchmark, video samples must include audio and should use media_type=video or auto.",
            ],
            "artifacts": {
                "predictions_csv": str(detail_path),
                "summary_json": str(summary_path),
            },
        }

        with detail_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = sorted({key for row in detail_rows for key in row.keys()})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detail_rows)

        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        if callable(close):
            close()


if __name__ == "__main__":
    main()
