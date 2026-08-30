from pathlib import Path

from scripts.evaluate_realtime import _compute_metrics, _endpoint_for, load_manifest


def test_endpoint_for_media_type():
    assert _endpoint_for("image") == "/analyze/image"
    assert _endpoint_for("video") == "/analyze/video"
    assert _endpoint_for("auto") == "/analyze/media"


def test_compute_metrics_binary_scores():
    metrics = _compute_metrics(labels=[0, 0, 1, 1], probs=[0.1, 0.8, 0.7, 0.9], threshold=0.5)
    assert metrics["n"] == 4
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}
    assert metrics["accuracy"] == 0.75
    assert 0.0 <= metrics["auc"] <= 1.0


def test_load_manifest_accepts_relative_paths(tmp_path: Path):
    sample = tmp_path / "sample.jpg"
    sample.write_bytes(b"abc")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "path,label,media_type,platform,source\n"
        f"{sample},1,image,youtube,row-1\n",
        encoding="utf-8",
    )
    rows = load_manifest(manifest, default_platform="unknown")
    assert len(rows) == 1
    assert rows[0].path == sample
    assert rows[0].label == 1
    assert rows[0].media_type == "image"
    assert rows[0].platform == "youtube"
    assert rows[0].source == "row-1"
