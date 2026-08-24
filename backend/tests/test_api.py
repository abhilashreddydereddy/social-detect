import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def _synthetic_jpeg(size=(320, 240), noisy=True) -> bytes:
    rng = np.random.default_rng(42)
    if noisy:
        arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    else:
        arr = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return buf.read()


def test_status():
    resp = client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert isinstance(data["detectors"], list)
    assert len(data["detectors"]) > 0
    assert any(d["name"] == "social_multimodal_ensemble" for d in data["detectors"])


def test_analyze_image_returns_probabilistic_fields():
    jpeg_bytes = _synthetic_jpeg()
    resp = client.post(
        "/analyze/image",
        files={"file": ("test.jpg", jpeg_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert 0.0 <= data["ai_probability"] <= 1.0
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["classification"] in {
        "Likely Authentic", "Possibly Manipulated", "Likely AI Generated", "Inconclusive",
    }
    assert isinstance(data["evidence"], list)
    assert len(data["detector_results"]) > 0
    detector_names = {item["detector"] for item in data["detector_results"]}
    assert "social_multimodal_ensemble" in detector_names
    # never claims certainty
    assert "not proof" in data["disclaimer"].lower()


def test_analyze_image_rejects_bad_content_type():
    resp = client.post(
        "/analyze/image",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 415


def test_analyze_url_rejects_non_media_page():
    resp = client.post("/analyze/url", json={"url": "https://example.com"})
    assert resp.status_code in (415, 422)
