import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core import media_utils
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


def _synthetic_wav(seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Simple tone + noise WAV used to exercise the audio detector path."""
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * 220 * t)
    noise = 0.02 * np.random.default_rng(0).standard_normal(t.shape[0])
    pcm = np.clip(tone + noise, -1, 1)
    samples = (pcm * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def _minimal_mp4_with_audio() -> bytes | None:
    """Build a tiny mp4 via ffmpeg if available; skip video tests otherwise."""
    import subprocess
    import tempfile
    from pathlib import Path

    try:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "clip.mp4"
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", "color=c=gray:s=160x120:d=1",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest",
                    str(out),
                ],
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0 or not out.exists():
                return None
            return out.read_bytes()
    except Exception:
        return None


def test_status():
    resp = client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert isinstance(data["detectors"], list)
    assert len(data["detectors"]) > 0
    assert any(d["name"] == "social_multimodal_ensemble" for d in data["detectors"])
    assert any(d["name"] == "synthetic_speech_audio" for d in data["detectors"])


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


def test_detect_media_kind_image_and_video_magic():
    jpeg = _synthetic_jpeg()
    assert media_utils.detect_media_kind(jpeg, content_type="application/octet-stream") == "image"

    # Minimal WebM/Matroska EBML header
    webm_head = b"\x1a\x45\xdf\xa3" + b"\x00" * 20
    assert media_utils.detect_media_kind(webm_head, filename="clip.webm") == "video"


def test_analyze_media_auto_detects_image():
    jpeg_bytes = _synthetic_jpeg()
    resp = client.post(
        "/analyze/media",
        files={"file": ("mystery.bin", jpeg_bytes, "application/octet-stream")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["media_type"] == "image"
    assert data["metadata"].get("input_kind") == "image"


def test_analyze_frames_endpoint():
    frames = [_synthetic_jpeg(noisy=True), _synthetic_jpeg(noisy=False), _synthetic_jpeg(noisy=True)]
    files = [
        ("files", (f"f{i}.jpg", frame, "image/jpeg"))
        for i, frame in enumerate(frames)
    ]
    resp = client.post("/analyze/frames", files=files, data={"platform": "youtube"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["media_type"] == "video"
    assert data["frame_results"] and len(data["frame_results"]) == 3
    assert data["audio_result"] is not None
    assert data["audio_result"]["available"] is False
    assert data["metadata"]["pipeline"] == "frames_as_images"


def test_analyze_video_runs_frames_and_audio_parallel():
    clip = _minimal_mp4_with_audio()
    if clip is None:
        pytest.skip("ffmpeg could not produce a test mp4")

    resp = client.post(
        "/analyze/video",
        files={"file": ("clip.mp4", clip, "video/mp4")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["media_type"] == "video"
    assert data["frame_results"] and len(data["frame_results"]) >= 1
    assert data["audio_result"] is not None
    assert data["metadata"].get("pipeline") == "frames_as_images+audio_parallel"
    detector_names = {d["detector"] for d in data["detector_results"]}
    assert "synthetic_speech_audio" in detector_names
    assert "temporal_consistency" in detector_names or any(
        n.startswith("frequency") or n.startswith("sensor") for n in detector_names
    )


def test_synthetic_speech_detector_on_wav():
    from app.detectors.audio.synthetic_speech_detector import SyntheticSpeechDetector

    wav = _synthetic_wav(seconds=1.2)
    with wave.open(io.BytesIO(wav), "rb") as wf:
        sr = wf.getframerate()
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

    det = SyntheticSpeechDetector()
    det.ensure_loaded()
    result = det.analyze_audio(pcm, sr)
    assert result.error is None
    assert 0.0 <= result.ai_probability <= 1.0
    assert result.evidence
