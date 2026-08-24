"""
Media I/O helpers shared by the analysis service:
  - decode uploaded image bytes -> numpy RGB array
  - extract sampled frames from a video file via OpenCV/FFmpeg
  - fetch remote media (direct media URLs, or resolve a social post URL to
    its underlying media URL via lightweight platform-specific resolvers)

Platform note: Instagram/TikTok/X/Facebook do not offer a stable public API
for arbitrary post scraping, and actively discourage server-side scraping.
The intended real-world flow is:
  - the BROWSER EXTENSION already has DOM access to the rendered post and
    extracts the actual <img>/<video> src directly from the page (see
    extension/content_scripts/*.js) and either uploads the media blob or
    sends the resolved direct media URL -- it does not ask the backend to
    scrape the platform.
  - POST /analyze/url exists mainly for the dashboard's "paste a URL" demo
    flow and for direct media URLs (e.g. a CDN-hosted image/video link).
    Platform-specific resolvers below are best-effort and intentionally
    isolated so they can be hardened or replaced (e.g. with an official
    Graph API integration) without touching the rest of the pipeline.
"""
from __future__ import annotations

import io
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlparse

import cv2
import httpx
import numpy as np
from PIL import Image

from app.config import settings

MAX_VIDEO_FRAMES = settings.max_video_frames


class MediaFetchError(Exception):
    pass


def decode_image_bytes(raw: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(img)


async def fetch_url_bytes(url: str) -> Tuple[bytes, str]:
    """Returns (raw_bytes, content_type)."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=settings.download_timeout_seconds) as client:
        try:
            resp = await client.get(url, headers={"User-Agent": "SocialDetect/1.0"})
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise MediaFetchError(f"Failed to download media: {exc}") from exc
        content_type = resp.headers.get("content-type", "")
        return resp.content, content_type


def resolve_platform_hint(url: str) -> str:
    host = urlparse(url).netloc.lower()
    mapping = {
        "instagram.com": "instagram",
        "x.com": "x",
        "twitter.com": "x",
        "reddit.com": "reddit",
        "facebook.com": "facebook",
        "tiktok.com": "tiktok",
        "youtube.com": "youtube",
        "youtu.be": "youtube",
    }
    for domain, platform in mapping.items():
        if domain in host:
            return platform
    return "direct"


def extract_video_frames(video_bytes: bytes, max_frames: int = MAX_VIDEO_FRAMES) -> Tuple[List[np.ndarray], List[float]]:
    """Uniformly sample up to `max_frames` frames across the clip using OpenCV
    (which shells out to ffmpeg via its backend). Returns (frames, timestamps_s).
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise MediaFetchError("Could not open video for frame extraction")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise MediaFetchError("Video has no readable frames")

        sample_count = min(max_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, sample_count).astype(int)

        frames, timestamps = [], []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame_bgr = cap.read()
            if not ok:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            timestamps.append(float(idx) / float(fps))

        cap.release()
        if not frames:
            raise MediaFetchError("No frames could be decoded from video")
        return frames, timestamps
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def probe_video_metadata(video_bytes: bytes) -> dict:
    """Uses ffprobe (bundled with ffmpeg) to pull duration/codec/resolution.
    Falls back to {} if ffprobe isn't available in the environment."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", tmp_path,
            ],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return {}
        import json
        return json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return {}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def thumbnail_base64(frame_rgb: np.ndarray, max_dim: int = 160) -> str:
    import base64
    h, w = frame_rgb.shape[:2]
    scale = max_dim / max(h, w)
    small = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")
