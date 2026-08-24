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
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import httpx
import numpy as np
from PIL import Image

from app.config import settings

MAX_VIDEO_FRAMES = settings.max_video_frames

IMAGE_CONTENT_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp", "image/tiff",
}
VIDEO_CONTENT_TYPES = {
    "video/mp4", "video/webm", "video/quicktime", "video/x-matroska", "video/x-msvideo",
}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff")
VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v")


class MediaFetchError(Exception):
    pass


def detect_media_kind(
    raw: bytes,
    content_type: str | None = None,
    filename: str | None = None,
) -> str:
    """Return 'image' or 'video' from magic bytes, content-type, or filename.

    Raises MediaFetchError when the payload cannot be classified as either.
    """
    kind = _sniff_magic(raw)
    if kind:
        return kind

    ct = (content_type or "").split(";")[0].strip().lower()
    if ct in IMAGE_CONTENT_TYPES or ct.startswith("image/"):
        return "image"
    if ct in VIDEO_CONTENT_TYPES or ct.startswith("video/"):
        return "video"

    name = (filename or "").lower().split("?")[0]
    if name.endswith(IMAGE_EXTENSIONS):
        return "image"
    if name.endswith(VIDEO_EXTENSIONS):
        return "video"

    # Last-resort: try decoding as an image (covers odd content-types).
    try:
        Image.open(io.BytesIO(raw)).verify()
        return "image"
    except Exception:
        pass

    raise MediaFetchError(
        f"Could not determine media type (content-type='{content_type or ''}', "
        f"filename='{filename or ''}'). Upload an image or video file."
    )


def _sniff_magic(raw: bytes) -> str | None:
    if not raw or len(raw) < 12:
        return None
    head = raw[:32]
    # Images
    if head.startswith(b"\xff\xd8\xff"):
        return "image"  # JPEG
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "image"
    if head.startswith(b"BM"):
        return "image"
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "image"
    # Videos / containers
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        return "video"  # Matroska / WebM
    if head[4:8] == b"ftyp":
        # ISO BMFF: mp4, mov, m4v, etc. — treat as video (HEIC rare for this API).
        brand = head[8:12]
        if brand in (b"heic", b"heif", b"mif1", b"msf1"):
            return "image"
        return "video"
    if head.startswith(b"RIFF") and head[8:12] == b"AVI ":
        return "video"
    if head.startswith(b"OggS"):
        return "video"
    return None


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


def extract_audio_waveform(
    video_bytes: bytes,
    sample_rate: int = 16000,
) -> Tuple[Optional[np.ndarray], int, Optional[str]]:
    """Extract mono PCM float32 audio from a video container via ffmpeg.

    Returns (waveform, sample_rate, error_message).
    waveform is None when the container has no audio stream or ffmpeg fails.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
        tmp_in.write(video_bytes)
        in_path = tmp_in.name

    out_path = in_path + ".wav"
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", in_path,
                "-vn",
                "-ac", "1",
                "-ar", str(sample_rate),
                "-f", "wav",
                out_path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            err = (result.stderr or "").strip().splitlines()
            detail = err[-1] if err else "ffmpeg audio extraction failed"
            # Common case: video with no audio track.
            if "does not contain any stream" in (result.stderr or "") or "Output file does not contain any stream" in (result.stderr or ""):
                return None, sample_rate, "Video has no audio track"
            return None, sample_rate, detail

        out = Path(out_path)
        if not out.exists() or out.stat().st_size < 44:
            return None, sample_rate, "Video has no audio track"

        import wave
        with wave.open(out_path, "rb") as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            raw_pcm = wf.readframes(n)

        if width == 2:
            pcm = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        elif width == 1:
            pcm = (np.frombuffer(raw_pcm, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif width == 4:
            pcm = np.frombuffer(raw_pcm, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            return None, sample_rate, f"Unsupported PCM sample width: {width}"

        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1)

        if pcm.size == 0:
            return None, sr, "Extracted audio is empty"
        return pcm, sr, None
    except FileNotFoundError:
        return None, sample_rate, "ffmpeg is not installed"
    except subprocess.TimeoutExpired:
        return None, sample_rate, "Audio extraction timed out"
    except Exception as exc:  # noqa: BLE001
        return None, sample_rate, str(exc)
    finally:
        Path(in_path).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)


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
