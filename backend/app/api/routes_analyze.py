from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.core import media_utils
from app.core.media_utils import (
    IMAGE_CONTENT_TYPES,
    VIDEO_CONTENT_TYPES,
    MediaFetchError,
)
from app.core.schemas import AnalysisResponse, UrlAnalyzeRequest
from app.services import analysis_service

router = APIRouter(prefix="/analyze", tags=["analyze"])

MAX_BYTES = settings.max_upload_mb * 1024 * 1024


def _check_size(raw: bytes) -> None:
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb}MB limit")


@router.post("/image", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    if file.content_type and file.content_type not in IMAGE_CONTENT_TYPES and not file.content_type.startswith("image/"):
        # Still allow if magic bytes say image (handled by /media); keep strict here.
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")
    raw = await file.read()
    _check_size(raw)
    try:
        return await analysis_service.analyze_image_bytes(raw, source="upload")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Could not process image: {exc}") from exc


@router.post("/video", response_model=AnalysisResponse)
async def analyze_video(file: UploadFile = File(...)):
    if file.content_type and file.content_type not in VIDEO_CONTENT_TYPES and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")
    raw = await file.read()
    _check_size(raw)
    try:
        return await analysis_service.analyze_video_bytes(raw, source="upload")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Could not process video: {exc}") from exc


@router.post("/media", response_model=AnalysisResponse)
async def analyze_media(file: UploadFile = File(...)):
    """Auto-detect whether the upload is an image or a video, then process accordingly.

    Videos are cut into frames (analyzed as images + temporal cues) while the
    soundtrack is extracted and scored for synthetic/TTS artifacts in parallel.
    """
    raw = await file.read()
    _check_size(raw)
    try:
        return await analysis_service.analyze_media_bytes(
            raw,
            source="upload",
            content_type=file.content_type,
            filename=file.filename,
        )
    except MediaFetchError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Could not process media: {exc}") from exc


@router.post("/frames", response_model=AnalysisResponse)
async def analyze_frames(
    files: list[UploadFile] = File(...),
    platform: str | None = Form(default=None),
    source_url: str | None = Form(default=None),
    timestamps: str | None = Form(default=None),
):
    """Analyze a sequence of already-extracted frames as a silent video clip.

    Used by the extension when only canvas snapshots are available (e.g. YouTube
    MSE without MediaRecorder). Frame detectors + temporal consistency run;
    audio analysis is skipped (no soundtrack in the payload).
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one frame is required")

    images = []
    for upload in files[: settings.max_video_frames]:
        raw = await upload.read()
        _check_size(raw)
        try:
            images.append(media_utils.decode_image_bytes(raw))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=f"Could not decode frame: {exc}") from exc

    if not images:
        raise HTTPException(status_code=422, detail="No readable frames in upload")

    ts_list = None
    if timestamps:
        try:
            import json
            parsed = json.loads(timestamps)
            if isinstance(parsed, list):
                ts_list = [float(x) for x in parsed]
        except Exception:  # noqa: BLE001
            ts_list = None

    try:
        return await analysis_service.analyze_frame_sequence(
            images,
            source=source_url or "extension-frames",
            platform=platform or "unknown",
            timestamps=ts_list,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Could not process frames: {exc}") from exc


@router.post("/url", response_model=AnalysisResponse)
async def analyze_url(payload: UrlAnalyzeRequest):
    url = str(payload.url)
    platform = payload.platform_hint or media_utils.resolve_platform_hint(url)

    try:
        raw, content_type = await media_utils.fetch_url_bytes(url)
    except MediaFetchError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    _check_size(raw)

    try:
        return await analysis_service.analyze_media_bytes(
            raw,
            source=url,
            platform=platform,
            content_type=content_type,
            filename=url,
        )
    except MediaFetchError as exc:
        raise HTTPException(
            status_code=415,
            detail=(
                f"{exc} If this is a social media post page (not a direct media link), "
                "use the browser extension, which reads the actual media element from the page."
            ),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Could not process media from URL: {exc}") from exc
