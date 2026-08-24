from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
from app.core import media_utils
from app.core.media_utils import MediaFetchError
from app.core.schemas import AnalysisResponse, UrlAnalyzeRequest
from app.services import analysis_service

router = APIRouter(prefix="/analyze", tags=["analyze"])

IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}
VIDEO_CONTENT_TYPES = {"video/mp4", "video/webm", "video/quicktime", "video/x-matroska"}
MAX_BYTES = settings.max_upload_mb * 1024 * 1024


def _check_size(raw: bytes) -> None:
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb}MB limit")


@router.post("/image", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    if file.content_type not in IMAGE_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")
    raw = await file.read()
    _check_size(raw)
    try:
        return await analysis_service.analyze_image_bytes(raw, source="upload")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Could not process image: {exc}") from exc


@router.post("/video", response_model=AnalysisResponse)
async def analyze_video(file: UploadFile = File(...)):
    if file.content_type not in VIDEO_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")
    raw = await file.read()
    _check_size(raw)
    try:
        return await analysis_service.analyze_video_bytes(raw, source="upload")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Could not process video: {exc}") from exc


@router.post("/url", response_model=AnalysisResponse)
async def analyze_url(payload: UrlAnalyzeRequest):
    url = str(payload.url)
    platform = payload.platform_hint or media_utils.resolve_platform_hint(url)

    try:
        raw, content_type = await media_utils.fetch_url_bytes(url)
    except MediaFetchError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    _check_size(raw)

    if content_type in VIDEO_CONTENT_TYPES or url.lower().split("?")[0].endswith((".mp4", ".webm", ".mov")):
        try:
            return await analysis_service.analyze_video_bytes(raw, source=url, platform=platform)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=f"Could not process video from URL: {exc}") from exc

    if content_type in IMAGE_CONTENT_TYPES or url.lower().split("?")[0].endswith((".jpg", ".jpeg", ".png", ".webp")):
        try:
            return await analysis_service.analyze_image_bytes(raw, source=url, platform=platform)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=f"Could not process image from URL: {exc}") from exc

    raise HTTPException(
        status_code=415,
        detail=(
            f"Could not determine media type from URL (content-type: '{content_type}'). "
            "If this is a social media post page (not a direct media link), use the "
            "browser extension, which reads the actual media element from the page."
        ),
    )
