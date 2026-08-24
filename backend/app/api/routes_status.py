from __future__ import annotations

from fastapi import APIRouter

from app.config import settings
from app.core.schemas import StatusResponse
from app.detectors.registry import registry_status

router = APIRouter(tags=["status"])


@router.get("/status", response_model=StatusResponse)
async def status():
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass

    return StatusResponse(
        status="ok",
        version=settings.version,
        detectors=registry_status(),
        gpu_available=gpu_available,
    )
