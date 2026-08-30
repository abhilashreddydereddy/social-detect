from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_analyze, routes_status
from app.config import settings
from app.db.session import init_db

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    # Load CIFake (and any other learned checkpoint) so the first dashboard /
    # extension request is not a 20–30s cold start.
    try:
        import asyncio
        from app.detectors.registry import IMAGE_DETECTORS

        def _warm() -> None:
            for detector in IMAGE_DETECTORS:
                if getattr(detector, "learned", False) and detector.available:
                    detector.ensure_loaded()

        await asyncio.get_running_loop().run_in_executor(None, _warm)
    except Exception:  # noqa: BLE001
        logging.getLogger("social_detect").warning("Learned-detector warmup skipped", exc_info=True)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=(
        "Modular AI media authenticity detection API. Returns probability, "
        "confidence, and explainable evidence -- never a definitive verdict."
    ),
    lifespan=lifespan,
)

# Chrome/Edge extensions call from an origin like chrome-extension://<id>.
# Regex allows any extension id in dev; lock this to your published
# extension's id in production via settings.cors_allow_origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_origin_regex=r"chrome-extension://.*|moz-extension://.*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_analyze.router)
app.include_router(routes_status.router)


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": settings.version,
        "docs": "/docs",
        "endpoints": [
            "/analyze/image",
            "/analyze/video",
            "/analyze/media",
            "/analyze/frames",
            "/analyze/url",
            "/status",
        ],
    }
