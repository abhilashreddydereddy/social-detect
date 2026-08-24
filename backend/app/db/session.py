"""
Database session handling.

Kept deliberately optional/best-effort: `Load Unpacked` local development
and the dashboard demo flow should work even if PostgreSQL isn't running
(e.g. first-time setup before `docker compose up db`). Analysis history
persistence is a nice-to-have, not a hard dependency of the detection
pipeline -- so failures here are logged and swallowed, never surfaced as
a 500 to the extension/dashboard caller.
"""
from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.db.models import Base

logger = logging.getLogger("social_detect.db")

engine = create_async_engine(settings.database_url, pool_pre_ping=True, future=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

_db_available = True


async def init_db() -> None:
    global _db_available
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        _db_available = True
        logger.info("Database initialized.")
    except Exception as exc:  # noqa: BLE001
        _db_available = False
        logger.warning("Database unavailable, history persistence disabled: %s", exc)


async def save_record(record) -> None:
    if not _db_available:
        return
    try:
        async with SessionLocal() as session:
            session.add(record)
            await session.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to persist analysis record: %s", exc)
