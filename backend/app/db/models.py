from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class AnalysisRecord(Base):
    """Persisted record of every analysis run, used for the dashboard's
    'analysis history' view and for detector-comparison/tuning over time.
    """
    __tablename__ = "analysis_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    media_type: Mapped[str] = mapped_column(String(16))
    source: Mapped[str] = mapped_column(Text)
    platform: Mapped[str] = mapped_column(String(32), default="unknown")

    ai_probability: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    classification: Mapped[str] = mapped_column(String(32))

    detector_results: Mapped[dict] = mapped_column(JSON)
    evidence: Mapped[dict] = mapped_column(JSON)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)

    processing_time_ms: Mapped[int] = mapped_column(default=0)
