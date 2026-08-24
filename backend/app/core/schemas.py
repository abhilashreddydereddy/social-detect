"""
Shared response/request schemas for the Social Detect API.

Design principle: the API NEVER asserts ground truth ("this is fake").
It always returns a probability + confidence + evidence, so the caller
(extension overlay or dashboard) can present nuance instead of a verdict.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl


class MediaType(str, Enum):
    image = "image"
    video = "video"


class EvidenceCategory(str, Enum):
    frequency_artifact = "frequency_artifact"      # GAN/diffusion frequency-domain fingerprints
    texture_repetition = "texture_repetition"       # repeated/tiled texture patches
    noise_residual = "noise_residual"                # missing/abnormal camera sensor (PRNU-like) noise
    compression_artifact = "compression_artifact"    # ELA / double-JPEG-compression irregularities
    lighting_inconsistency = "lighting_inconsistency"
    metadata = "metadata"                            # EXIF / container metadata anomalies
    semantic = "semantic"                            # CLIP/ViT semantic-embedding classifier
    temporal_inconsistency = "temporal_inconsistency"  # video-only: flicker between frames
    face_artifact = "face_artifact"                  # video/image: face-forensics style cues
    audio_artifact = "audio_artifact"                # video soundtrack: TTS / vocoder / synthetic speech cues


class Evidence(BaseModel):
    category: EvidenceCategory
    summary: str = Field(..., description="Human-readable explanation of what was found")
    score: float = Field(..., ge=0.0, le=1.0, description="This signal's own AI-likelihood, 0-1")
    weight: float = Field(..., ge=0.0, le=1.0, description="How much this signal contributed to the fused score")
    detector: str = Field(..., description="Name of the detector module that produced this evidence")


class DetectorResult(BaseModel):
    """Raw output of a single pluggable detector, before fusion."""
    detector: str
    ai_probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: List[Evidence] = []
    error: Optional[str] = None  # populated if the detector failed/was unavailable, never raises


class Classification(str, Enum):
    likely_authentic = "Likely Authentic"
    possibly_manipulated = "Possibly Manipulated"
    likely_ai_generated = "Likely AI Generated"
    inconclusive = "Inconclusive"


class FrameResult(BaseModel):
    """Per-frame breakdown for video analysis."""
    frame_index: int
    timestamp_seconds: float
    ai_probability: float
    confidence: float
    thumbnail_base64: Optional[str] = None


class AudioResult(BaseModel):
    """Summary of the parallel soundtrack authenticity probe for video."""
    available: bool = True
    ai_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    sample_rate: Optional[int] = None
    duration_seconds: Optional[float] = None
    detector: Optional[str] = None
    error: Optional[str] = None


class AnalysisResponse(BaseModel):
    request_id: str
    media_type: MediaType
    source: str = Field(..., description="'upload' or the originating URL")
    ai_probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    classification: Classification
    evidence: List[Evidence]
    detector_results: List[DetectorResult]
    frame_results: Optional[List[FrameResult]] = None
    audio_result: Optional[AudioResult] = None
    metadata: dict = {}
    processing_time_ms: int
    disclaimer: str = (
        "This is a probabilistic estimate produced by automated heuristics and models. "
        "It is not proof of authenticity or manipulation."
    )


class UrlAnalyzeRequest(BaseModel):
    url: HttpUrl
    platform_hint: Optional[str] = Field(
        None, description="e.g. 'instagram', 'x', 'reddit', 'tiktok', 'youtube', 'facebook'"
    )


class StatusResponse(BaseModel):
    status: str
    version: str
    detectors: List[dict]
    gpu_available: bool
