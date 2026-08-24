"""
Detector registry.

This is the ONE place you touch to add, remove, or swap a detector.
Each detector is self-contained (app/detectors/image/*.py or
app/detectors/video/*.py) and implements the BaseDetector interface.

To add a new model (e.g. a real UniversalFakeDetect / DIRE / XceptionNet /
FaceForensics++ / EfficientNet checkpoint, or a VideoMAE/TimeSformer video
model):
  1. Create app/detectors/image/my_detector.py or detectors/video/my_detector.py
     implementing BaseDetector (see base.py for the contract).
  2. Import and add an instance to IMAGE_DETECTORS / VIDEO_DETECTORS below.
  3. Nothing else changes -- the API, fusion, and dashboard/extension all
     consume detectors polymorphically.
"""
from __future__ import annotations

from typing import List

from app.detectors.base import BaseDetector
from app.detectors.image.frequency_artifact_detector import FrequencyArtifactDetector
from app.detectors.image.noise_residual_detector import NoiseResidualDetector
from app.detectors.image.compression_artifact_detector import CompressionArtifactDetector
from app.detectors.image.metadata_detector import MetadataDetector
from app.detectors.image.clip_semantic_detector import ClipSemanticDetector
from app.detectors.multimodal_ensemble_detector import MultimodalEnsembleDetector
from app.detectors.video.temporal_consistency_detector import TemporalConsistencyDetector

# Singleton instances -- lazy-loaded on first use, so app startup stays fast
# and detectors with optional heavy deps don't block the whole service.
IMAGE_DETECTORS: List[BaseDetector] = [
    MultimodalEnsembleDetector(),
    FrequencyArtifactDetector(),
    NoiseResidualDetector(),
    CompressionArtifactDetector(),
    MetadataDetector(),
    ClipSemanticDetector(),
]

VIDEO_DETECTORS: List[BaseDetector] = [
    MultimodalEnsembleDetector(),
    FrequencyArtifactDetector(),   # reused per-frame
    NoiseResidualDetector(),       # reused per-frame
    TemporalConsistencyDetector(),
    ClipSemanticDetector(),        # reused per-frame
]


def active_image_detectors() -> List[BaseDetector]:
    return [d for d in IMAGE_DETECTORS if d.available]


def active_video_detectors() -> List[BaseDetector]:
    return [d for d in VIDEO_DETECTORS if d.available]


def registry_status() -> List[dict]:
    seen = {}
    for d in IMAGE_DETECTORS + VIDEO_DETECTORS:
        if d.name in seen:
            continue
        seen[d.name] = {
            "name": d.name,
            "available": d.available,
            "supports_image": d.supports_image,
            "supports_video": d.supports_video,
            "default_weight": d.default_weight,
        }
    return list(seen.values())
