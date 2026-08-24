"""
Detector registry.

This is the ONE place you touch to add, remove, or swap a detector.
Each detector is self-contained (app/detectors/image/*.py or
app/detectors/video/*.py) and implements the BaseDetector interface.

MFAD-Net (`mfad_net`) is registered for both image and video. It reports
`available=False` until a trained checkpoint exists under
`backend/models/mfad_net/` or `training/exports/mfad_net/`.
"""
from __future__ import annotations

from typing import List

from app.detectors.audio.synthetic_speech_detector import SyntheticSpeechDetector
from app.detectors.base import BaseDetector
from app.detectors.image.frequency_artifact_detector import FrequencyArtifactDetector
from app.detectors.image.noise_residual_detector import NoiseResidualDetector
from app.detectors.image.compression_artifact_detector import CompressionArtifactDetector
from app.detectors.image.metadata_detector import MetadataDetector
from app.detectors.image.clip_semantic_detector import ClipSemanticDetector
from app.detectors.mfad.mfad_net_detector import MFADNetDetector
from app.detectors.multimodal_ensemble_detector import MultimodalEnsembleDetector
from app.detectors.video.temporal_consistency_detector import TemporalConsistencyDetector

# Singleton instances -- lazy-loaded on first use, so app startup stays fast
# and detectors with optional heavy deps don't block the whole service.
_MFAD = MFADNetDetector()

IMAGE_DETECTORS: List[BaseDetector] = [
    _MFAD,
    MultimodalEnsembleDetector(),
    FrequencyArtifactDetector(),
    NoiseResidualDetector(),
    CompressionArtifactDetector(),
    MetadataDetector(),
    ClipSemanticDetector(),
]

VIDEO_DETECTORS: List[BaseDetector] = [
    _MFAD,
    MultimodalEnsembleDetector(),
    FrequencyArtifactDetector(),   # reused per-frame
    NoiseResidualDetector(),       # reused per-frame
    TemporalConsistencyDetector(),
    ClipSemanticDetector(),        # reused per-frame
]

AUDIO_DETECTORS: List[BaseDetector] = [
    SyntheticSpeechDetector(),
]


def active_image_detectors() -> List[BaseDetector]:
    return [d for d in IMAGE_DETECTORS if d.available]


def active_video_detectors() -> List[BaseDetector]:
    return [d for d in VIDEO_DETECTORS if d.available]


def active_audio_detectors() -> List[BaseDetector]:
    return [d for d in AUDIO_DETECTORS if d.available]


def registry_status() -> List[dict]:
    seen = {}
    for d in IMAGE_DETECTORS + VIDEO_DETECTORS + AUDIO_DETECTORS:
        if d.name in seen:
            continue
        seen[d.name] = {
            "name": d.name,
            "available": d.available,
            "supports_image": d.supports_image,
            "supports_video": d.supports_video,
            "supports_audio": getattr(d, "supports_audio", False),
            "default_weight": d.default_weight,
        }
    return list(seen.values())
