"""
Detector registry.

This is the ONE place you touch to add, remove, or swap a detector.
Each detector is self-contained (app/detectors/image/*.py or
app/detectors/video/*.py) and implements the BaseDetector interface.

Modality routing:
  - Image uploads → IMAGE_DETECTORS (includes CIFake-trained `image_branch_cifake`)
  - Video uploads → sample frames → VIDEO_DETECTORS (same image model aggregates
    frame scores) + AUDIO_DETECTORS on the soundtrack
  - Audio → AUDIO_DETECTORS (separate model; not CIFake)

`image_branch_cifake` / `mfad_net` report available=False until checkpoints exist.
"""
from __future__ import annotations

from typing import List

from app.detectors.audio.synthetic_speech_detector import SyntheticSpeechDetector
from app.detectors.audio.asvspoof5_detector import ASVSpoof5Detector
from app.detectors.base import BaseDetector
from app.detectors.image.frequency_artifact_detector import FrequencyArtifactDetector
from app.detectors.image.noise_residual_detector import NoiseResidualDetector
from app.detectors.image.compression_artifact_detector import CompressionArtifactDetector
from app.detectors.image.metadata_detector import MetadataDetector
from app.detectors.image.clip_semantic_detector import ClipSemanticDetector
from app.detectors.image.image_branch_detector import ImageBranchDetector
from app.detectors.mfad.mfad_net_detector import MFADNetDetector
from app.detectors.multimodal_ensemble_detector import MultimodalEnsembleDetector
from app.detectors.video.temporal_consistency_detector import TemporalConsistencyDetector

# Singleton instances -- lazy-loaded on first use, so app startup stays fast
# and detectors with optional heavy deps don't block the whole service.
_MFAD = MFADNetDetector()
_IMAGE_BRANCH = ImageBranchDetector()

IMAGE_DETECTORS: List[BaseDetector] = [
    _IMAGE_BRANCH,
    _MFAD,
    MultimodalEnsembleDetector(),
    FrequencyArtifactDetector(),
    NoiseResidualDetector(),
    CompressionArtifactDetector(),
    MetadataDetector(),
    ClipSemanticDetector(),
]

VIDEO_DETECTORS: List[BaseDetector] = [
    _IMAGE_BRANCH,                 # cut frames → CIFake image model → aggregate
    _MFAD,
    MultimodalEnsembleDetector(),
    FrequencyArtifactDetector(),   # reused per-frame
    NoiseResidualDetector(),       # reused per-frame
    TemporalConsistencyDetector(),
    ClipSemanticDetector(),        # reused per-frame
]

AUDIO_DETECTORS: List[BaseDetector] = [
    ASVSpoof5Detector(),
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
            "learned": bool(getattr(d, "learned", False)),
        }
    return list(seen.values())


def detector_weight_map() -> dict[str, float]:
    weights: dict[str, float] = {}
    for d in IMAGE_DETECTORS + VIDEO_DETECTORS + AUDIO_DETECTORS:
        weights[d.name] = float(d.default_weight)
    return weights
