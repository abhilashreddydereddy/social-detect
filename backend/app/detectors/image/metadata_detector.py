"""
MetadataDetector
-------------------
Inspects EXIF/XMP metadata. This detector operates on raw file bytes
(passed alongside the decoded numpy array by the analysis service) rather
than pixel content. Absence of camera metadata is weak evidence on its
own (social platforms strip EXIF on upload), so this detector is
deliberately given a lower default_weight and a capped confidence -- it
mainly contributes when metadata IS present and reveals something
concrete, like known AI-tool signatures (e.g. "Software: Midjourney",
"C2PA: trained-algorithmic-media") or internal inconsistencies
(creation vs modification timestamps, resolution mismatches).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image, ExifTags

from app.core.schemas import DetectorResult, Evidence, EvidenceCategory
from app.detectors.base import BaseDetector

AI_TOOL_SIGNATURES = [
    "midjourney", "dall-e", "dalle", "stable diffusion", "stability.ai",
    "runwayml", "firefly", "adobe firefly", "leonardo.ai", "ideogram",
    "flux", "comfyui", "automatic1111", "sora", "veo", "kling", "gemini imagen",
    "imagen", "sdxl", "novelai",
]


class MetadataDetector(BaseDetector):
    name = "metadata_inspection"
    default_weight = 0.35
    supports_image = True
    supports_video = False

    def __init__(self):
        super().__init__()
        self._pending_bytes: Optional[bytes] = None

    def set_raw_bytes(self, raw: bytes) -> None:
        """The analysis service calls this before analyze_image since this
        detector needs the original file bytes, not just decoded pixels."""
        self._pending_bytes = raw

    def analyze_image(self, image: np.ndarray) -> DetectorResult:
        raw = self._pending_bytes
        self._pending_bytes = None
        try:
            evidence = []
            ai_probability = 0.5
            confidence = 0.1  # metadata alone rarely earns high confidence

            if raw is None:
                return DetectorResult(
                    detector=self.name,
                    ai_probability=0.5,
                    confidence=0.0,
                    evidence=[Evidence(
                        category=EvidenceCategory.metadata,
                        summary="No raw file bytes available for metadata inspection.",
                        score=0.5, weight=0.1, detector=self.name,
                    )],
                )

            import io
            pil_img = Image.open(io.BytesIO(raw))
            exif_raw = pil_img.getexif()
            exif = {ExifTags.TAGS.get(k, k): v for k, v in exif_raw.items()} if exif_raw else {}

            software = str(exif.get("Software", "")).lower()
            has_camera_make = bool(exif.get("Make"))
            has_camera_model = bool(exif.get("Model"))
            has_gps = bool(exif.get("GPSInfo"))
            has_datetime = bool(exif.get("DateTime"))

            matched_tool = next((t for t in AI_TOOL_SIGNATURES if t in software), None)

            if matched_tool:
                ai_probability = 0.93
                confidence = 0.85
                evidence.append(Evidence(
                    category=EvidenceCategory.metadata,
                    summary=f"Image metadata 'Software' field references a known AI generation tool ('{matched_tool}').",
                    score=0.93, weight=0.9, detector=self.name,
                ))
            elif not exif:
                # Very common for social-media re-encodes AND for AI exports.
                # Weak signal, kept low-weight and low-confidence deliberately.
                ai_probability = 0.55
                confidence = 0.15
                evidence.append(Evidence(
                    category=EvidenceCategory.metadata,
                    summary=(
                        "No EXIF metadata present. This is common both for images "
                        "re-encoded by social platforms and for AI-generated images, "
                        "so it is treated as weak evidence."
                    ),
                    score=0.55, weight=0.15, detector=self.name,
                ))
            elif has_camera_make and has_camera_model:
                ai_probability = 0.2
                confidence = 0.4
                evidence.append(Evidence(
                    category=EvidenceCategory.metadata,
                    summary=(
                        f"Metadata includes a camera make/model "
                        f"({exif.get('Make')} {exif.get('Model')}), consistent with a "
                        "real camera or phone capture."
                    ),
                    score=0.2, weight=0.4, detector=self.name,
                ))
            else:
                ai_probability = 0.5
                confidence = 0.1
                evidence.append(Evidence(
                    category=EvidenceCategory.metadata,
                    summary="Metadata present but inconclusive (no camera make/model or known AI tool signature).",
                    score=0.5, weight=0.15, detector=self.name,
                ))

            return DetectorResult(
                detector=self.name,
                ai_probability=ai_probability,
                confidence=confidence,
                evidence=evidence,
            )
        except Exception as exc:  # noqa: BLE001
            return self.safe_result(str(exc))
