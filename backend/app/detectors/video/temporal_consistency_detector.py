"""
TemporalConsistencyDetector
------------------------------
Frame-to-frame deepfake/AI-video artifacts often show up as flicker: subtle
inconsistencies in texture, identity, or lighting that a per-frame image
detector alone would miss, but which stand out when frames are compared
over time. This is a simplified stand-in for full spatio-temporal models
like VideoMAE / TimeSformer -- same interface, so a real transformer-based
video classifier can be dropped in later without touching the API/service
layer (see the class docstring in detectors/base.py for the contract).
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from app.core.schemas import DetectorResult, Evidence, EvidenceCategory
from app.detectors.base import BaseDetector


class TemporalConsistencyDetector(BaseDetector):
    name = "temporal_consistency"
    default_weight = 0.65
    supports_image = False
    supports_video = True

    def analyze_image(self, image: np.ndarray) -> DetectorResult:
        # Not meaningful on a single frame.
        return self.safe_result("temporal_consistency requires multiple frames")

    def analyze_video_frames(self, frames: List[np.ndarray], timestamps: List[float]) -> List[DetectorResult]:
        if len(frames) < 2:
            return [self.safe_result("Not enough frames for temporal analysis")]

        try:
            grays = [cv2.cvtColor(cv2.resize(f, (256, 256)), cv2.COLOR_RGB2GRAY) for f in frames]
            flicker_scores = []
            for i in range(1, len(grays)):
                diff = cv2.absdiff(grays[i], grays[i - 1]).astype(np.float32)
                # High-frequency flicker: local variance of the frame delta.
                # Natural motion produces smooth, spatially coherent deltas;
                # per-frame-regenerated content often produces noisy,
                # spatially incoherent deltas even in static regions.
                flicker_scores.append(float(diff.std()))

            flicker_scores = np.array(flicker_scores)
            baseline = np.percentile(flicker_scores, 25)  # robust "static region" baseline
            volatility = float(flicker_scores.std() / (flicker_scores.mean() + 1e-6))

            flicker_score = float(np.clip((baseline - 6.0) / 20.0, 0.0, 1.0))
            volatility_score = float(np.clip((volatility - 0.3) / 1.0, 0.0, 1.0))
            ai_probability = float(np.clip(0.6 * flicker_score + 0.4 * volatility_score, 0.0, 1.0))
            confidence = float(np.clip(abs(ai_probability - 0.5) * 1.6, 0.05, 0.85))

            evidence = []
            if flicker_score > 0.4:
                evidence.append(Evidence(
                    category=EvidenceCategory.temporal_inconsistency,
                    summary=(
                        "Frame-to-frame differences remain elevated even in low-motion "
                        "segments, suggesting per-frame regeneration rather than a "
                        "continuous camera capture."
                    ),
                    score=flicker_score, weight=0.6, detector=self.name,
                ))
            if volatility_score > 0.3:
                evidence.append(Evidence(
                    category=EvidenceCategory.temporal_inconsistency,
                    summary="Motion magnitude fluctuates unevenly across the clip, inconsistent with smooth natural motion.",
                    score=volatility_score, weight=0.4, detector=self.name,
                ))
            if not evidence:
                evidence.append(Evidence(
                    category=EvidenceCategory.temporal_inconsistency,
                    summary="Frame-to-frame motion is smooth and consistent with natural video capture.",
                    score=ai_probability, weight=0.2, detector=self.name,
                ))

            result = DetectorResult(
                detector=self.name,
                ai_probability=ai_probability,
                confidence=confidence,
                evidence=evidence,
            )
            # Same aggregate result applies to the whole clip; per-frame image
            # detectors provide the finer-grained frame_results breakdown.
            return [result]
        except Exception as exc:  # noqa: BLE001
            return [self.safe_result(str(exc))]
