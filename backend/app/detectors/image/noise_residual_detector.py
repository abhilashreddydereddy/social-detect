"""
NoiseResidualDetector
-----------------------
Real camera sensors leave a characteristic high-frequency noise residual
(shot noise, read noise, and a fixed-pattern component related to PRNU --
Photo Response Non-Uniformity) that survives even after JPEG compression.
AI-generated images, and images that have been heavily inpainted/denoised,
typically have a residual that is either too smooth (near-zero variance,
generative decoders don't model sensor noise) or too uniform/synthetic
(denoising-diffusion artifacts look statistically different from Gaussian
sensor noise).

This module extracts a noise residual with a high-pass (wavelet-like)
filter and scores how "camera-like" its local statistics are. It's a
simplified stand-in for a full PRNU correlation pipeline, kept dependency
light (opencv + numpy) and designed to be swapped for a trained
noiseprint / PRNU-correlation model later.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.core.schemas import DetectorResult, Evidence, EvidenceCategory
from app.detectors.base import BaseDetector


class NoiseResidualDetector(BaseDetector):
    name = "sensor_noise_residual"
    default_weight = 0.7
    supports_image = True
    supports_video = True

    def _residual(self, gray: np.ndarray) -> np.ndarray:
        denoised = cv2.medianBlur(gray, 3)
        return gray.astype(np.float32) - denoised.astype(np.float32)

    def analyze_image(self, image: np.ndarray) -> DetectorResult:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, (512, 512))
            residual = self._residual(gray)

            # Tile the residual and compute local variance per tile. Real
            # sensor noise is fairly uniform in variance across flat regions
            # but present everywhere; generated images often show patches
            # of near-zero residual variance (over-smoothed regions) next
            # to patches with abnormally high variance (hallucinated detail).
            tiles = []
            step = 32
            for y in range(0, residual.shape[0] - step, step):
                for x in range(0, residual.shape[1] - step, step):
                    tile = residual[y:y + step, x:x + step]
                    tiles.append(tile.var())
            tiles = np.array(tiles)

            mean_var = float(tiles.mean())
            cv_of_var = float(tiles.std() / (tiles.mean() + 1e-6))  # coefficient of variation
            near_zero_fraction = float((tiles < 0.5).mean())

            # Low absolute noise energy -> smoothed/generated look.
            low_noise_score = float(np.clip(1.0 - (mean_var / 6.0), 0.0, 1.0))
            # High patchiness (some tiles ~0, others high) -> inconsistent, synthetic.
            patchiness_score = float(np.clip((cv_of_var - 0.6) / 1.5, 0.0, 1.0))

            ai_probability = float(np.clip(
                0.55 * low_noise_score + 0.25 * patchiness_score + 0.2 * near_zero_fraction,
                0.0, 1.0,
            ))
            confidence = float(np.clip(abs(ai_probability - 0.5) * 1.7, 0.05, 0.9))

            evidence = []
            if low_noise_score > 0.55:
                evidence.append(Evidence(
                    category=EvidenceCategory.noise_residual,
                    summary=(
                        "Overall noise-residual energy is much lower than typical camera "
                        "sensor output, suggesting the image may lack authentic sensor "
                        "noise (consistent with synthetic generation or heavy smoothing)."
                    ),
                    score=low_noise_score,
                    weight=0.6,
                    detector=self.name,
                ))
            if patchiness_score > 0.4:
                evidence.append(Evidence(
                    category=EvidenceCategory.noise_residual,
                    summary=(
                        "Noise variance is inconsistent across image regions, which can "
                        "indicate localized generation/inpainting rather than a uniform "
                        "capture process."
                    ),
                    score=patchiness_score,
                    weight=0.4,
                    detector=self.name,
                ))
            if not evidence:
                evidence.append(Evidence(
                    category=EvidenceCategory.noise_residual,
                    summary="Noise residual is consistent with typical camera sensor output.",
                    score=ai_probability,
                    weight=0.3,
                    detector=self.name,
                ))

            return DetectorResult(
                detector=self.name,
                ai_probability=ai_probability,
                confidence=confidence,
                evidence=evidence,
            )
        except Exception as exc:  # noqa: BLE001
            return self.safe_result(str(exc))
