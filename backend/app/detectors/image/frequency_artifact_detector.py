"""
FrequencyArtifactDetector
--------------------------
GANs and many diffusion decoders use transposed convolutions / repeated
up-sampling, which tends to leave periodic, grid-like fingerprints in the
frequency domain (visible as symmetric peaks off the DC component in the
2D FFT magnitude spectrum, and as unusually smooth high-frequency roll-off
compared to real camera sensor output).

This is a lightweight, dependency-free heuristic (numpy only) and is meant
as ONE signal among many, not a standalone verdict -- exactly the kind of
detector that should be replaceable by a trained model (e.g. a spectral
CNN, DIRE, or a UniversalFakeDetect checkpoint) without changing its
interface.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.core.schemas import DetectorResult, Evidence, EvidenceCategory
from app.detectors.base import BaseDetector


class FrequencyArtifactDetector(BaseDetector):
    name = "frequency_artifact_fft"
    default_weight = 0.6
    supports_image = True
    supports_video = True

    def analyze_image(self, image: np.ndarray) -> DetectorResult:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
            gray = cv2.resize(gray, (256, 256))

            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude = np.log(np.abs(fshift) + 1e-8)

            h, w = magnitude.shape
            cy, cx = h // 2, w // 2

            # Radial energy profile: real photos decay smoothly; synthetic
            # upsampling artifacts often show a bump in the mid-high band
            # and/or strong periodic peaks away from the axes.
            y, x = np.ogrid[:h, :w]
            radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(np.int32)
            max_r = min(cy, cx)
            radial_mean = np.array([
                magnitude[radius == r].mean() if np.any(radius == r) else 0.0
                for r in range(max_r)
            ])
            radial_mean = radial_mean / (radial_mean.max() + 1e-8)

            low = radial_mean[: int(max_r * 0.2)].mean()
            mid = radial_mean[int(max_r * 0.2): int(max_r * 0.6)].mean()
            high = radial_mean[int(max_r * 0.6):].mean()

            # Real photographic sensor noise -> high band stays reasonably
            # energetic (natural noise floor). Heavy denoising / generative
            # decoders often over-smooth the high band relative to mid band.
            high_to_mid = high / (mid + 1e-8)

            # Detect discrete grid peaks (periodic checkerboard artifacts)
            mask = magnitude > (magnitude.mean() + 3.5 * magnitude.std())
            mask[cy - 2:cy + 2, cx - 2:cx + 2] = False  # ignore DC spike
            peak_ratio = mask.sum() / mask.size

            # Heuristic scoring, calibrated to sit near 0.5 (uninformative)
            # when signals are ambiguous, and move toward 1.0 only when
            # multiple cues agree.
            smoothness_score = float(np.clip(1.0 - high_to_mid * 4.0, 0.0, 1.0))
            peak_score = float(np.clip(peak_ratio * 400.0, 0.0, 1.0))

            ai_probability = float(np.clip(0.5 * smoothness_score + 0.5 * peak_score, 0.0, 1.0))
            # Confidence reflects how far the signal is from the noisy midpoint,
            # not correctness -- we genuinely have less information near 0.5.
            confidence = float(np.clip(abs(ai_probability - 0.5) * 1.8, 0.05, 0.9))

            evidence = []
            if smoothness_score > 0.55:
                evidence.append(Evidence(
                    category=EvidenceCategory.frequency_artifact,
                    summary=(
                        "High-frequency spectral energy is unusually low relative to "
                        "mid-frequency content, consistent with generative up-sampling "
                        "or heavy denoising rather than raw sensor capture."
                    ),
                    score=smoothness_score,
                    weight=0.5,
                    detector=self.name,
                ))
            if peak_score > 0.4:
                evidence.append(Evidence(
                    category=EvidenceCategory.texture_repetition,
                    summary=(
                        "Detected periodic, grid-like peaks in the frequency spectrum, "
                        "a common fingerprint of transposed-convolution up-sampling used "
                        "by many GAN/diffusion decoders."
                    ),
                    score=peak_score,
                    weight=0.5,
                    detector=self.name,
                ))
            if not evidence:
                evidence.append(Evidence(
                    category=EvidenceCategory.frequency_artifact,
                    summary="No strong periodic or spectral-smoothing artifacts detected.",
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
        except Exception as exc:  # noqa: BLE001 - detectors must never crash the pipeline
            return self.safe_result(str(exc))
