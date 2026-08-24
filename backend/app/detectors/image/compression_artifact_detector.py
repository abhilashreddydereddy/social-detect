"""
CompressionArtifactDetector
-----------------------------
Uses Error Level Analysis (ELA): re-compress the image at a known JPEG
quality and diff against the original. Regions that were edited/generated
and then pasted into an otherwise normally-compressed photo tend to show
a different error level than the rest of the image (because they weren't
subject to the same compression history). Uniformly AI-generated images
also tend to show a flatter, more uniform ELA response than a genuine
multi-generation-compressed photo pulled off social media.

This is a classic, well-understood forensic technique -- intentionally
simple and explainable, complementary to the learned/semantic detectors.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
import io

from app.core.schemas import DetectorResult, Evidence, EvidenceCategory
from app.detectors.base import BaseDetector


class CompressionArtifactDetector(BaseDetector):
    name = "compression_ela"
    default_weight = 0.5
    supports_image = True
    supports_video = False  # video frames are already post-codec; ELA is unreliable there

    def analyze_image(self, image: np.ndarray) -> DetectorResult:
        try:
            pil_img = Image.fromarray(image).convert("RGB")
            buf = io.BytesIO()
            pil_img.save(buf, "JPEG", quality=90)
            buf.seek(0)
            recompressed = np.array(Image.open(buf).convert("RGB"))

            diff = cv2.absdiff(image, recompressed).astype(np.float32)
            diff_gray = diff.mean(axis=2)

            # Split into blocks, compare ELA intensity variance across blocks.
            step = 32
            block_means = []
            for y in range(0, diff_gray.shape[0] - step, step):
                for x in range(0, diff_gray.shape[1] - step, step):
                    block_means.append(diff_gray[y:y + step, x:x + step].mean())
            block_means = np.array(block_means) if block_means else np.array([0.0])

            overall_level = float(diff_gray.mean())
            block_std = float(block_means.std())
            block_mean = float(block_means.mean() + 1e-6)
            local_inconsistency = float(block_std / block_mean)

            # Very low overall ELA response + very low block-to-block variance
            # is typical of images that never went through an authentic
            # multi-generation JPEG history (e.g. a single fresh AI export).
            flatness_score = float(np.clip(1.0 - (overall_level / 8.0), 0.0, 1.0))
            flatness_score *= float(np.clip(1.0 - local_inconsistency, 0.0, 1.0))

            # A few blocks with dramatically higher ELA than the rest can
            # indicate a locally edited/inpainted region.
            spike_score = float(np.clip((local_inconsistency - 0.8) / 2.0, 0.0, 1.0))

            ai_probability = float(np.clip(0.5 * flatness_score + 0.5 * spike_score, 0.0, 1.0))
            confidence = float(np.clip(abs(ai_probability - 0.5) * 1.5, 0.05, 0.8))

            evidence = []
            if flatness_score > 0.5:
                evidence.append(Evidence(
                    category=EvidenceCategory.compression_artifact,
                    summary=(
                        "Error Level Analysis shows an unusually flat, low response across "
                        "the whole image, consistent with a single fresh export rather than "
                        "an image with an authentic multi-generation compression history."
                    ),
                    score=flatness_score,
                    weight=0.5,
                    detector=self.name,
                ))
            if spike_score > 0.4:
                evidence.append(Evidence(
                    category=EvidenceCategory.compression_artifact,
                    summary=(
                        "Localized regions show a compression error level that differs "
                        "sharply from the rest of the image, which can indicate a "
                        "pasted, inpainted, or locally regenerated region."
                    ),
                    score=spike_score,
                    weight=0.5,
                    detector=self.name,
                ))
            if not evidence:
                evidence.append(Evidence(
                    category=EvidenceCategory.compression_artifact,
                    summary="Compression error levels look consistent with normal JPEG re-encoding history.",
                    score=ai_probability,
                    weight=0.2,
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
