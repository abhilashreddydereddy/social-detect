from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import cv2
import numpy as np

from app.config import settings
from app.core.schemas import DetectorResult, Evidence, EvidenceCategory
from app.detectors.base import BaseDetector
from app.detectors.image.compression_artifact_detector import CompressionArtifactDetector
from app.detectors.image.frequency_artifact_detector import FrequencyArtifactDetector
from app.detectors.image.metadata_detector import MetadataDetector
from app.detectors.image.noise_residual_detector import NoiseResidualDetector
from app.detectors.video.temporal_consistency_detector import TemporalConsistencyDetector


@dataclass
class BranchScore:
    name: str
    probability: float
    confidence: float
    evidence: list[Evidence]


class MultimodalEnsembleDetector(BaseDetector):
    """Higher-level detector that fuses complementary cues across branches.

    This is the repo's MVP "real model" layer: not a single trained network,
    but a multimodal ensemble that behaves much closer to one by combining:
      - whole-image forensic signals
      - crop-level signals for localized manipulations
      - optional face-region analysis
      - temporal consistency for videos

    It is intentionally structured so a future pretrained GRIP-UNINA / CLIP
    detector or a DeepfakeBench video model can be dropped into one branch
    without changing the API or the rest of the backend.
    """

    name = "social_multimodal_ensemble"
    default_weight = 0.3
    supports_image = True
    supports_video = True

    def __init__(self) -> None:
        super().__init__()
        self._face_cascade = None
        self._full_image_detectors = [
            FrequencyArtifactDetector(),
            NoiseResidualDetector(),
            CompressionArtifactDetector(),
            MetadataDetector(),
        ]
        self._temporal_detector = TemporalConsistencyDetector()

    @property
    def available(self) -> bool:
        return True

    def load(self) -> None:
        self._loaded = True
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
            if self._face_cascade.empty():
                self._face_cascade = None
        except Exception:
            self._face_cascade = None

    def analyze_image(self, image: np.ndarray) -> DetectorResult:
        try:
            branches = [
                self._score_whole_image(image),
                self._score_structural_crops(image),
            ]
            face_branch = self._score_faces(image)
            if face_branch is not None:
                branches.append(face_branch)

            ai_probability, confidence = self._fuse_branches(branches)
            evidence = self._summarize_branches(branches)

            return DetectorResult(
                detector=self.name,
                ai_probability=ai_probability,
                confidence=confidence,
                evidence=evidence,
            )
        except Exception as exc:  # noqa: BLE001
            return self.safe_result(str(exc))

    def analyze_video_frames(self, frames: List[np.ndarray], timestamps: List[float]) -> List[DetectorResult]:
        try:
            sampled_frames = self._uniform_sample(frames, settings.ensemble_video_frame_samples)
            frame_scores = [self.analyze_image(frame) for frame in sampled_frames]

            usable_frame_scores = [r for r in frame_scores if r.error is None]
            if usable_frame_scores:
                frame_probs = np.array([r.ai_probability for r in usable_frame_scores], dtype=np.float32)
                frame_confs = np.array([max(r.confidence, 0.05) for r in usable_frame_scores], dtype=np.float32)
                frame_prob = float(np.average(frame_probs, weights=frame_confs))
                frame_spread = float(np.std(frame_probs))
                frame_conf = float(np.clip(np.average(frame_confs) - frame_spread * 0.45, 0.0, 1.0))
                frame_evidence = self._collect_branch_evidence(usable_frame_scores)
            else:
                frame_prob = 0.5
                frame_conf = 0.0
                frame_evidence = []

            temporal_results = self._temporal_detector.analyze_video_frames(frames, timestamps)
            temporal_usable = [r for r in temporal_results if r.error is None]
            if temporal_usable:
                temporal = temporal_usable[0]
                video_prob, video_conf = self._fuse_modalities(
                    [(frame_prob, frame_conf, 0.65), (temporal.ai_probability, temporal.confidence, 0.35)]
                )
                evidence = self._dedupe_evidence(frame_evidence + temporal.evidence)
            else:
                video_prob, video_conf = frame_prob, frame_conf
                evidence = frame_evidence

            return [DetectorResult(
                detector=self.name,
                ai_probability=video_prob,
                confidence=video_conf,
                evidence=evidence[:5],
            )]
        except Exception as exc:  # noqa: BLE001
            return [self.safe_result(str(exc))]

    def _score_whole_image(self, image: np.ndarray) -> BranchScore:
        results = self._run_forensic_detectors(image)
        prob, conf = self._fuse_detector_results(
            results,
            {"frequency_artifact_fft": 0.3, "sensor_noise_residual": 0.35, "compression_ela": 0.25, "metadata_inspection": 0.10},
        )
        return BranchScore("whole_image", prob, conf, self._collect_branch_evidence(results))

    def _score_structural_crops(self, image: np.ndarray) -> BranchScore:
        crops = self._generate_crops(image)
        crop_results = [self._run_forensic_detectors(crop) for crop in crops]
        crop_scores = []
        crop_evidence: list[Evidence] = []
        for results in crop_results:
            prob, conf = self._fuse_detector_results(
                results,
                {"frequency_artifact_fft": 0.35, "sensor_noise_residual": 0.35, "compression_ela": 0.30},
            )
            crop_scores.append((prob, conf))
            crop_evidence.extend(self._collect_branch_evidence(results))

        if not crop_scores:
            return BranchScore("structural_crops", 0.5, 0.0, [])

        probs = np.array([p for p, _ in crop_scores], dtype=np.float32)
        confs = np.array([max(c, 0.05) for _, c in crop_scores], dtype=np.float32)
        prob = float(np.average(probs, weights=confs))
        localized_spike = float(np.clip(probs.max() - probs.mean(), 0.0, 1.0))
        conf = float(np.clip(np.average(confs) + localized_spike * 0.2, 0.0, 1.0))

        evidence = self._dedupe_evidence(crop_evidence)
        if localized_spike > 0.18:
            evidence.insert(0, Evidence(
                category=EvidenceCategory.face_artifact,
                summary="One or more local crops look substantially more suspicious than the rest of the image, consistent with partial edits or localized generation.",
                score=min(1.0, 0.5 + localized_spike),
                weight=0.35,
                detector=self.name,
            ))
        return BranchScore("structural_crops", prob, conf, evidence[:4])

    def _score_faces(self, image: np.ndarray) -> BranchScore | None:
        if not settings.ensemble_enable_face_branch or self._face_cascade is None:
            return None

        face_crops = self._extract_face_crops(image)
        if not face_crops:
            return None

        face_scores = []
        face_evidence: list[Evidence] = []
        for crop in face_crops[: settings.ensemble_face_max_crops]:
            results = self._run_forensic_detectors(crop)
            prob, conf = self._fuse_detector_results(
                results,
                {"frequency_artifact_fft": 0.25, "sensor_noise_residual": 0.45, "compression_ela": 0.30},
            )
            face_scores.append((prob, conf))
            face_evidence.extend(self._collect_branch_evidence(results))

        if not face_scores:
            return None

        probs = np.array([p for p, _ in face_scores], dtype=np.float32)
        confs = np.array([max(c, 0.05) for _, c in face_scores], dtype=np.float32)
        prob = float(np.average(probs, weights=confs))
        conf = float(np.clip(np.average(confs) + max(0.0, probs.max() - 0.5) * 0.12, 0.0, 1.0))
        evidence = self._dedupe_evidence(face_evidence)
        evidence.insert(0, Evidence(
            category=EvidenceCategory.face_artifact,
            summary="Face-region analysis was included in the final score to focus on the manipulation patterns most common in social-media deepfakes.",
            score=prob,
            weight=0.4,
            detector=self.name,
        ))
        return BranchScore("face_region", prob, conf, evidence[:4])

    def _run_forensic_detectors(self, image: np.ndarray) -> list[DetectorResult]:
        results = []
        for detector in self._full_image_detectors:
            detector.ensure_loaded()
            if isinstance(detector, MetadataDetector):
                detector.set_raw_bytes(None)
            results.append(detector.analyze_image(image))
        return results

    def _fuse_branches(self, branches: Iterable[BranchScore]) -> tuple[float, float]:
        weighted = []
        for branch in branches:
            weight = {"whole_image": 0.45, "structural_crops": 0.30, "face_region": 0.25}.get(branch.name, 0.2)
            weighted.append((branch.probability, branch.confidence, weight))
        return self._fuse_modalities(weighted)

    def _fuse_modalities(self, parts: list[tuple[float, float, float]]) -> tuple[float, float]:
        usable = [(p, c, w) for p, c, w in parts if c > 0.0]
        if not usable:
            return 0.5, 0.0
        weights = np.array([max(c, 0.05) * w for p, c, w in usable], dtype=np.float32)
        probs = np.array([p for p, c, w in usable], dtype=np.float32)
        prob = float(np.average(probs, weights=weights))
        spread = float(np.average((probs - prob) ** 2, weights=weights))
        conf = float(np.clip(np.average([c for _, c, _ in usable], weights=weights) - spread * 0.5 + 0.08, 0.0, 1.0))
        return prob, conf

    def _fuse_detector_results(self, results: list[DetectorResult], detector_weights: dict[str, float]) -> tuple[float, float]:
        usable = [r for r in results if r.error is None]
        if not usable:
            return 0.5, 0.0

        weights = np.array(
            [max(r.confidence, 0.05) * detector_weights.get(r.detector, 0.2) for r in usable],
            dtype=np.float32,
        )
        probs = np.array([r.ai_probability for r in usable], dtype=np.float32)
        prob = float(np.average(probs, weights=weights))
        spread = float(np.average((probs - prob) ** 2, weights=weights))
        conf = float(np.clip(np.average([r.confidence for r in usable], weights=weights) - spread * 0.6 + 0.05, 0.0, 1.0))
        return prob, conf

    def _generate_crops(self, image: np.ndarray) -> list[np.ndarray]:
        h, w = image.shape[:2]
        size = int(min(h, w) * 0.55)
        if size < 96:
            return []

        coords = [
            (0, 0),
            (0, max(0, w - size)),
            (max(0, h - size), 0),
            (max(0, h - size), max(0, w - size)),
            (max(0, (h - size) // 2), max(0, (w - size) // 2)),
        ]
        crops = []
        seen = set()
        for y, x in coords:
            key = (int(y), int(x), size)
            if key in seen:
                continue
            seen.add(key)
            crops.append(image[y:y + size, x:x + size])
        return crops

    def _extract_face_crops(self, image: np.ndarray) -> list[np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        crops = []
        h, w = image.shape[:2]
        for x, y, fw, fh in faces:
            pad_x = int(fw * 0.25)
            pad_y = int(fh * 0.25)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + fw + pad_x)
            y2 = min(h, y + fh + pad_y)
            crops.append(image[y1:y2, x1:x2])
        return crops

    def _collect_branch_evidence(self, results: Iterable[DetectorResult]) -> list[Evidence]:
        evidence: list[Evidence] = []
        for result in results:
            evidence.extend(result.evidence)
        return self._dedupe_evidence(evidence)

    def _summarize_branches(self, branches: Iterable[BranchScore]) -> list[Evidence]:
        evidence = []
        for branch in branches:
            evidence.extend(branch.evidence[:2])
        evidence = self._dedupe_evidence(evidence)
        evidence.sort(key=lambda e: e.weight * abs(e.score - 0.5), reverse=True)
        return evidence[:6]

    def _dedupe_evidence(self, evidence: list[Evidence]) -> list[Evidence]:
        seen = set()
        deduped = []
        for item in evidence:
            key = (item.category, item.summary)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _uniform_sample(self, frames: list[np.ndarray], target_count: int) -> list[np.ndarray]:
        if len(frames) <= target_count:
            return frames
        indices = np.linspace(0, len(frames) - 1, target_count).astype(int)
        return [frames[i] for i in indices]
