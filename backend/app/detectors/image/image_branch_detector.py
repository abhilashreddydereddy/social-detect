"""
Image-branch detector — CIFake-trained (or similar) single-image classifier.

Loads a checkpoint exported by:
  python -m training.image_branch.scripts.train --config training/image_branch/configs/cifake.yaml

Images: scored directly.
Videos: sample frames → same image model → aggregate (mean) into one clip score
so fusion is not overweighted by N per-frame votes.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from app.config import settings
from app.core.schemas import DetectorResult, Evidence, EvidenceCategory
from app.detectors.base import BaseDetector

_REPO_ROOT = Path(__file__).resolve().parents[4]

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ImageBranchDetector(BaseDetector):
    name = "image_branch_cifake"
    default_weight = 1.0
    learned = True
    supports_image = True
    supports_video = True
    supports_audio = False

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        super().__init__()
        self._checkpoint_path = checkpoint_path or settings.image_model_checkpoint_path
        self._model = None
        self._device = "cpu"
        self._available = False
        self._image_size = 224
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        if self._loaded:
            return self._available
        # Cheap existence check before full load
        return self._resolve_checkpoint() is not None

    def _resolve_checkpoint(self) -> Path | None:
        candidates: list[Path] = []
        if self._checkpoint_path:
            candidates.append(Path(self._checkpoint_path))
        candidates.extend([
            _REPO_ROOT / "backend" / "models" / "image_branch" / "cifake_best.pth",
            Path(__file__).resolve().parents[3] / "models" / "image_branch" / "cifake_best.pth",
            _REPO_ROOT / "training" / "exports" / "image_branch" / "cifake" / "best_model.pth",
        ])
        return next((p for p in candidates if p.exists()), None)

    def load(self) -> None:
        self._loaded = True
        try:
            import torch
            from app.detectors.image.image_classifier import (
                ImageClassifier,
                ImageClassifierConfig,
            )

            ckpt_file = self._resolve_checkpoint()
            if ckpt_file is None:
                self._available = False
                self._load_error = "No image-branch checkpoint found"
                return

            self._device = "cuda" if torch.cuda.is_available() and settings.device != "cpu" else "cpu"
            ckpt = torch.load(ckpt_file, map_location=self._device, weights_only=False)
            cfg_payload = dict(ckpt.get("model_config") or {})
            cfg_payload["pretrained"] = False  # ImageNet init is unused; weights come from ckpt
            cfg = ImageClassifierConfig.from_dict(cfg_payload)
            model = ImageClassifier(cfg)
            model.load_state_dict(ckpt["state_dict"])
            model.to(self._device)
            model.eval()
            self._model = model
            self._image_size = int(ckpt.get("image_size", 224))
            self._checkpoint_path = str(ckpt_file)
            self._available = True
        except Exception as exc:  # noqa: BLE001
            self._available = False
            self._model = None
            self._load_error = str(exc)

    def _prep(self, image: np.ndarray):
        import torch
        from PIL import Image as PILImage

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        pil = PILImage.fromarray(image).convert("RGB")
        # Match the training loader's early downscale before the final resize.
        if max(pil.size) > 256:
            pil.thumbnail((256, 256), PILImage.BILINEAR)
        pil = pil.resize((self._image_size, self._image_size), PILImage.BICUBIC)
        arr = np.asarray(pil).astype(np.float32) / 255.0
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self._device)

    def _score_batch(self, images: List[np.ndarray]) -> list[float]:
        import torch

        if not images:
            return []
        batch = torch.cat([self._prep(img) for img in images], dim=0)
        with torch.no_grad():
            out = self._model(batch)
        return [float(x) for x in out["p_fake"].detach().cpu().tolist()]

    def analyze_image(self, image: np.ndarray) -> DetectorResult:
        try:
            self.ensure_loaded()
            if self._model is None:
                return self.safe_result(self._load_error or "Image-branch checkpoint not loaded")
            p_fake = self._score_batch([image])[0]
            conf = float(min(0.95, abs(p_fake - 0.5) * 1.8 + 0.2))
            return DetectorResult(
                detector=self.name,
                ai_probability=round(p_fake, 4),
                confidence=round(conf, 4),
                evidence=[
                    Evidence(
                        category=EvidenceCategory.semantic,
                        summary=(
                            "CIFake-trained image classifier scored this still. "
                            f"P(AI/fake)={p_fake:.3f}."
                        ),
                        score=p_fake,
                        weight=self.default_weight,
                        detector=self.name,
                    )
                ],
            )
        except Exception as exc:  # noqa: BLE001
            return self.safe_result(str(exc))

    def analyze_video_frames(self, frames: List[np.ndarray], timestamps: List[float]) -> List[DetectorResult]:
        """Cut/sample frames → image model → one aggregated clip score."""
        try:
            self.ensure_loaded()
            if self._model is None:
                return [self.safe_result(self._load_error or "Image-branch checkpoint not loaded")]
            if not frames:
                return [self.safe_result("No frames")]

            max_frames = max(1, min(len(frames), int(getattr(settings, "ensemble_video_frame_samples", 12))))
            idxs = np.linspace(0, len(frames) - 1, num=max_frames).astype(int)
            sampled = [frames[i] for i in idxs]
            probs = self._score_batch(sampled)
            mean_p = float(np.mean(probs))
            spread = float(np.std(probs)) if len(probs) > 1 else 0.0
            conf = float(np.clip(abs(mean_p - 0.5) * 1.8 + 0.15 - spread * 0.4, 0.05, 0.95))

            mapped = [mean_p] * len(frames)
            for idx, prob in zip(idxs, probs):
                mapped[int(idx)] = float(prob)

            return [
                DetectorResult(
                    detector=self.name,
                    ai_probability=round(mean_p, 4),
                    confidence=round(conf, 4),
                    frame_scores=[round(p, 4) for p in mapped],
                    evidence=[
                        Evidence(
                            category=EvidenceCategory.temporal_inconsistency,
                            summary=(
                                f"Video scored by sampling {len(sampled)} frames through the "
                                f"CIFake image model (mean P(fake)={mean_p:.3f}, "
                                f"frame spread={spread:.3f})."
                            ),
                            score=mean_p,
                            weight=self.default_weight,
                            detector=self.name,
                        )
                    ],
                )
            ]
        except Exception as exc:  # noqa: BLE001
            return [self.safe_result(str(exc))]
