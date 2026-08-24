"""
ClipSemanticDetector
-----------------------
Slot for a learned classifier (e.g. a CLIP-ViT backbone with a linear probe
trained on real-vs-AI datasets, in the spirit of "UniversalFakeDetect" /
Ojha et al.'s CLIP-based fake detector). This is the module you replace
with a real fine-tuned checkpoint for production accuracy; the heuristic
detectors (frequency/noise/compression) still cover you when this one is
unavailable.

Because torch/transformers + downloaded weights are optional/heavy, this
detector:
  1. Only imports torch/transformers inside `load()`, not at module import
     time, so the whole backend still boots without them installed.
  2. Sets `available = False` if imports or weight loading fail, and the
     registry silently skips unavailable detectors.
  3. Reads the checkpoint path/name from settings so it's swappable via
     config/env var without code changes -- e.g. point CLIP_MODEL_NAME at
     a HF Hub checkpoint of a UniversalFakeDetect-style linear probe.
"""
from __future__ import annotations

import numpy as np

from app.config import settings
from app.core.schemas import DetectorResult, Evidence, EvidenceCategory
from app.detectors.base import BaseDetector


class ClipSemanticDetector(BaseDetector):
    name = "clip_semantic_probe"
    default_weight = 0.9  # trusted most, when available -- it's a trained model, not a heuristic
    supports_image = True
    supports_video = True

    def __init__(self):
        super().__init__()
        self._model = None
        self._processor = None
        self._device = None
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        if self._loaded:
            return self._model is not None
        # Cheap dependency check without fully loading weights.
        try:
            import importlib
            importlib.import_module("torch")
            importlib.import_module("transformers")
            return True
        except ImportError:
            return False

    def load(self) -> None:
        self._loaded = True
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor = CLIPProcessor.from_pretrained(settings.clip_model_name)
            self._model = CLIPModel.from_pretrained(settings.clip_model_name).to(self._device).eval()

            # NOTE: this loads a base CLIP encoder for embeddings only. For a
            # real deployment, attach a trained linear probe / classification
            # head (e.g. `settings.clip_probe_checkpoint_path`) here and load
            # its weights via torch.load(...). Left as a clear extension
            # point rather than shipping an untrained/placeholder head.
            self._probe = None
        except Exception as exc:  # noqa: BLE001
            self._model = None
            self._load_error = str(exc)

    def analyze_image(self, image: np.ndarray) -> DetectorResult:
        self.ensure_loaded()
        if self._model is None:
            return self.safe_result(self._load_error or "CLIP semantic detector unavailable (torch/transformers not installed).")

        try:
            import torch
            from PIL import Image as PILImage

            pil_img = PILImage.fromarray(image)
            inputs = self._processor(images=pil_img, return_tensors="pt").to(self._device)
            with torch.no_grad():
                _ = self._model.get_image_features(**inputs)

            # Without a trained probe head this cannot produce a calibrated
            # probability, so we surface it as a neutral, zero-confidence
            # result rather than fabricating a score. Once a real probe is
            # attached above, replace this block with its sigmoid output.
            return DetectorResult(
                detector=self.name,
                ai_probability=0.5,
                confidence=0.0,
                evidence=[Evidence(
                    category=EvidenceCategory.semantic,
                    summary=(
                        "CLIP embedding extracted successfully, but no trained "
                        "real-vs-AI probe head is attached yet -- this detector is "
                        "wired up but intentionally inert until a fine-tuned "
                        "checkpoint is configured."
                    ),
                    score=0.5, weight=0.0, detector=self.name,
                )],
            )
        except Exception as exc:  # noqa: BLE001
            return self.safe_result(str(exc))
