"""
Base interface every detector must implement.

Detectors are intentionally small, single-responsibility, and side-effect free:
each one takes an image (or list of video frames) and returns a DetectorResult.
This makes it possible to add/remove/replace models (UniversalFakeDetect, DIRE,
CLIP-based detectors, XceptionNet, VideoMAE, TimeSformer, etc.) without touching
the API layer or the fusion logic.

Contract:
- __init__ should be cheap. Expensive work (loading model weights) happens lazily
  in `load()`, which is called once by the registry on first use.
- `analyze_image` / `analyze_video_frames` must NEVER raise. On failure they
  should return a DetectorResult with `error` set and ai_probability=0.5 /
  confidence=0.0 so the fusion stage can down-weight it automatically.
- `available` reports whether the detector's dependencies/weights are present.
  Detectors that need optional heavy deps (torch, transformers, downloaded
  weights) should set this to False gracefully instead of crashing the app.
"""
from __future__ import annotations

import abc
from typing import List, Optional

import numpy as np

from app.core.schemas import DetectorResult


class BaseDetector(abc.ABC):
    #: Unique, stable identifier used in API responses and config.
    name: str = "base_detector"

    #: Relative trust weight used by the fusion stage (0-1). Tune per detector
    #: quality; heuristic detectors are weighted lower than trained classifiers.
    default_weight: float = 0.5

    #: Which media types this detector supports.
    supports_image: bool = True
    supports_video: bool = False
    supports_audio: bool = False

    def __init__(self) -> None:
        self._loaded = False

    @property
    def available(self) -> bool:
        """Override if the detector depends on optional packages/weights."""
        return True

    def load(self) -> None:
        """Lazily load model weights. Called once before first use."""
        self._loaded = True

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    @abc.abstractmethod
    def analyze_image(self, image: np.ndarray) -> DetectorResult:
        """image: HxWx3 RGB uint8 numpy array."""
        raise NotImplementedError

    def analyze_video_frames(self, frames: List[np.ndarray], timestamps: List[float]) -> List[DetectorResult]:
        """Default video handling: run the image detector on sampled frames.

        Detectors with real temporal modeling (VideoMAE, TimeSformer) should
        override this to analyze the whole clip instead of frame-by-frame.
        """
        return [self.analyze_image(f) for f in frames]

    def analyze_audio(self, waveform: np.ndarray, sample_rate: int) -> DetectorResult:
        """Optional audio path for video soundtracks. Override in audio detectors."""
        return self.safe_result("audio analysis not supported by this detector")

    def safe_result(self, error: str) -> DetectorResult:
        return DetectorResult(
            detector=self.name,
            ai_probability=0.5,
            confidence=0.0,
            evidence=[],
            error=error,
        )
