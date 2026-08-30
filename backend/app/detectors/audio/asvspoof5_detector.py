from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np

from app.config import settings
from app.core.schemas import DetectorResult, Evidence, EvidenceCategory
from app.detectors.base import BaseDetector

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class ASVSpoof5Detector(BaseDetector):
    name = "asvspoof5_audio"
    default_weight = 0.9
    learned = True
    supports_image = False
    supports_video = True
    supports_audio = True

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._device = "cpu"
        self._available = False
        self._checkpoint_path = getattr(settings, "asvspoof5_audio_checkpoint_path", None) or str(_REPO_ROOT / "backend" / "models" / "audio_branch" / "asvspoof5_best.pth")
        self._load_error = None

    @property
    def available(self) -> bool:
        return self._available if self._loaded else Path(self._checkpoint_path).exists()

    def load(self) -> None:
        self._loaded = True
        try:
            import torch
            from app.detectors.audio.audio_classifier import AudioClassifier, AudioClassifierConfig
            path = Path(self._checkpoint_path)
            if not path.exists():
                return
            self._device = "cuda" if torch.cuda.is_available() and settings.device != "cpu" else "cpu"
            checkpoint = torch.load(path, map_location=self._device, weights_only=False)
            self._model = AudioClassifier(AudioClassifierConfig.from_dict(checkpoint.get("model_config")))
            self._model.load_state_dict(checkpoint["state_dict"])
            self._model.to(self._device).eval()
            self._available = True
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)

    def analyze_image(self, image: np.ndarray) -> DetectorResult:
        return self.safe_result("asvspoof5_audio requires an audio waveform")

    def analyze_video_frames(self, frames: List[np.ndarray], timestamps: List[float]) -> List[DetectorResult]:
        return [self.safe_result("Use analyze_audio() for the extracted soundtrack")]

    def analyze_audio(self, waveform: np.ndarray, sample_rate: int) -> DetectorResult:
        try:
            import torch

            self.ensure_loaded()
            if self._model is None:
                return self.safe_result(self._load_error or "ASVspoof 5 checkpoint not loaded")
            wav = np.asarray(waveform, dtype=np.float32)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            target = int(self._model.config.sample_rate * self._model.config.clip_seconds)
            if sample_rate != self._model.config.sample_rate:
                import scipy.signal
                wav = scipy.signal.resample(wav, round(wav.size * self._model.config.sample_rate / sample_rate)).astype(np.float32)
            if wav.size >= target:
                wav = wav[:target]
            else:
                wav = np.pad(wav, (0, target - wav.size))
            wav = wav / max(float(np.max(np.abs(wav))), 1e-5)
            with torch.no_grad():
                output = self._model(torch.from_numpy(wav).unsqueeze(0).to(self._device))
            probability = float(output["p_fake"][0].item())
            return DetectorResult(detector=self.name, ai_probability=probability, confidence=float(np.clip(abs(probability - 0.5) * 1.8 + 0.15, 0.1, 0.95)), evidence=[Evidence(category=EvidenceCategory.audio_artifact, summary=f"ASVspoof 5 log-Mel audio classifier score (P(spoof)={probability:.3f}).", score=probability, weight=self.default_weight, detector=self.name)])
        except Exception as exc:  # noqa: BLE001
            return self.safe_result(str(exc))