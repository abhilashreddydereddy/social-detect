"""Optional WeDefense TorchScript audio-deepfake detector.

WeDefense exports its deployable detection models as two TorchScript modules:
the frontend converts 16 kHz waveforms to SSL features and the backend returns
bonafide/spoof logits.  Keeping that exported format here avoids coupling the
API service to WeDefense's complete research/training dependency stack.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from app.config import settings
from app.core.schemas import DetectorResult, Evidence, EvidenceCategory
from app.detectors.base import BaseDetector


class WeDefenseAudioDetector(BaseDetector):
    name = "wedefense_audio"
    default_weight = 0.95
    learned = True
    supports_image = False
    supports_video = True
    supports_audio = True
    sample_rate = 16000

    def __init__(self) -> None:
        super().__init__()
        self._frontend = None
        self._backend = None
        self._device = "cpu"
        self._available = False
        self._load_error: str | None = None
        self._frontend_path = getattr(settings, "wedefense_audio_frontend_path", None)
        self._backend_path = getattr(settings, "wedefense_audio_backend_path", None)

    @property
    def available(self) -> bool:
        if self._loaded:
            return self._available
        return bool(
            self._frontend_path
            and self._backend_path
            and Path(self._frontend_path).is_file()
            and Path(self._backend_path).is_file()
        )

    def load(self) -> None:
        self._loaded = True
        try:
            import torch

            if not (
                self._frontend_path
                and self._backend_path
                and Path(self._frontend_path).is_file()
                and Path(self._backend_path).is_file()
            ):
                self._load_error = "WeDefense frontend/backend TorchScript files are not configured"
                return
            self._device = "cuda" if torch.cuda.is_available() and settings.device != "cpu" else "cpu"
            self._frontend = torch.jit.load(str(self._frontend_path), map_location=self._device).eval()
            self._backend = torch.jit.load(str(self._backend_path), map_location=self._device).eval()
            self._available = True
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"Could not load WeDefense TorchScript modules: {exc}"
            self._frontend = self._backend = None

    def analyze_image(self, image: np.ndarray) -> DetectorResult:
        return self.safe_result("wedefense_audio requires an audio waveform")

    def analyze_video_frames(self, frames: List[np.ndarray], timestamps: List[float]) -> List[DetectorResult]:
        return [self.safe_result("Use analyze_audio() for the extracted soundtrack")]

    def analyze_audio(self, waveform: np.ndarray, sample_rate: int) -> DetectorResult:
        try:
            self.ensure_loaded()
            if self._frontend is None or self._backend is None:
                return self.safe_result(self._load_error or "WeDefense detector is unavailable")
            wav = _resample_mono(waveform, sample_rate, self.sample_rate)
            if wav.size < self.sample_rate // 4:
                return self.safe_result("Audio track too short for WeDefense analysis")

            import torch

            scores = []
            for chunk in _chunks(wav, self.sample_rate):
                tensor = torch.from_numpy(chunk).unsqueeze(0).to(self._device)
                lengths = torch.tensor([chunk.size], dtype=torch.long, device=self._device)
                with torch.no_grad():
                    output = self._frontend(tensor, lengths)
                    features = output[0] if isinstance(output, tuple) else output
                    logits = self._backend(features)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    probs = torch.softmax(logits, dim=-1)
                spoof_index = int(settings.wedefense_audio_spoof_class_index)
                if probs.ndim != 2 or not 0 <= spoof_index < probs.shape[1]:
                    raise ValueError(f"invalid spoof class index {spoof_index} for logits shape {tuple(probs.shape)}")
                scores.append(float(probs[0, spoof_index].item()))

            probability = float(np.mean(scores))
            spread = float(np.std(scores)) if len(scores) > 1 else 0.0
            confidence = float(np.clip(abs(probability - 0.5) * 1.8 + 0.2 - spread * 0.25, 0.1, 0.98))
            return DetectorResult(
                detector=self.name,
                ai_probability=probability,
                confidence=confidence,
                evidence=[Evidence(
                    category=EvidenceCategory.audio_artifact,
                    summary=("WeDefense SSL anti-spoofing model score "
                             f"(P(spoof)={probability:.3f}; {len(scores)} audio window(s))."),
                    score=probability,
                    weight=self.default_weight,
                    detector=self.name,
                )],
            )
        except Exception as exc:  # noqa: BLE001
            return self.safe_result(str(exc))


def _resample_mono(waveform: np.ndarray, input_rate: int, target_rate: int) -> np.ndarray:
    if input_rate <= 0:
        raise ValueError("sample rate must be positive")
    wav = np.asarray(waveform, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=0 if wav.shape[0] <= 8 and wav.shape[0] < wav.shape[1] else 1)
    wav = wav.reshape(-1)
    if input_rate != target_rate:
        from scipy import signal
        wav = signal.resample_poly(wav, target_rate, input_rate).astype(np.float32)
    return wav / max(float(np.max(np.abs(wav))) if wav.size else 0.0, 1e-5)


def _chunks(waveform: np.ndarray, sample_rate: int) -> List[np.ndarray]:
    window = max(int(settings.wedefense_audio_window_seconds * sample_rate), sample_rate)
    hop = max(int(settings.wedefense_audio_hop_seconds * sample_rate), sample_rate // 2)
    if waveform.size <= window:
        return [waveform]
    starts = list(range(0, waveform.size - window + 1, hop))
    if starts[-1] + window < waveform.size:
        starts.append(waveform.size - window)
    return [waveform[start:start + window] for start in starts]
