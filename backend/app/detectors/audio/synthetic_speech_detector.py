"""
SyntheticSpeechDetector
-----------------------
Heuristic audio authenticity probe for video clips.

Looks for cues common in TTS / voice-cloned / fully synthetic soundtracks:
  - unnaturally flat spectral envelope (low spectral variance over time)
  - near-perfect periodicity / robotic pitch stability
  - missing low-level ambient floor (studio-clean silence between phrases)
  - excessive high-band energy consistency typical of vocoders

This is intentionally dependency-light (numpy + scipy only) so it runs
alongside the image/video frame detectors without requiring torchaudio or
pretrained ASR models. A learned audio classifier can replace the scoring
block later behind the same BaseDetector interface.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy import signal

from app.core.schemas import DetectorResult, Evidence, EvidenceCategory
from app.detectors.base import BaseDetector


class SyntheticSpeechDetector(BaseDetector):
    name = "synthetic_speech_audio"
    default_weight = 0.55
    supports_image = False
    supports_video = True
    supports_audio = True

    def analyze_image(self, image: np.ndarray) -> DetectorResult:
        return self.safe_result("synthetic_speech_audio requires an audio waveform")

    def analyze_video_frames(self, frames: List[np.ndarray], timestamps: List[float]) -> List[DetectorResult]:
        return [self.safe_result("Use analyze_audio() for the extracted soundtrack")]

    def analyze_audio(self, waveform: np.ndarray, sample_rate: int) -> DetectorResult:
        try:
            if waveform is None or waveform.size == 0 or sample_rate <= 0:
                return self.safe_result("No audio track available")

            mono = _to_mono(waveform)
            if mono.size < sample_rate // 4:
                return self.safe_result("Audio track too short for analysis")

            mono = mono - float(np.mean(mono))
            peak = float(np.max(np.abs(mono)) + 1e-9)
            mono = mono / peak

            flatness = _spectral_flatness_score(mono, sample_rate)
            pitch_stability = _pitch_stability_score(mono, sample_rate)
            silence_floor = _silence_floor_score(mono)
            band_consistency = _highband_consistency_score(mono, sample_rate)

            # Higher score => more likely synthetic.
            ai_probability = float(np.clip(
                0.30 * flatness
                + 0.30 * pitch_stability
                + 0.20 * silence_floor
                + 0.20 * band_consistency,
                0.0,
                1.0,
            ))
            confidence = float(np.clip(abs(ai_probability - 0.5) * 1.7 + 0.15, 0.1, 0.85))

            evidence: List[Evidence] = []
            if flatness > 0.55:
                evidence.append(Evidence(
                    category=EvidenceCategory.audio_artifact,
                    summary=(
                        "Spectral flatness is unusually steady across the clip, "
                        "a pattern often left by vocoders and TTS renderers."
                    ),
                    score=flatness, weight=0.3, detector=self.name,
                ))
            if pitch_stability > 0.55:
                evidence.append(Evidence(
                    category=EvidenceCategory.audio_artifact,
                    summary=(
                        "Pitch contour is unnaturally stable with low micro-variation, "
                        "consistent with synthetic or heavily processed speech."
                    ),
                    score=pitch_stability, weight=0.3, detector=self.name,
                ))
            if silence_floor > 0.55:
                evidence.append(Evidence(
                    category=EvidenceCategory.audio_artifact,
                    summary=(
                        "Inter-phrase silence lacks a natural ambient noise floor, "
                        "suggesting digitally generated or heavily denoised audio."
                    ),
                    score=silence_floor, weight=0.2, detector=self.name,
                ))
            if band_consistency > 0.55:
                evidence.append(Evidence(
                    category=EvidenceCategory.audio_artifact,
                    summary=(
                        "High-frequency energy is overly consistent frame-to-frame, "
                        "which is common in neural vocoder output."
                    ),
                    score=band_consistency, weight=0.2, detector=self.name,
                ))
            if not evidence:
                evidence.append(Evidence(
                    category=EvidenceCategory.audio_artifact,
                    summary=(
                        "Audio spectral and pitch statistics are within the range "
                        "typical of natural recordings."
                    ),
                    score=ai_probability, weight=0.2, detector=self.name,
                ))

            return DetectorResult(
                detector=self.name,
                ai_probability=ai_probability,
                confidence=confidence,
                evidence=evidence,
            )
        except Exception as exc:  # noqa: BLE001
            return self.safe_result(str(exc))


def _to_mono(waveform: np.ndarray) -> np.ndarray:
    arr = np.asarray(waveform, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        # (channels, samples) or (samples, channels)
        if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1]:
            return arr.mean(axis=0)
        return arr.mean(axis=1)
    return arr.reshape(-1)


def _frame_signal(mono: np.ndarray, sample_rate: int, frame_ms: float = 25.0, hop_ms: float = 10.0):
    frame = max(int(sample_rate * frame_ms / 1000.0), 16)
    hop = max(int(sample_rate * hop_ms / 1000.0), 8)
    if mono.size < frame:
        return np.empty((0, frame), dtype=np.float32), frame, hop
    n = 1 + (mono.size - frame) // hop
    frames = np.lib.stride_tricks.as_strided(
        mono,
        shape=(n, frame),
        strides=(mono.strides[0] * hop, mono.strides[0]),
        writeable=False,
    ).copy()
    window = np.hanning(frame).astype(np.float32)
    return frames * window, frame, hop


def _spectral_flatness_score(mono: np.ndarray, sample_rate: int) -> float:
    frames, _, _ = _frame_signal(mono, sample_rate)
    if frames.shape[0] < 4:
        return 0.5
    specs = np.abs(np.fft.rfft(frames, axis=1)) + 1e-12
    geo = np.exp(np.mean(np.log(specs), axis=1))
    arith = np.mean(specs, axis=1)
    flatness = geo / arith
    # Synthetic speech often has persistently mid/high flatness with low variance.
    mean_f = float(np.mean(flatness))
    var_f = float(np.var(flatness))
    steady = float(np.clip(1.0 - var_f * 40.0, 0.0, 1.0))
    level = float(np.clip((mean_f - 0.05) / 0.35, 0.0, 1.0))
    return float(np.clip(0.55 * level + 0.45 * steady, 0.0, 1.0))


def _pitch_stability_score(mono: np.ndarray, sample_rate: int) -> float:
    # Autocorrelation pitch estimate on overlapping windows.
    frame = max(int(sample_rate * 0.04), 64)
    hop = max(int(sample_rate * 0.02), 32)
    if mono.size < frame * 3:
        return 0.5

    min_lag = int(sample_rate / 400)  # ~400 Hz
    max_lag = int(sample_rate / 60)   # ~60 Hz
    pitches = []
    for start in range(0, mono.size - frame, hop):
        chunk = mono[start:start + frame]
        if float(np.sqrt(np.mean(chunk ** 2))) < 0.02:
            continue
        corr = signal.correlate(chunk, chunk, mode="full")
        mid = len(corr) // 2
        segment = corr[mid + min_lag: mid + max_lag + 1]
        if segment.size == 0:
            continue
        lag = int(np.argmax(segment)) + min_lag
        pitches.append(sample_rate / float(lag))

    if len(pitches) < 4:
        return 0.4

    pitches = np.array(pitches, dtype=np.float32)
    # Coefficient of variation: natural speech usually varies more.
    cv = float(np.std(pitches) / (np.mean(pitches) + 1e-6))
    return float(np.clip(1.0 - cv / 0.25, 0.0, 1.0))


def _silence_floor_score(mono: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(mono ** 2)))
    # Quietest 15% of sliding windows — natural room tone sits above digital zero.
    win = max(len(mono) // 100, 256)
    if mono.size < win * 4:
        return 0.4
    energies = []
    for i in range(0, mono.size - win, win):
        chunk = mono[i:i + win]
        energies.append(float(np.sqrt(np.mean(chunk ** 2))))
    energies = np.array(sorted(energies))
    floor = float(np.mean(energies[: max(1, len(energies) // 7)]))
    # Extremely low floor relative to overall RMS => synthetic / gated.
    ratio = floor / (rms + 1e-9)
    return float(np.clip(1.0 - ratio / 0.08, 0.0, 1.0))


def _highband_consistency_score(mono: np.ndarray, sample_rate: int) -> float:
    frames, frame, _ = _frame_signal(mono, sample_rate)
    if frames.shape[0] < 4:
        return 0.5
    specs = np.abs(np.fft.rfft(frames, axis=1)) + 1e-12
    freqs = np.fft.rfftfreq(frame, d=1.0 / sample_rate)
    high = freqs >= max(3000.0, sample_rate * 0.15)
    if not np.any(high):
        return 0.5
    high_energy = specs[:, high].mean(axis=1)
    # Low relative variance in high band => vocoder-like steadiness.
    cv = float(np.std(high_energy) / (np.mean(high_energy) + 1e-9))
    return float(np.clip(1.0 - cv / 0.6, 0.0, 1.0))
