"""
MFAD-Net detector — loads a trained checkpoint and runs trimodal inference.

When no checkpoint is present (or torch is unavailable), `available` is False
and the registry skips this detector. Visual-only fallback: if audio/meta are
missing, zeros are used for those branches (still runs visual+FFT path).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from app.config import settings
from app.core.schemas import DetectorResult, Evidence, EvidenceCategory
from app.detectors.base import BaseDetector

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class MFADNetDetector(BaseDetector):
    name = "mfad_net"
    default_weight = 0.9
    learned = True
    supports_image = True
    supports_video = True
    supports_audio = True

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        super().__init__()
        self._checkpoint_path = (
            checkpoint_path
            or getattr(settings, "mfad_net_checkpoint_path", None)
            or str(_REPO_ROOT / "backend" / "models" / "mfad_net" / "mfad_net_best.pth")
        )
        self._model = None
        self._device = "cpu"
        self._available = False
        self._last_drift = None

    @property
    def available(self) -> bool:
        return self._available

    def load(self) -> None:
        self._loaded = True
        try:
            import torch
            from training.mfad_net.models.mfad_net import MFADNet, MFADNetConfig

            path = Path(self._checkpoint_path)
            # Also accept training export path
            candidates = [
                path,
                _REPO_ROOT / "training" / "exports" / "mfad_net" / "mfad_net_best.pth",
                _REPO_ROOT / "backend" / "models" / "mfad_net" / "mfad_net_best.pth",
            ]
            ckpt_file = next((p for p in candidates if p.exists()), None)
            if ckpt_file is None:
                self._available = False
                return

            self._device = "cuda" if torch.cuda.is_available() and settings.device != "cpu" else "cpu"
            ckpt = torch.load(ckpt_file, map_location=self._device, weights_only=False)
            cfg = MFADNetConfig.from_dict(ckpt.get("model_config", {"lite": True}))
            model = MFADNet(cfg)
            model.load_state_dict(ckpt["state_dict"])
            model.to(self._device)
            model.eval()
            self._model = model
            self._checkpoint_path = str(ckpt_file)
            self._available = True
        except Exception as exc:  # noqa: BLE001
            self._available = False
            self._model = None
            self._load_error = str(exc)

    def analyze_image(self, image: np.ndarray) -> DetectorResult:
        try:
            if self._model is None:
                return self.safe_result("MFAD-Net checkpoint not loaded")
            import torch

            frame = self._prep_frame(image)
            frames = frame.unsqueeze(0).unsqueeze(0)  # (1,1,3,H,W)
            wav = torch.zeros(1, 16000, device=self._device)
            meta = torch.zeros(1, 16, device=self._device)
            return self._run(frames, wav, meta, source="image")
        except Exception as exc:  # noqa: BLE001
            return self.safe_result(str(exc))

    def analyze_video_frames(self, frames: List[np.ndarray], timestamps: List[float]) -> List[DetectorResult]:
        try:
            if self._model is None:
                return [self.safe_result("MFAD-Net checkpoint not loaded")]
            import torch

            if not frames:
                return [self.safe_result("No frames")]
            # Sample up to 8 frames
            idxs = np.linspace(0, len(frames) - 1, num=min(8, len(frames))).astype(int)
            tensor_frames = torch.stack([self._prep_frame(frames[i]) for i in idxs], dim=0)  # (T,3,H,W)
            tensor_frames = tensor_frames.unsqueeze(0).to(self._device)
            wav = torch.zeros(1, 16000, device=self._device)
            meta = torch.zeros(1, 16, device=self._device)
            result = self._run(tensor_frames, wav, meta, source="video_frames")
            return [result]
        except Exception as exc:  # noqa: BLE001
            return [self.safe_result(str(exc))]

    def analyze_audio(self, waveform: np.ndarray, sample_rate: int) -> DetectorResult:
        """Audio-only path: blank visual/meta, still runs full graph."""
        try:
            if self._model is None:
                return self.safe_result("MFAD-Net checkpoint not loaded")
            import torch

            wav = self._prep_wav(waveform, sample_rate)
            frames = torch.zeros(1, 1, 3, 224, 224, device=self._device)
            meta = torch.zeros(1, 16, device=self._device)
            return self._run(frames, wav, meta, source="audio")
        except Exception as exc:  # noqa: BLE001
            return self.safe_result(str(exc))

    def analyze_multimodal(
        self,
        frames: List[np.ndarray],
        waveform: Optional[np.ndarray],
        sample_rate: int = 16000,
        meta_vec: Optional[np.ndarray] = None,
    ) -> DetectorResult:
        try:
            if self._model is None:
                return self.safe_result("MFAD-Net checkpoint not loaded")
            import torch

            if frames:
                idxs = np.linspace(0, len(frames) - 1, num=min(8, len(frames))).astype(int)
                tensor_frames = torch.stack([self._prep_frame(frames[i]) for i in idxs], dim=0).unsqueeze(0)
            else:
                tensor_frames = torch.zeros(1, 1, 3, 224, 224)
            tensor_frames = tensor_frames.to(self._device)

            if waveform is not None:
                wav = self._prep_wav(waveform, sample_rate)
            else:
                wav = torch.zeros(1, 16000, device=self._device)

            if meta_vec is not None:
                meta = torch.from_numpy(np.asarray(meta_vec, dtype=np.float32).reshape(-1)[:16])
                if meta.numel() < 16:
                    meta = torch.nn.functional.pad(meta, (0, 16 - meta.numel()))
                meta = meta.unsqueeze(0).to(self._device)
            else:
                meta = torch.zeros(1, 16, device=self._device)

            return self._run(tensor_frames, wav, meta, source="multimodal")
        except Exception as exc:  # noqa: BLE001
            return self.safe_result(str(exc))

    def _run(self, frames, wav, meta, source: str) -> DetectorResult:
        import torch

        with torch.no_grad():
            out = self._model(frames, wav, meta, update_drift=True)

        p_fake = float(out["p_fake"][0].item())
        conf = float(min(0.95, abs(p_fake - 0.5) * 1.8 + 0.15))
        evidence = [
            Evidence(
                category=EvidenceCategory.semantic,
                summary=f"MFAD-Net trimodal fusion score ({source}).",
                score=p_fake,
                weight=0.5,
                detector=self.name,
            )
        ]
        av = float(out["score_av"][0].item())
        if av > 0.55:
            evidence.append(Evidence(
                category=EvidenceCategory.audio_artifact,
                summary="Cross-modal AV sync head flags audio-visual inconsistency.",
                score=av,
                weight=0.25,
                detector=self.name,
            ))
        meta_s = float(out["score_meta"][0].item())
        if meta_s > 0.55:
            evidence.append(Evidence(
                category=EvidenceCategory.metadata,
                summary="Metadata anomaly head flags unusual propagation / account signals.",
                score=meta_s,
                weight=0.15,
                detector=self.name,
            ))
        if "mmd_score" in out and float(out["mmd_score"].item()) > 0.15:
            evidence.append(Evidence(
                category=EvidenceCategory.semantic,
                summary=f"TSDD drift alert: MMD={float(out['mmd_score'].item()):.3f} vs prototype memory.",
                score=min(1.0, float(out["mmd_score"].item())),
                weight=0.2,
                detector=self.name,
            ))

        self._last_drift = {
            "mmd_score": float(out["mmd_score"].item()) if "mmd_score" in out else 0.0,
            "drift_flag": bool(out["drift_flag"].item()) if "drift_flag" in out else False,
            "tool_match": out.get("tool_match"),
            "checkpoint": self._checkpoint_path,
        }
        return DetectorResult(
            detector=self.name,
            ai_probability=p_fake,
            confidence=conf,
            evidence=evidence,
        )

    def _prep_frame(self, image: np.ndarray):
        import torch
        import cv2

        rgb = image
        if rgb.shape[0] != 224 or rgb.shape[1] != 224:
            rgb = cv2.resize(rgb, (224, 224))
        arr = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        return torch.from_numpy(arr).permute(2, 0, 1).to(self._device)

    def _prep_wav(self, waveform: np.ndarray, sample_rate: int):
        import torch

        wav = np.asarray(waveform, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        target_sr = 16000
        if sample_rate != target_sr and sample_rate > 0:
            # Linear resample
            duration = len(wav) / float(sample_rate)
            new_len = max(1, int(duration * target_sr))
            x_old = np.linspace(0, 1, num=len(wav), endpoint=False)
            x_new = np.linspace(0, 1, num=new_len, endpoint=False)
            wav = np.interp(x_new, x_old, wav).astype(np.float32)
        target_len = target_sr  # 1 second
        if len(wav) < target_len:
            wav = np.pad(wav, (0, target_len - len(wav)))
        else:
            wav = wav[:target_len]
        return torch.from_numpy(wav).unsqueeze(0).to(self._device)
