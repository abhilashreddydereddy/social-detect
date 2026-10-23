import numpy as np

from app.detectors.audio.wedefense_detector import _chunks, _resample_mono


def test_wedefense_audio_resamples_and_normalizes_stereo():
    stereo = np.column_stack((np.ones(8000, dtype=np.float32) * 0.25, np.ones(8000, dtype=np.float32) * 0.5))
    audio = _resample_mono(stereo, 8000, 16000)
    assert audio.shape == (16000,)
    assert np.isclose(np.max(np.abs(audio)), 1.0, atol=0.05)


def test_wedefense_audio_chunks_cover_tail(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "wedefense_audio_window_seconds", 2.0)
    monkeypatch.setattr(settings, "wedefense_audio_hop_seconds", 1.0)
    chunks = _chunks(np.zeros(5500, dtype=np.float32), 1000)
    assert [chunk.size for chunk in chunks] == [2000, 2000, 2000, 2000, 2000]
