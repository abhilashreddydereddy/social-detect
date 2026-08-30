# Audio Branch

Separate from the CIFake **image** model.

## Role

- Score AI / synthetic speech and audio tracks from video uploads
- Backend: `SyntheticSpeechDetector` under `backend/app/detectors/audio/`
- Training data: audio corpora (e.g. ASVspoof, FakeAVCeleb audio) — **not CIFake**

## Status

Heuristic audio detection is live in the API. A learned ASVspoof-compatible
trainer is now available in `scripts/train.py` with `configs/asvspoof5.yaml`.
It expects WAV/FLAC files and creates manifests from ASVspoof protocol files.

## ASVspoof 5 workflow

After downloading ASVspoof 5, build manifests for each protocol split:

```powershell
python -m training.audio_branch.scripts.prepare_manifest --protocol path\to\protocol.txt --audio-root path\to\audio --split train --output training\data\manifests\asvspoof5_train.csv
python -m training.audio_branch.scripts.prepare_manifest --protocol path\to\protocol.txt --audio-root path\to\audio --split val --output training\data\manifests\asvspoof5_val.csv
```

Then train on CUDA:

```powershell
python -m training.audio_branch.scripts.train --config training/audio_branch/configs/asvspoof5.yaml
```

## Inference path

```text
video upload → extract soundtrack → AUDIO_DETECTORS
audio-only   → AUDIO_DETECTORS
image upload → skipped (no audio)
```
