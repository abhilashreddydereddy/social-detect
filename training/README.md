# Training Workspace

This folder is the model-development side of Social Detect. Keep training,
dataset prep, experiment tracking, and exported checkpoints here. The live
API under `backend/` should stay inference-only.

## Modality models

| Branch | Dataset | Inference |
|--------|---------|-----------|
| **Image** (`image_branch/`) | **CIFake** | Direct image uploads |
| **Video** | Frames via image model first; optional temporal later | Sample frames → image model |
| **Audio** (`audio_branch/`) | Audio corpora (not CIFake) | Soundtrack / audio-only |
| MFAD-Net (`mfad_net/`) | Multimodal deepfake sets | Optional ensemble |

## Layout

```text
training/
  README.md
  requirements.txt
  image_branch/     # CIFake image classifier (primary for stills + video frames)
  audio_branch/     # Separate audio model track
  video_branch/     # Optional dedicated temporal models
  mfad_net/         # Multimodal paper model
  fusion/
  data/
  exports/
```

## Recommended workflow

1. **Train image branch on CIFake** (see `image_branch/README.md` and `data/cifake/README.md`).
2. Copy `best_model.pth` → `backend/models/image_branch/cifake_best.pth`.
3. Restart backend — images and video frames use that checkpoint.
4. (Optional) Train MFAD-Net / video temporal / audio models and register weights.

### CIFake quick start

```powershell
pip install -r training/requirements.txt
python -m training.image_branch.scripts.prepare_manifest from-tree --root path\to\CIFAKE
python -m training.image_branch.scripts.train --config training/image_branch/configs/cifake.yaml
Copy-Item training\exports\image_branch\cifake\best_model.pth backend\models\image_branch\cifake_best.pth
```

## MFAD-Net (multimodal)

```bash
pip install -r training/requirements.txt
python -m training.mfad_net.scripts.train --config training/mfad_net/configs/smoke.yaml
cp training/exports/mfad_net/mfad_net_best.pth backend/models/mfad_net/
```

See `training/mfad_net/README.md` for FaceForensics++ / DFDC / WildDeepfake.

## Suggested environments

- Local CPU only: smoke tests, config validation, tiny runs (`backbone: tiny_cnn`)
- GPU box / cloud GPU: full CIFake + EfficientNet training

## Notes

- Do not commit raw datasets here.
- Keep large checkpoints in external storage or Git LFS if you later decide to version them.
- Prefer reproducible config files over one-off notebook-only training.
