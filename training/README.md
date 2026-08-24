# Training Workspace

This folder is the model-development side of Social Detect. Keep training,
dataset prep, experiment tracking, and exported checkpoints here. The live
API under `backend/` should stay inference-only.

## Layout

```text
training/
  README.md
  requirements.txt
  image_branch/
    README.md
    configs/
    scripts/
  video_branch/
    README.md
    configs/
    scripts/
  fusion/
    README.md
    configs/
    scripts/
  data/
    README.md
  exports/
    README.md
```

## Recommended workflow

1. Prepare datasets under `training/data/` or point config files to external
   dataset mounts.
2. Train the image branch first.
3. Export image scores for a held-out validation set.
4. Train the fusion model on branch scores + metadata features.
5. Train the video branch after the image branch is stable.
6. Copy the best exported checkpoints into `backend/models/` and point the
   backend config to them.

## First target

**MFAD-Net** (paper implementation) lives under `training/mfad_net/`:

```bash
pip install -r training/requirements.txt
python -m training.mfad_net.scripts.train --config training/mfad_net/configs/smoke.yaml
cp training/exports/mfad_net/mfad_net_best.pth backend/models/mfad_net/
```

See `training/mfad_net/README.md` for the full architecture (EfficientNet+FFT,
Wav2Vec2+MFCC, GAT metadata, CMAF fusion, TSDD drift detector) and how to
point `full.yaml` at FaceForensics++ / DFDC / WildDeepfake manifests.

Older stubs remain under `image_branch/`, `video_branch/`, and `fusion/`.

## Suggested environments

- Local CPU only: smoke tests, config validation, tiny runs
- GPU box / cloud GPU: real training and checkpoint export

## Notes

- Do not commit raw datasets here.
- Keep large checkpoints in external storage or Git LFS if you later decide to version them.
- Prefer reproducible config files over one-off notebook-only training.
