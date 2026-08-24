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

The fastest practical target is:

- `image_branch`: integrate a pretrained or fine-tuned GRIP-UNINA-style image detector
- `fusion`: train a small logistic-regression or gradient-boosted final scorer
- `video_branch`: add a DeepfakeBench-backed video detector after the image path is working

## Suggested environments

- Local CPU only: smoke tests, config validation, tiny runs
- GPU box / cloud GPU: real training and checkpoint export

## Notes

- Do not commit raw datasets here.
- Keep large checkpoints in external storage or Git LFS if you later decide to version them.
- Prefer reproducible config files over one-off notebook-only training.
