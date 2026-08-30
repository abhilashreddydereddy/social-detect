# Video Branch

## Current production path (no separate video weights required)

Video uploads are **cut into frames** and scored by the **CIFake image model**
(`image_branch_cifake`). Aggregation (mean `p(fake)` across sampled frames)
happens inside that detector. Train CIFake first — see
[`training/image_branch/README.md`](../image_branch/README.md).

## Optional dedicated temporal model

This folder is for a future clip-level temporal network (e.g. DeepfakeBench /
VideoMAE) that can **add** to fusion alongside frame scoring.

Recommended starting point:

- use DeepfakeBench as the training/evaluation ecosystem
- begin with one stable detector rather than many at once

## Outputs (when implemented)

- checkpoint: `best_model.pth`
- validation predictions: `val_predictions.csv`

## Suggested datasets

- DFDC
- FaceForensics++
- Celeb-DF
- DF40

## Next step

Create dataset manifests or point the config to your DeepfakeBench-prepared data.
