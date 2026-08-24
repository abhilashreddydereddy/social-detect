# Video Branch

This branch should produce a learned `p(fake)` score for a full clip.

Recommended starting point:

- use DeepfakeBench as the training/evaluation ecosystem
- begin with one stable detector rather than many at once

## Outputs

- checkpoint: `best_model.pth`
- validation predictions: `val_predictions.csv`

## Suggested datasets

- DFDC
- FaceForensics++
- Celeb-DF
- DF40

## Next step

Create dataset manifests or point the config to your DeepfakeBench-prepared data.
