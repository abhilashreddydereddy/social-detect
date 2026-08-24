# Image Branch

This branch should produce a learned `p(fake)` score for a single image.

Recommended starting point:

- integrate or mirror the GRIP-UNINA CLIP-based detector
- evaluate on a held-out social-media validation set

## Outputs

- checkpoint: `best_model.pth`
- validation predictions: `val_predictions.csv`

## Training goal

Train or fine-tune a branch that generalizes across:

- clean synthetic images
- recompressed social-media images
- screenshots
- reposted / cropped images

## Next step

Start by creating a manifest CSV and a config file under `configs/`.
