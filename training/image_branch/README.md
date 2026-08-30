# Image Branch

Learned `p(fake)` for a **single image**. Trained primarily on **CIFake**;
videos reuse this model by sampling frames and aggregating scores.

## Train (CIFake)

1. Place or import data — see [`training/data/cifake/README.md`](../data/cifake/README.md).
2. Build manifests (if not using `from-tree`).
3. Train:

```powershell
pip install -r training/requirements.txt
python -m training.image_branch.scripts.train --config training/image_branch/configs/cifake.yaml
```

4. Export:

```powershell
Copy-Item training\exports\image_branch\cifake\best_model.pth backend\models\image_branch\cifake_best.pth
```

## Outputs

- `training/exports/image_branch/cifake/best_model.pth`
- `history.json`
- `val_predictions.csv`

## Model

Default: EfficientNet-B0 (timm) + optional FFT frequency branch
(`training/image_branch/models/image_classifier.py`). Swap to `efficientnet_b4`
or `tiny_cnn` in `configs/cifake.yaml`.

## Backend

Detector name: `image_branch_cifake`  
Registered for both image and video. Reports unavailable until a checkpoint exists.

## Related branches

- **Video temporal** (optional later): `training/video_branch/` — adds temporal modeling on top of frame scoring
- **Audio** (separate): `training/audio_branch/` — not trained on CIFake
