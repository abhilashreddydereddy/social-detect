# Image-branch checkpoints (CIFake-trained EfficientNet + optional FFT)

Place the exported weights here after training:

```powershell
Copy-Item training\exports\image_branch\cifake\best_model.pth `
  backend\models\image_branch\cifake_best.pth
```

The dashboard and extension pick this up from `GET /status` (`image_branch_cifake.available`)
and from analysis metadata (`primary_model`). Restart the backend after copying.

Or set:

```text
SOCIAL_DETECT_IMAGE_MODEL_CHECKPOINT_PATH=/absolute/path/to/best_model.pth
```

The `image_branch_cifake` detector loads this file (or the training export path)
and activates automatically. Until a checkpoint exists it reports `available=false`.
