# Backend Model Artifacts

Copy validated exports from `training/exports/` into this folder when you are
ready to use them in the API.

Suggested layout:

```text
backend/models/
  image_branch/
    best_model.pth
  video_branch/
    best_model.pth
  fusion/
    fusion.pkl
```

Then point `SOCIAL_DETECT_IMAGE_MODEL_CHECKPOINT_PATH`,
`SOCIAL_DETECT_VIDEO_MODEL_CHECKPOINT_PATH`, and
`SOCIAL_DETECT_FUSION_MODEL_PATH` at those files.
