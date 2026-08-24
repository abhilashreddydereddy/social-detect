# Exported Artifacts

This folder is the handoff point between training and inference.

Suggested structure:

```text
training/exports/
  image_branch/
    best_model.pth
    val_predictions.csv
  video_branch/
    best_model.pth
    val_predictions.csv
  fusion/
    fusion.pkl
    feature_schema.json
```

After validating an artifact, copy or sync it into `backend/models/` and
point the backend config at that file.
