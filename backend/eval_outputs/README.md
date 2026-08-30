# Realtime Evaluation

This folder stores artifacts for the paper row described as full multimodal realtime data.

Use `backend/scripts/evaluate_realtime.py` with a labeled CSV manifest:

```csv
path,label,media_type,platform,source
training/data/example/real_clip.mp4,0,video,youtube,yt_real_001
training/data/example/fake_clip.mp4,1,video,instagram,ig_fake_001
training/data/example/real_image.jpg,0,image,instagram,ig_real_014
```

Required columns:

- `path`: media file path, absolute or repo-relative
- `label`: `0` for authentic, `1` for AI/fake

Optional columns:

- `media_type`: `image`, `video`, or `auto`
- `platform`: platform label written into the request metadata
- `source`: human-readable sample id

Example:

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\evaluate_realtime.py `
  --manifest ..\Paper\realtime_manifest.csv `
  --output-dir eval_outputs\realtime `
  --run-name faceforensics_realtime
```

Use `--base-url http://127.0.0.1:8000` to benchmark a running backend instead of the in-process test client.
