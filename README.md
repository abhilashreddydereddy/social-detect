# Social Detect

A browser-first AI media authenticity platform. Social Detect estimates
whether an image or video is likely AI-generated or manipulated — and
always returns a **probability**, a **confidence score**, and
**explainable evidence**, never a definitive "real" or "fake" verdict.

```
Chrome/Edge Extension  ──┐
                          ├──▶  FastAPI Backend ──▶ Detector Registry ──▶ Score Fusion ──▶ JSON
React Dashboard (demo) ──┘                          (image + video)
```

## Repository layout

```
social-detect/
├── backend/      FastAPI service: modular detectors, ensemble fusion, REST API
├── dashboard/    React + Tailwind demo/testing UI (same backend as the extension)
├── extension/    Manifest V3 Chrome/Edge extension (Instagram first)
└── docker-compose.yml
```

## Design principle: no verdicts

Every response — from the API, the dashboard, and the extension overlay —
carries the same disclaimer baked into the schema itself
(`AnalysisResponse.disclaimer`): this is a probabilistic estimate, not proof.
Classification labels are deliberately hedged (`Likely AI Generated`,
`Possibly Manipulated`, `Likely Authentic`, `Inconclusive`), and confidence
is computed independently from probability so the system can say "I don't
know" instead of forcing a lopsided guess into false certainty.

---

## 1. Backend (`/backend`)

FastAPI service exposing:

| Endpoint | Purpose |
|---|---|
| `POST /analyze/image` | multipart image upload |
| `POST /analyze/video` | multipart video upload — frames cut + scored as images; audio extracted and scored in parallel |
| `POST /analyze/media` | multipart upload with **auto image/video detection** (magic bytes / content-type / filename) |
| `POST /analyze/frames` | multipart sequence of JPEG frames (extension fallback when only canvas snapshots are available) |
| `POST /analyze/url` | JSON `{ url, platform_hint? }` — direct media URL (auto-detects kind) |
| `GET /status` | health check + which detectors are currently active |

### Modular detector architecture

Every detector implements `app/detectors/base.py::BaseDetector` and is
registered in **one place**, `app/detectors/registry.py`. Detectors are
lazy-loaded, never raise (failures degrade to a neutral, zero-confidence
result instead of a 500), and self-report `available` so optional heavy
dependencies (torch/transformers) can be absent without breaking the API.

Shipped out of the box (dependency-light, no downloaded weights required):

- **`frequency_artifact_fft`** — 2D FFT spectral analysis; flags periodic
  grid artifacts and unnatural high-frequency roll-off typical of GAN/
  diffusion up-sampling.
- **`sensor_noise_residual`** — high-pass noise-residual analysis; flags
  images missing the camera sensor noise floor real photos have.
- **`compression_ela`** — Error Level Analysis; flags flat/uniform
  compression history (fresh AI export) or localized inpainting.
- **`metadata_inspection`** — EXIF inspection; flags known AI-tool
  signatures in the `Software` field, rewards genuine camera make/model.
- **`temporal_consistency`** (video) — frame-to-frame flicker analysis.
- **`synthetic_speech_audio`** (video soundtrack) — extracts audio via ffmpeg
  and scores TTS/vocoder-like cues (spectral flatness, pitch stability, silence
  floor) **in parallel** with the visual/frame pipeline.
- **`clip_semantic_probe`** — wired up but intentionally inert until a
  trained real-vs-AI probe head is attached (see below); demonstrates how
  to slot in CLIP/ViT/UniversalFakeDetect-style learned classifiers.

### Adding a real trained model

1. Create `app/detectors/image/my_model.py` (or `detectors/video/`)
   implementing `BaseDetector`.
2. Load weights lazily inside `load()`; set `available` to reflect whether
   deps/weights are actually present.
3. Add an instance to `IMAGE_DETECTORS` / `VIDEO_DETECTORS` in
   `app/detectors/registry.py`. Nothing else changes — the API, fusion
   logic, dashboard, and extension all consume detectors polymorphically.

**MFAD-Net** (paper architecture) is already wired as detector `mfad_net`:
EfficientNet+FFT visual branch, Wav2Vec2+MFCC audio branch, GAT metadata,
CMAF transformer fusion, and TSDD drift detection. Train with:

```bash
pip install -r training/requirements.txt
python -m training.mfad_net.scripts.train --config training/mfad_net/configs/smoke.yaml
cp training/exports/mfad_net/mfad_net_best.pth backend/models/mfad_net/
```

See `training/mfad_net/README.md` for full-dataset training
(FaceForensics++ / DFDC / WildDeepfake).

### Score fusion (`app/core/fusion.py`)

Confidence-weighted average (not majority vote), with an **agreement
bonus**: detectors that agree with each other push confidence up, and
disagreement pulls it down — correlated independent signals are stronger
evidence than any single loud detector. Failed detectors are excluded from
fusion entirely rather than counted as neutral.

### Running locally

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate

# Heuristic detectors only (fast, no torch/transformers):
pip install -r requirements-lite.txt

# Full stack incl. optional CLIP slot + GPU support:
pip install -r requirements.txt

# ffmpeg/ffprobe are required for video (system package, not pip):
#   macOS:  brew install ffmpeg
#   Ubuntu: sudo apt install ffmpeg

uvicorn app.main:app --reload
# -> http://localhost:8000/docs
```

Run tests: `pytest tests/ -v` (validated — 4/4 passing against the
heuristic detector stack, including a synthetic-image round trip through
the full pipeline).

### Docker

```bash
docker compose up --build
```

Starts Postgres, Redis, the backend (port 8000), and the dashboard (port
5173). GPU inference: uncomment the `deploy.resources.reservations.devices`
block in `docker-compose.yml` (requires the NVIDIA Container Toolkit) and
swap the backend's base image for a CUDA runtime image.

---

## 2. React Dashboard (`/dashboard`)

Vite + React + Tailwind. Three input modes (upload image, upload video,
paste a direct media URL) against the same backend the extension uses —
built for demos, debugging, and visually comparing detector output.
Includes a per-detector comparison table, evidence list, and a
frame-by-frame timeline for video (thumbnail + per-frame AI-probability
strip, seismograph-style).

```bash
cd dashboard
npm install
npm run dev
# -> http://localhost:5173, expects backend at http://localhost:8000
#    (override with VITE_API_BASE_URL)
```

Verified: `npm run build` completes cleanly (production bundle, 158KB JS /
14KB CSS gzip-friendly).

---

## 3. Browser Extension (`/extension`) — the primary product

Manifest V3. Supports **Instagram** and **YouTube** (watch pages + Shorts).
Injects a small "🔍 Analyze" control into the corner of each detected post;
clicking it messages the background service worker, which calls the backend
and returns probability/confidence/evidence rendered in a compact Shadow-DOM
overlay (fully style-isolated from the host page).

### Load unpacked (local dev)

1. `chrome://extensions` (or `edge://extensions`) → enable **Developer mode**.
2. **Load unpacked** → select the `extension/` folder.
3. Click the extension icon → confirm the backend URL (default
   `http://localhost:8000`) and that detectors show "online".
4. Visit instagram.com or youtube.com — an "🔍 Analyze" button appears on
   detected media.

### How media is obtained

- **Images**: the rendered `<img>`'s resolved URL is sent to
  `POST /analyze/url`; the backend downloads and analyzes it server-side.
- **Video (direct URL)**: same as above — backend auto-detects video, cuts
  frames, and scores audio in parallel.
- **Video (MSE / blob:, e.g. YouTube)**: the adapter prefers
  `captureStream()` + MediaRecorder for a short clip (video+audio) posted to
  `/analyze/video`. If recording is blocked, it seeks across the timeline,
  captures multiple canvas frames, and posts them to `/analyze/frames`
  (visual pipeline; no audio).

### Adding a new platform (roadmap step 6)

Each platform is a small, self-contained adapter
(`content_scripts/platforms/<name>.js`) implementing:

```js
{
  name: "platform_id",
  postSelector: "css selector matching each post container",
  findMediaElement(postEl) { /* return the <img>/<video> to analyze */ },
  async extractMedia(mediaEl) { /* return { kind: "url"|"frame"|"clip"|"frames", ... } */ },
}
```

Stub adapters with selector notes and TODOs already exist for **X**,
**Reddit**, **Facebook**, and **TikTok** in `content_scripts/platforms/`.
Activating one is: fill in the selectors, then add a `content_scripts`
block to `manifest.json` (exact snippet is in each stub's file header
comment).

---

## Development roadmap

1. ✅ FastAPI backend scaffold with dummy responses
2. ✅ Real (heuristic) image detectors — frequency, noise, compression, metadata
3. ✅ React dashboard
4. ✅ Chrome extension, Instagram first
5. ✅ Video analysis + frame-by-frame breakdown, temporal-consistency detector
6. ✅ YouTube extension support + auto image/video detection + parallel audio authenticity
7. ⬜ Expand remaining platforms (X, Reddit, Facebook, TikTok) — adapters stubbed
8. ⬜ Deeper explainability: saliency/heatmap overlays on the image itself,
   persisted detector-vs-detector comparison dashboards, full analysis
   history browsing (the DB schema and `/status` groundwork are already in
   place — `AnalysisRecord` + `active_*_detectors()`)
9. ⬜ Swap in trained models (UniversalFakeDetect / DIRE / XceptionNet /
   FaceForensics++ / VideoMAE) behind the existing `clip_semantic_probe`
   and `temporal_consistency` extension points
   (and a learned audio classifier behind `synthetic_speech_audio`)

## Tech stack

FastAPI · PyTorch (optional) · Hugging Face Transformers (optional) ·
OpenCV · FFmpeg · PostgreSQL · Redis (optional) · Docker · React · Vite ·
Tailwind CSS · Manifest V3

---

## 4. Training workspace (`/training`)

Use `training/` for dataset prep, branch training, fusion training, and
artifact export. Keep the live API under `backend/` inference-only.

### Workflow

1. Prepare manifests under `training/data/`
2. Train the image branch first
3. Export branch predictions on a validation set
4. Train the fusion model on those features
5. Train the video branch after the image path is stable
6. Copy validated artifacts into `backend/models/` and point config/env vars
   at them

### Starter files

- `training/image_branch/configs/baseline.yaml`
- `training/image_branch/scripts/prepare_manifest.py`
- `training/image_branch/scripts/train_stub.py`
- `training/video_branch/configs/baseline.yaml`
- `training/video_branch/scripts/train_stub.py`
- `training/fusion/configs/baseline.yaml`
- `training/fusion/scripts/train_fusion.py`

### Backend artifact paths

The backend now exposes config hooks for:

- `SOCIAL_DETECT_IMAGE_MODEL_CHECKPOINT_PATH`
- `SOCIAL_DETECT_VIDEO_MODEL_CHECKPOINT_PATH`
- `SOCIAL_DETECT_FUSION_MODEL_PATH`

Point these at files under `backend/models/` once you have trained exports
ready.
