# Social Detect

A browser-first AI media authenticity platform. Social Detect estimates
whether an image or video is likely AI-generated or manipulated — and
always returns a **probability**, a **confidence score**, and
**explainable evidence**, never a definitive "real" or "fake" verdict.

```
Chrome/Edge Extension  ──┐
                          ├──▶  FastAPI Backend ──▶ Detector Registry ──▶ Score Fusion ──▶ JSON
React Dashboard (demo) ──┘                          (image + video + audio)
```

**Supported extension sites:** Instagram · YouTube (watch + Shorts)

---

## Setup & Installation

### Prerequisites

| Tool | Windows | Linux |
|------|---------|-------|
| Git | [git-scm.com](https://git-scm.com/download/win) | `sudo apt install git` |
| Python 3.11+ | [python.org](https://www.python.org/downloads/) (tick **Add to PATH**) | `sudo apt install python3 python3-venv python3-pip` |
| Node.js 18+ | [nodejs.org](https://nodejs.org/) | [NodeSource](https://github.com/nodesource/distributions) or `sudo apt install nodejs npm` |
| FFmpeg | `winget install Gyan.FFmpeg` | `sudo apt install ffmpeg` |
| Chrome or Edge | — | — |

Optional (GPU training / MFAD-Net): NVIDIA driver + CUDA-enabled PyTorch.

Verify tools:

**Windows (PowerShell)**
```powershell
git --version
py -3.12 --version
node --version
npm --version
ffmpeg -version
```

**Linux (bash)**
```bash
git --version
python3 --version
node --version
npm --version
ffmpeg -version
```

---

### 1. Clone the repository

**Windows (PowerShell)**
```powershell
git clone https://github.com/abhilashreddydereddy/social-detect.git
cd social-detect
git checkout main
git pull origin main
```

**Linux (bash)**
```bash
git clone https://github.com/abhilashreddydereddy/social-detect.git
cd social-detect
git checkout main
git pull origin main
```

---

### 2. Backend (API on port 8000)

Keep this terminal open while using the dashboard or extension.

#### Windows (PowerShell)

```powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Heuristic detectors only (fast, recommended first run):
pip install -r requirements-lite.txt

# OR full stack (Torch / transformers / MFAD support):
# pip install -r requirements.txt
# For NVIDIA GPU Torch instead of CPU:
# pip uninstall -y torch torchvision torchaudio
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

#### Linux (bash)

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate

# Heuristic detectors only (fast, recommended first run):
pip install -r requirements-lite.txt

# OR full stack (Torch / transformers / MFAD support):
# pip install -r requirements.txt
# For NVIDIA GPU Torch:
# pip uninstall -y torch torchvision torchaudio
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

#### Verify backend

Open:

- API docs: http://localhost:8000/docs  
- Health: http://localhost:8000/status  

You should see `"status": "ok"` and a list of detectors.

---

### 3. Dashboard (optional demo UI on port 5173)

Open a **second** terminal.

#### Windows (PowerShell)

```powershell
cd social-detect\dashboard
npm install
npm run dev
```

#### Linux (bash)

```bash
cd social-detect/dashboard
npm install
npm run dev
```

Open http://localhost:5173  

Override API URL if needed:
```powershell
# Windows
$env:VITE_API_BASE_URL="http://localhost:8000"
npm run dev
```
```bash
# Linux
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

---

### 4. Browser extension (primary product)

The extension captures pixels from Instagram/YouTube in the page and uploads
them to your local backend (`POST /analyze/image`). CDN URLs are **not**
sent to the server (they are usually blocked).

1. Start the **backend** (step 2) and leave it running.
2. Open Chrome/Edge → `chrome://extensions` (or `edge://extensions`).
3. Enable **Developer mode**.
4. Click **Load unpacked**.
5. Select the `social-detect/extension` folder (the folder that contains `manifest.json`).
6. Confirm the extension version in the card (current: **0.3.2**).
7. Click the extension icon → Backend URL = `http://localhost:8000` → detectors should show online.
8. Open Instagram or a YouTube `/watch` / Shorts page → click **🔍 Analyze** on the media.

If YouTube shows no button: open a watch page, click the extension popup → **Inject on this tab**.

After `git pull`, always **Remove** the extension and **Load unpacked** again so new permissions/scripts apply.

---

### 5. Quick end-to-end check

| Step | Expected |
|------|----------|
| http://localhost:8000/status | `"status": "ok"` |
| http://localhost:5173 | Dashboard loads |
| Extension popup | Detectors online |
| Instagram / YouTube Analyze | Overlay with probability + evidence |

DevTools console on the social page should log something like:
```text
[Social Detect] sending to backend: { kind: "frame", method: "canvas"|"fetch"|"viewport", ... }
```

---

### 6. Optional — MFAD-Net training

Smoke train (CPU or GPU) from the **repo root** (not inside `training/`):

#### Windows (PowerShell)

```powershell
cd social-detect
py -3.12 -m venv training\.venv
.\training\.venv\Scripts\Activate.ps1
pip install -r training\requirements.txt

# GPU (optional):
# pip uninstall -y torch torchvision torchaudio
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

python -m training.mfad_net.scripts.train --config training/mfad_net/configs/smoke.yaml
copy training\exports\mfad_net\mfad_net_best.pth backend\models\mfad_net\
```

#### Linux (bash)

```bash
cd social-detect
python3 -m venv training/.venv
source training/.venv/bin/activate
pip install -r training/requirements.txt

# GPU (optional):
# pip uninstall -y torch torchvision torchaudio
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

python -m training.mfad_net.scripts.train --config training/mfad_net/configs/smoke.yaml
cp training/exports/mfad_net/mfad_net_best.pth backend/models/mfad_net/
```

Confirm GPU:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Force CUDA in config: set `device: cuda` in `training/mfad_net/configs/smoke.yaml`.

Restart the backend after copying weights. Full-dataset training (FF++ / DFDC / WildDeepfake): see `training/mfad_net/README.md`.

> The committed / smoke checkpoint validates the architecture. For real deepfake accuracy, train on the paper datasets with `full.yaml` on GPU.

#### Image branch (CIFake) — primary still / video-frame model

```powershell
python -m training.image_branch.scripts.prepare_manifest from-tree --root path\to\CIFAKE
python -m training.image_branch.scripts.train --config training/image_branch/configs/cifake.yaml
Copy-Item training\exports\image_branch\cifake\best_model.pth backend\models\image_branch\cifake_best.pth
```

Image uploads use this checkpoint; video uploads sample frames and score them with the same model. See `training/image_branch/README.md`.

---

### 7. Optional — Docker

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows) or Docker Engine (Linux).

```bash
docker compose up --build
```

- Backend: http://localhost:8000  
- Dashboard: http://localhost:5173  

---

## Repository layout

```
social-detect/
├── backend/      FastAPI: detectors, fusion, REST API
├── dashboard/    React + Tailwind demo UI
├── extension/    Manifest V3 Chrome/Edge extension (Instagram + YouTube)
├── training/     MFAD-Net and other training code
└── docker-compose.yml
```

## Design principle: no verdicts

Every response carries a disclaimer (`AnalysisResponse.disclaimer`): this is a
probabilistic estimate, not proof. Labels are hedged (`Likely AI Generated`,
`Possibly Manipulated`, `Likely Authentic`, `Inconclusive`).

---

## Backend API

| Endpoint | Purpose |
|---|---|
| `POST /analyze/image` | multipart image upload |
| `POST /analyze/video` | video → frames as images + parallel audio scoring |
| `POST /analyze/media` | auto image/video detection |
| `POST /analyze/frames` | multipart JPEG frame sequence |
| `POST /analyze/url` | direct media URL (best-effort) |
| `GET /status` | health + active detectors |

### Detectors (registry)

Heuristic (no weights required): frequency FFT, noise residual, compression ELA,
metadata, temporal consistency, synthetic speech audio.

Learned: `image_branch_cifake` (loads `backend/models/image_branch/cifake_best.pth`
or the training export path) is the primary still/video-frame scorer after CIFake
training. Optional: `mfad_net`, `clip_semantic_probe`.

Add a model: implement `BaseDetector`, register in `app/detectors/registry.py`.

### Tests

```bash
# from backend/ with venv active
cd backend
pytest tests/ -v
# if needed: PYTHONPATH=. pytest tests/ -v
```

---

## Extension — how media is extracted

1. **Canvas** — draw the visible `<img>` / `<video>`  
2. **Page fetch** — download the media URL in-page (cookies/referrer)  
3. **Viewport crop** — `captureVisibleTab` + crop to the element  

Captured JPEG → `POST /analyze/image` on your local backend.

---

## Development roadmap

1. ✅ FastAPI backend scaffold  
2. ✅ Heuristic image detectors  
3. ✅ React dashboard  
4. ✅ Chrome extension (Instagram + YouTube)  
5. ✅ Video frames + temporal detector + parallel audio  
6. ✅ Auto media detection + pixel extraction for extension  
7. ✅ MFAD-Net architecture + smoke training path  
8. ⬜ Remaining platforms (X, Reddit, Facebook, TikTok)  
9. ⬜ Full-dataset MFAD training + GradCAM/LIME explainability UI  

## Tech stack

FastAPI · PyTorch (optional) · Transformers (optional) · OpenCV · FFmpeg ·
PostgreSQL · Redis (optional) · Docker · React · Vite · Tailwind · MV3

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: training` | Run train commands from **repo root**, not `training/` |
| Extension still old version | Remove extension → Load unpacked again after `git pull` |
| Backend unreachable in popup | Start uvicorn; URL must be `http://localhost:8000` |
| Analyze fails / no pixels | Wait for media to load; check console for `[Social Detect] sending to backend` |
| `torch.cuda.is_available() == False` | Install CUDA wheel (`cu124` / `cu121`), not the default CPU build |
| Video/audio path errors | Install system `ffmpeg` / `ffprobe` and restart backend |
| PowerShell `Activate.ps1` blocked | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
