# MFAD-Net Training & Inference
# =============================
#
# Implements the architecture from the MFAD-Net project report:
# Multi-modal Fusion with Adaptive Drift Detection Network.
#
# Modules (paper §8.2):
#   models/visual_encoder.py  EfficientNet(+FFT) → 1792-d
#   models/audio_encoder.py   Wav2Vec2(+MFCC) → 808-d   [lite: 1D CNN]
#   models/meta_encoder.py    Graph Attention → 256-d
#   models/cmaf.py            Cross-Modal Attention Fusion (4-head transformer)
#   models/tsdd.py            Temporal-Semantic Drift Detector (MMD + GAN memory)
#   models/mfad_net.py        Full assembled model
#
# Quick start (CPU smoke train — no FF++/DFDC download required):
#
#   cd /path/to/social-detect
#   python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
#   pip install -r training/requirements.txt
#   python -m training.mfad_net.scripts.train --config training/mfad_net/configs/smoke.yaml
#   python -m training.mfad_net.scripts.evaluate --config training/mfad_net/configs/smoke.yaml
#   cp training/exports/mfad_net/mfad_net_best.pth backend/models/mfad_net/
#
# Full training (GPU + real datasets — paper Phase 1–4):
#   1. Download FaceForensics++, DFDC subset, WildDeepfake
#   2. Run MTCNN face crops → training/data/mfad_net/processed/
#   3. Build CSV manifests (columns: sample_id,path,label,source,split)
#   4. python -m training.mfad_net.scripts.train --config training/mfad_net/configs/full.yaml
#
# Backend: detector `mfad_net` auto-loads the checkpoint when present and is
# fused with the existing heuristic detectors via the registry.
