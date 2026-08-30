# Build from the repository root:
#   docker build -f training/Dockerfile .
#
# Override BASE_IMAGE for GPU training, e.g.:
#   docker build -f training/Dockerfile --build-arg BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime .
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY training/requirements.txt training/requirements.txt
RUN pip install --no-cache-dir -r training/requirements.txt

COPY training/ training/

# Smoke config generates synthetic data on first run when manifests are missing.
CMD ["python", "-m", "training.mfad_net.scripts.train", "--config", "training/mfad_net/configs/smoke.yaml"]
