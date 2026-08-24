# Multi-stage-friendly single stage; swap the base image for a CUDA runtime
# image (e.g. nvidia/cuda:12.4.1-runtime-ubuntu22.04 + system Python) if you
# need GPU inference in the container. CPU-only works out of the box.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg provides both ffmpeg and ffprobe, used for video frame extraction
# and metadata probing.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# For a lighter CPU-only image without the heavy optional ML stack, swap in
# requirements-lite.txt here.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
