from __future__ import annotations

import time
import uuid
from typing import Optional

from app.core import media_utils
from app.core.fusion import fuse
from app.core.schemas import (
    AnalysisResponse, Classification, FrameResult, MediaType,
)
from app.db.models import AnalysisRecord
from app.db.session import save_record
from app.detectors.image.metadata_detector import MetadataDetector
from app.detectors.registry import active_image_detectors, active_video_detectors


async def analyze_image_bytes(raw: bytes, source: str, platform: str = "unknown") -> AnalysisResponse:
    start = time.perf_counter()
    image = media_utils.decode_image_bytes(raw)

    detectors = active_image_detectors()
    results = []
    for detector in detectors:
        detector.ensure_loaded()
        if isinstance(detector, MetadataDetector):
            detector.set_raw_bytes(raw)
        results.append(detector.analyze_image(image))

    ai_probability, confidence, classification, evidence = fuse(results)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    response = AnalysisResponse(
        request_id=str(uuid.uuid4()),
        media_type=MediaType.image,
        source=source,
        ai_probability=ai_probability,
        confidence=confidence,
        classification=classification,
        evidence=evidence,
        detector_results=results,
        metadata={"width": image.shape[1], "height": image.shape[0], "platform": platform},
        processing_time_ms=elapsed_ms,
    )

    await _persist(response, platform)
    return response


async def analyze_video_bytes(raw: bytes, source: str, platform: str = "unknown") -> AnalysisResponse:
    start = time.perf_counter()
    frames, timestamps = media_utils.extract_video_frames(raw)
    probe = media_utils.probe_video_metadata(raw)

    detectors = active_video_detectors()
    all_results = []
    per_frame_probs = {i: [] for i in range(len(frames))}

    for detector in detectors:
        detector.ensure_loaded()
        if detector.name == "temporal_consistency":
            all_results.extend(detector.analyze_video_frames(frames, timestamps))
            continue
        # Per-frame detectors: run on each sampled frame, track per-frame scores
        frame_results = detector.analyze_video_frames(frames, timestamps)
        all_results.extend(frame_results)
        if len(frame_results) == len(frames):
            for i, fr in enumerate(frame_results):
                if fr.error is None:
                    per_frame_probs[i].append(fr.ai_probability)
        elif len(frame_results) == 1 and frame_results[0].error is None:
            for i in range(len(frames)):
                per_frame_probs[i].append(frame_results[0].ai_probability)

    ai_probability, confidence, classification, evidence = fuse(all_results)

    frame_breakdown = []
    for i, ts in enumerate(timestamps):
        probs = per_frame_probs.get(i, [])
        avg_prob = sum(probs) / len(probs) if probs else ai_probability
        frame_breakdown.append(FrameResult(
            frame_index=i,
            timestamp_seconds=round(ts, 2),
            ai_probability=round(avg_prob, 4),
            confidence=round(confidence, 4),
            thumbnail_base64=media_utils.thumbnail_base64(frames[i]),
        ))

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    duration = None
    if probe.get("format", {}).get("duration"):
        duration = float(probe["format"]["duration"])

    response = AnalysisResponse(
        request_id=str(uuid.uuid4()),
        media_type=MediaType.video,
        source=source,
        ai_probability=ai_probability,
        confidence=confidence,
        classification=classification,
        evidence=evidence,
        detector_results=all_results,
        frame_results=frame_breakdown,
        metadata={"platform": platform, "duration_seconds": duration, "sampled_frames": len(frames)},
        processing_time_ms=elapsed_ms,
    )

    await _persist(response, platform)
    return response


async def _persist(response: AnalysisResponse, platform: str) -> None:
    record = AnalysisRecord(
        id=response.request_id,
        media_type=response.media_type.value,
        source=response.source,
        platform=platform,
        ai_probability=response.ai_probability,
        confidence=response.confidence,
        classification=response.classification.value,
        detector_results=[r.model_dump(mode="json") for r in response.detector_results],
        evidence=[e.model_dump(mode="json") for e in response.evidence],
        metadata_json=response.metadata,
        processing_time_ms=response.processing_time_ms,
    )
    await save_record(record)
