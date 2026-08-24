from __future__ import annotations

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np

from app.core import media_utils
from app.core.fusion import fuse
from app.core.schemas import (
    AnalysisResponse, AudioResult, DetectorResult, FrameResult, MediaType,
)
from app.db.models import AnalysisRecord
from app.db.session import save_record
from app.detectors.image.metadata_detector import MetadataDetector
from app.detectors.registry import (
    active_audio_detectors, active_image_detectors, active_video_detectors,
)

# Shared pool so frame cutting and audio extraction can run truly in parallel.
_MEDIA_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="media-pipeline")


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
        metadata={"width": image.shape[1], "height": image.shape[0], "platform": platform, "input_kind": "image"},
        processing_time_ms=elapsed_ms,
    )

    await _persist(response, platform)
    return response


async def analyze_video_bytes(raw: bytes, source: str, platform: str = "unknown") -> AnalysisResponse:
    """Cut video into frames, run image/temporal detectors, and analyze audio in parallel."""
    start = time.perf_counter()
    loop = asyncio.get_running_loop()

    # Parallel media prep: frame sampling + audio extraction + container probe.
    frames_fut = loop.run_in_executor(_MEDIA_POOL, media_utils.extract_video_frames, raw)
    audio_fut = loop.run_in_executor(_MEDIA_POOL, media_utils.extract_audio_waveform, raw)
    probe_fut = loop.run_in_executor(_MEDIA_POOL, media_utils.probe_video_metadata, raw)

    (frames, timestamps), (waveform, sample_rate, audio_error), probe = await asyncio.gather(
        frames_fut, audio_fut, probe_fut,
    )

    # Parallel detector branches: visual (frames-as-images + temporal) vs audio.
    visual_fut = loop.run_in_executor(
        _MEDIA_POOL, _run_visual_detectors, frames, timestamps,
    )
    audio_fut2 = loop.run_in_executor(
        _MEDIA_POOL, _run_audio_detectors, waveform, sample_rate, audio_error,
    )
    visual_pack, audio_pack = await asyncio.gather(visual_fut, audio_fut2)

    all_results: List[DetectorResult] = list(visual_pack["results"])
    all_results.extend(audio_pack["results"])
    per_frame_probs = visual_pack["per_frame_probs"]
    audio_result: AudioResult = audio_pack["audio_result"]

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
        audio_result=audio_result,
        metadata={
            "platform": platform,
            "input_kind": "video",
            "duration_seconds": duration,
            "sampled_frames": len(frames),
            "audio_analyzed": bool(audio_result.available and audio_result.error is None),
            "pipeline": "frames_as_images+audio_parallel",
        },
        processing_time_ms=elapsed_ms,
    )

    await _persist(response, platform)
    return response


async def analyze_frame_sequence(
    frames: list,
    source: str = "frames",
    platform: str = "unknown",
    timestamps: list[float] | None = None,
) -> AnalysisResponse:
    """Run the visual video pipeline on pre-extracted frames (no audio track)."""
    start = time.perf_counter()
    if not frames:
        raise ValueError("No frames provided")

    if timestamps is None or len(timestamps) != len(frames):
        timestamps = [float(i) for i in range(len(frames))]

    loop = asyncio.get_running_loop()
    visual_pack = await loop.run_in_executor(
        _MEDIA_POOL, _run_visual_detectors, frames, timestamps,
    )
    all_results: List[DetectorResult] = list(visual_pack["results"])
    # Explicit placeholder so callers see that audio was not available.
    audio_placeholder = DetectorResult(
        detector="synthetic_speech_audio",
        ai_probability=0.5,
        confidence=0.0,
        evidence=[],
        error="No audio track in frame-only upload",
    )
    all_results.append(audio_placeholder)
    per_frame_probs = visual_pack["per_frame_probs"]

    ai_probability, confidence, classification, evidence = fuse(all_results)

    frame_breakdown = []
    for i, ts in enumerate(timestamps):
        probs = per_frame_probs.get(i, [])
        avg_prob = sum(probs) / len(probs) if probs else ai_probability
        frame_breakdown.append(FrameResult(
            frame_index=i,
            timestamp_seconds=round(float(ts), 2),
            ai_probability=round(avg_prob, 4),
            confidence=round(confidence, 4),
            thumbnail_base64=media_utils.thumbnail_base64(frames[i]),
        ))

    elapsed_ms = int((time.perf_counter() - start) * 1000)
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
        audio_result=AudioResult(
            available=False,
            error="No audio track in frame-only upload",
            detector="synthetic_speech_audio",
        ),
        metadata={
            "platform": platform,
            "input_kind": "video_frames",
            "sampled_frames": len(frames),
            "audio_analyzed": False,
            "pipeline": "frames_as_images",
        },
        processing_time_ms=elapsed_ms,
    )
    await _persist(response, platform)
    return response


async def analyze_media_bytes(
    raw: bytes,
    source: str,
    platform: str = "unknown",
    content_type: str | None = None,
    filename: str | None = None,
) -> AnalysisResponse:
    """Auto-detect image vs video, then route to the matching pipeline."""
    kind = media_utils.detect_media_kind(raw, content_type=content_type, filename=filename)
    if kind == "video":
        return await analyze_video_bytes(raw, source=source, platform=platform)
    return await analyze_image_bytes(raw, source=source, platform=platform)


def _run_visual_detectors(
    frames: List[np.ndarray],
    timestamps: List[float],
) -> dict:
    detectors = active_video_detectors()
    all_results: List[DetectorResult] = []
    per_frame_probs = {i: [] for i in range(len(frames))}

    for detector in detectors:
        detector.ensure_loaded()
        if detector.name == "temporal_consistency":
            all_results.extend(detector.analyze_video_frames(frames, timestamps))
            continue
        # Per-frame detectors: treat sampled frames as images.
        frame_results = detector.analyze_video_frames(frames, timestamps)
        all_results.extend(frame_results)
        if len(frame_results) == len(frames):
            for i, fr in enumerate(frame_results):
                if fr.error is None:
                    per_frame_probs[i].append(fr.ai_probability)
        elif len(frame_results) == 1 and frame_results[0].error is None:
            for i in range(len(frames)):
                per_frame_probs[i].append(frame_results[0].ai_probability)

    return {"results": all_results, "per_frame_probs": per_frame_probs}


def _run_audio_detectors(
    waveform: Optional[np.ndarray],
    sample_rate: int,
    audio_error: Optional[str],
) -> dict:
    detectors = active_audio_detectors()
    results: List[DetectorResult] = []

    if audio_error or waveform is None:
        err = audio_error or "No audio track available"
        for detector in detectors:
            detector.ensure_loaded()
            results.append(detector.safe_result(err))
        audio_result = AudioResult(
            available=False,
            error=err,
            sample_rate=sample_rate,
            detector=detectors[0].name if detectors else None,
        )
        return {"results": results, "audio_result": audio_result}

    duration_seconds = float(len(waveform) / float(sample_rate)) if sample_rate else None
    primary: Optional[DetectorResult] = None
    for detector in detectors:
        detector.ensure_loaded()
        result = detector.analyze_audio(waveform, sample_rate)
        results.append(result)
        if primary is None:
            primary = result

    if primary is None:
        audio_result = AudioResult(available=False, error="No audio detectors registered", sample_rate=sample_rate)
    elif primary.error:
        audio_result = AudioResult(
            available=False,
            error=primary.error,
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            detector=primary.detector,
        )
    else:
        audio_result = AudioResult(
            available=True,
            ai_probability=primary.ai_probability,
            confidence=primary.confidence,
            sample_rate=sample_rate,
            duration_seconds=round(duration_seconds, 3) if duration_seconds is not None else None,
            detector=primary.detector,
        )

    return {"results": results, "audio_result": audio_result}


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
