"""
Score fusion.

Combines multiple DetectorResults into one AnalysisResponse-ready
(ai_probability, confidence, classification, evidence) tuple.

Design choices, made explicit because this is the part most likely to be
scrutinized/tuned over time:

1. Weighted average, not majority vote. Each detector's contribution is
   weighted by (detector.default_weight * this_result.confidence). When a
   learned checkpoint (CIFake / MFAD-Net) is present, heuristic detectors
   are scaled down so they explain the score rather than outvoting it.
   The public evidence list includes only CIFake (or MFAD-Net) copy when a
   learned detector succeeded, so heuristic FFT/grid text does not overlap.
2. Detectors that errored (result.error is not None) are excluded from
   fusion entirely, not counted as neutral -- a failed detector should not
   silently pull the average toward uncertainty in a way that masks the
   failure. It's surfaced in detector_results/error and in /status.
3. Overall confidence is NOT just the average of per-detector confidences;
   it also rewards agreement between detectors (low variance across their
   individual ai_probability estimates increases confidence, high
   disagreement decreases it) since correlated agreement across
   independent signals is stronger evidence than any single detector.
4. Classification buckets are intentionally conservative near the
   midpoint ("Inconclusive"/"Possibly Manipulated") -- the system is
   designed to avoid false certainty in either direction.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from app.core.schemas import Classification, DetectorResult, Evidence


LEARNED_DETECTORS = {"image_branch_cifake", "mfad_net", "asvspoof5_audio"}
PRIMARY_EVIDENCE_DETECTOR = "image_branch_cifake"
# Heuristics stay in the mix for evidence, but must not outvote a loaded
# checkpoint when one is present.
_HEURISTIC_SCALE_WHEN_LEARNED = 0.1


def _detector_weights() -> dict[str, float]:
    try:
        from app.detectors.registry import detector_weight_map

        return detector_weight_map()
    except Exception:  # noqa: BLE001
        return {}


def fuse(results: List[DetectorResult]) -> Tuple[float, float, Classification, List[Evidence]]:
    usable = [r for r in results if r.error is None]

    if not usable:
        return 0.5, 0.0, Classification.inconclusive, []

    named_weights = _detector_weights()
    has_learned = any(r.detector in LEARNED_DETECTORS for r in usable)
    fused_weights = []
    for r in usable:
        base = float(named_weights.get(r.detector, 0.5))
        if has_learned and r.detector not in LEARNED_DETECTORS:
            base *= _HEURISTIC_SCALE_WHEN_LEARNED
        fused_weights.append(max(r.confidence, 0.05) * max(base, 0.01))
    weights = np.array(fused_weights)
    probs = np.array([r.ai_probability for r in usable])

    fused_prob = float(np.average(probs, weights=weights))

    # Agreement bonus: low spread across detectors -> boost confidence;
    # high spread -> penalize it. Spread is measured as weighted std dev.
    weighted_mean = fused_prob
    variance = float(np.average((probs - weighted_mean) ** 2, weights=weights))
    spread_penalty = float(np.clip(variance * 4.0, 0.0, 0.6))  # up to -0.6

    base_confidence = float(np.average([r.confidence for r in usable], weights=weights))
    fused_confidence = float(np.clip(base_confidence - spread_penalty + 0.1 * len(usable) / len(results), 0.0, 1.0))

    classification = _classify(fused_prob, fused_confidence)

    evidence_sources = _evidence_sources(usable)
    all_evidence: List[Evidence] = []
    for r in evidence_sources:
        all_evidence.extend(r.evidence)
    all_evidence.sort(key=lambda e: e.weight * abs(e.score - 0.5), reverse=True)

    return fused_prob, fused_confidence, classification, all_evidence


def _evidence_sources(usable: List[DetectorResult]) -> List[DetectorResult]:
    """When CIFake ran, only its descriptions are shown (no heuristic overlap)."""
    cifake = [r for r in usable if r.detector == PRIMARY_EVIDENCE_DETECTOR]
    if cifake:
        return cifake
    learned = [r for r in usable if r.detector in LEARNED_DETECTORS]
    if learned:
        return learned
    return usable


def _classify(prob: float, confidence: float) -> Classification:
    if confidence < 0.2:
        return Classification.inconclusive
    if prob >= 0.75:
        return Classification.likely_ai_generated
    if prob >= 0.55:
        return Classification.possibly_manipulated
    if prob <= 0.3:
        return Classification.likely_authentic
    return Classification.inconclusive
