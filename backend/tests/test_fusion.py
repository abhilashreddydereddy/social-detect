from app.core.fusion import fuse
from app.core.schemas import DetectorResult, Evidence, EvidenceCategory


def _result(name: str, prob: float, conf: float = 0.9, summary: str | None = None) -> DetectorResult:
    evidence = []
    if summary:
        evidence = [
            Evidence(
                category=EvidenceCategory.semantic,
                summary=summary,
                score=prob,
                weight=1.0,
                detector=name,
            )
        ]
    return DetectorResult(
        detector=name,
        ai_probability=prob,
        confidence=conf,
        evidence=evidence,
    )


def test_fuse_learned_model_outweighs_heuristics():
    fused, confidence, classification, _ = fuse(
        [
            _result("image_branch_cifake", 0.92),
            _result("frequency_artifact_fft", 0.1),
            _result("sensor_noise_residual", 0.12),
            _result("compression_ela", 0.15),
        ]
    )
    assert fused > 0.78
    assert confidence > 0.2
    assert classification.value in {
        "Likely AI Generated",
        "Possibly Manipulated",
        "Likely Authentic",
        "Inconclusive",
    }


def test_fuse_without_learned_model_still_averages():
    fused, _, _, _ = fuse(
        [
            _result("frequency_artifact_fft", 0.8),
            _result("sensor_noise_residual", 0.2),
        ]
    )
    assert 0.4 < fused < 0.6


def test_fuse_evidence_is_cifake_only_when_present():
    _, _, _, evidence = fuse(
        [
            _result("image_branch_cifake", 0.9, summary="CIFake-trained image classifier scored this still."),
            _result(
                "frequency_artifact_fft",
                0.8,
                summary="Detected periodic, grid-like peaks in the frequency spectrum.",
            ),
            _result("sensor_noise_residual", 0.7, summary="Sensor noise looks synthetic."),
        ]
    )
    assert len(evidence) == 1
    assert evidence[0].detector == "image_branch_cifake"
    assert "CIFake" in evidence[0].summary
    assert all("grid" not in e.summary.lower() for e in evidence)
