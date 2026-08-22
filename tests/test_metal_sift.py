import pytest
import torch

from metal_sift import MetalSiftDetector, SiftDebugStats, SiftFeatures


def test_sift_features_count() -> None:
    features = SiftFeatures(
        keypoints_xy=torch.empty((7, 2)),
        responses=torch.empty(7),
        scales=torch.empty(7),
        orientations=torch.empty(7),
    )
    assert features.count == 7


def test_sift_debug_stats_near_cutoff_fraction() -> None:
    stats = SiftDebugStats(
        candidate_count=900,
        per_level_candidate_counts=(300, 220, 160, 100, 70, 50),
        pyramid_preselected_count=700,
        selected_count=512,
        strongest_abs_response=0.2,
        cutoff_abs_response=0.01,
        first_rejected_abs_response=0.00999,
        boundary_gap=0.00001,
        near_cutoff_count=64,
    )
    assert stats.near_cutoff_fraction == pytest.approx(0.125)


def test_detector_rejects_invalid_feature_count_before_device_check() -> None:
    with pytest.raises(ValueError, match="num_features must be positive"):
        MetalSiftDetector(num_features=0)


@pytest.mark.skipif(
    not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"),
    reason="custom Metal kernels require Apple MPS and torch.mps.compile_shader",
)
def test_metal_sift_smoke() -> None:
    detector = MetalSiftDetector(num_features=32, debug=True)
    image = torch.rand((64, 64, 3), device="mps", dtype=torch.float32)
    features = detector.detect(image)
    torch.mps.synchronize()

    assert features.count <= 32
    assert features.keypoints_xy.shape == (features.count, 2)
    assert features.responses.shape == (features.count,)
    assert features.scales.shape == (features.count,)
    assert features.orientations.shape == (features.count,)
    assert features.keypoints_xy.device.type == "mps"
    assert features.responses.device.type == "mps"
    assert features.debug is not None
    assert len(features.debug.per_level_candidate_counts) == 6
    assert features.debug.candidate_count >= features.count
