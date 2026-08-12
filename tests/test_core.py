import math
import sys
from types import SimpleNamespace

import pytest
import torch

from gs_lidar import (
    CpuLidarTracer,
    GaussianCloud,
    LidarConfig,
    LidarPose,
    MetalLidarTracer,
    build_bvh,
    gaussian_aabbs,
    generate_rays,
)
from gs_lidar.gaussian_geometry import ray_gaussian_peaks
from gs_lidar.rerun_viewer import visualize


def cloud(x, opacity):
    n = len(x)
    return GaussianCloud(
        torch.tensor([[v, 0.0, 0.0] for v in x]),
        torch.ones(n, 3),
        torch.tensor([[1.0, 0, 0, 0]] * n),
        torch.tensor(opacity),
    )


def pose():
    return LidarPose(torch.zeros(3), torch.tensor([1.0, 0, 0, 0]))


def config(threshold=0.5):
    return LidarConfig(
        azimuth_min=0,
        azimuth_max=2 * math.pi,
        elevation_min=0,
        elevation_max=0,
        azimuth_samples=1,
        elevation_samples=1,
        near=0.1,
        far=20,
        accumulated_alpha_threshold=threshold,
    )


def test_coordinate_convention_and_seam():
    c = LidarConfig(
        azimuth_min=0,
        azimuth_max=2 * math.pi,
        elevation_min=0,
        elevation_max=0,
        azimuth_samples=4,
        elevation_samples=1,
    )
    _, d = generate_rays(pose(), c)
    torch.testing.assert_close(
        d,
        torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [-1, 0, 0], [0, -1.0, 0]]),
        atol=1e-6,
        rtol=0,
    )


def test_cumulative_opacity_and_storage_order():
    scene = cloud([4.90, 5, 5.08], [0.2, 0.25, 0.3])
    lo, hi = gaussian_aabbs(scene.means, scene.scales, scene.rotations)
    scan = CpuLidarTracer().trace(scene, pose(), config(), build_bvh(lo, hi, 1))
    assert scan.hit_mask.item() and scan.gaussian_ids.item() == 2
    torch.testing.assert_close(scan.accumulated_alpha, torch.tensor([[0.58]]))
    perm = torch.tensor([2, 0, 1])
    shuffled = GaussianCloud(
        scene.means[perm],
        scene.scales[perm],
        scene.rotations[perm],
        scene.opacities[perm],
    )
    lo, hi = gaussian_aabbs(shuffled.means, shuffled.scales, shuffled.rotations)
    other = CpuLidarTracer().trace(shuffled, pose(), config(), build_bvh(lo, hi, 1))
    torch.testing.assert_close(other.ranges, scan.ranges)


def test_transparent_foreground_does_not_stop_ray():
    scene = cloud([2, 5], [0.05, 0.9])
    scan = CpuLidarTracer(False).trace(scene, pose(), config(), None)
    assert scan.gaussian_ids.item() == 1 and scan.ranges.item() == pytest.approx(5)


def test_cutoff_is_not_a_surface_hit():
    scene = GaussianCloud(
        torch.tensor([[5.0, 3.0, 0.0]]),
        torch.ones(1, 3),
        torch.tensor([[1.0, 0, 0, 0]]),
        torch.tensor([1.0]),
    )
    scan = CpuLidarTracer(False).trace(scene, pose(), config(0.5), None)
    assert (
        not scan.hit_mask.item()
    )  # q=9 participates, but alpha=exp(-4.5) is below threshold


def test_bvh_matches_bruteforce():
    torch.manual_seed(2)
    scene = cloud([2, 3, 4, 6], [0.1, 0.2, 0.3, 0.8])
    lo, hi = gaussian_aabbs(scene.means, scene.scales, scene.rotations)
    a = CpuLidarTracer(False).trace(scene, pose(), config(), None)
    b = CpuLidarTracer().trace(scene, pose(), config(), build_bvh(lo, hi, 2))
    assert torch.equal(a.hit_mask, b.hit_mask)
    assert torch.equal(a.gaussian_ids, b.gaussian_ids)
    torch.testing.assert_close(a.ranges, b.ranges)


def test_rotated_anisotropic_peak():
    q = torch.tensor([[math.sqrt(0.5), 0, 0, math.sqrt(0.5)]])
    means = torch.tensor([[5.0, 1.0, 0.0]])
    t, d = ray_gaussian_peaks(
        torch.zeros(1, 3),
        torch.tensor([[1.0, 0, 0]]),
        means,
        torch.tensor([[3.0, 0.5, 1.0]]),
        q,
    )
    assert t.item() == pytest.approx(5)
    assert d.item() == pytest.approx(4)


def test_rerun_visualizes_native_gaussian_splats(monkeypatch):
    logged = {}
    log_static = {}

    class GaussianSplats3D:
        def __init__(self, centers, *, scales, quaternions, colors):
            self.centers = centers
            self.scales = scales
            self.quaternions = quaternions
            self.colors = colors

    class Points3D:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class Arrows3D:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    view_coordinates = SimpleNamespace(
        RIGHT_HAND_X_UP="x-up",
        RIGHT_HAND_X_DOWN="x-down",
        RIGHT_HAND_Y_UP="y-up",
        RIGHT_HAND_Y_DOWN="y-down",
        RIGHT_HAND_Z_UP="z-up",
        RIGHT_HAND_Z_DOWN="z-down",
    )

    def log(path, value, *, static=False):
        logged[path] = value
        log_static[path] = static

    rerun = SimpleNamespace(
        GaussianSplats3D=GaussianSplats3D,
        Points3D=Points3D,
        Arrows3D=Arrows3D,
        ViewCoordinates=view_coordinates,
        init=lambda *args, **kwargs: None,
        log=log,
    )
    monkeypatch.setitem(sys.modules, "rerun", rerun)
    scene = GaussianCloud(
        means=torch.tensor([[1.0, 2.0, 3.0]]),
        scales=torch.tensor([[0.5, 1.0, 1.5]]),
        rotations=torch.tensor([[0.5, 0.1, 0.2, 0.3]]),
        opacities=torch.tensor([0.5]),
        colors=torch.tensor([[1.0, 0.25, 0.0]]),
    )
    scan = SimpleNamespace(valid_points=lambda: torch.tensor([[4.0, 5.0, 6.0]]))

    visualize(scene, pose(), scan, up_axis="-Y", axis_length=2.0)

    splats = logged["world/scene/gaussians"]
    assert splats.centers.tolist() == [[1.0, 2.0, 3.0]]
    assert splats.scales.tolist() == [[0.5, 1.0, 1.5]]
    assert splats.quaternions.tolist() == pytest.approx([[0.1, 0.2, 0.3, 0.5]])
    assert splats.colors.tolist() == [[255, 63, 0, 127]]
    assert logged["world/lidar/returns"].kwargs["radii"] == 0.01
    assert logged["world"] == "y-down"
    assert log_static["world"] and log_static["world/axes"]
    axes = logged["world/axes"].kwargs
    assert axes["vectors"] == [
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
    ]
    assert axes["labels"] == ["+X", "+Y", "+Z"]


def test_rerun_rejects_non_positive_axis_length(monkeypatch):
    monkeypatch.setitem(sys.modules, "rerun", SimpleNamespace())
    with pytest.raises(ValueError, match="axis_length must be positive"):
        visualize(None, None, None, axis_length=0)


@pytest.mark.skipif(
    not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"),
    reason="requires Apple MPS compile_shader",
)
def test_metal_smoke_and_cpu_parity():
    scene = cloud([2, 3, 4], [0.2, 0.25, 0.3])
    lo, hi = gaussian_aabbs(scene.means, scene.scales, scene.rotations)
    bvh = build_bvh(lo, hi, 1)
    cpu = CpuLidarTracer().trace(scene, pose(), config(), bvh)
    gpu = MetalLidarTracer().trace(
        scene.to("mps"), pose().to("mps"), config(), bvh.to("mps")
    )
    torch.testing.assert_close(gpu.ranges.cpu(), cpu.ranges)
    assert torch.equal(gpu.hit_mask.cpu(), cpu.hit_mask)
