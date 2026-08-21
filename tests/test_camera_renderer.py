import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gs_lidar import CameraIntrinsics, GaussianCloud, MetalGaussianRenderer
from gs_lidar.metal_renderer import _read_kernel, gaussian_covariances

EXPERIMENT_PATH = (
    Path(__file__).parents[1] / "experiments" / "03_circular_scan_with_camera.py"
)
SPEC = importlib.util.spec_from_file_location(
    "circular_scan_with_camera", EXPERIMENT_PATH
)
assert SPEC is not None and SPEC.loader is not None
circular_scan_with_camera = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(circular_scan_with_camera)


def test_camera_looks_toward_orbit_center_with_downward_pitch():
    pitch = math.radians(circular_scan_with_camera.CAMERA_DOWNWARD_PITCH_DEGREES)
    for angle in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2):
        c2w = circular_scan_with_camera.camera_c2w(torch.device("cpu"), angle)
        expected_forward = torch.tensor(
            [
                -math.cos(pitch) * math.cos(angle),
                -math.sin(pitch),
                -math.cos(pitch) * math.sin(angle),
            ],
            dtype=torch.float32,
        )

        torch.testing.assert_close(
            c2w[:3, 3],
            circular_scan_with_camera.orbit_position(torch.device("cpu"), angle),
        )
        assert c2w[1, 1].item() > 0
        torch.testing.assert_close(c2w[:3, 2], expected_forward, atol=1e-6, rtol=0)
        torch.testing.assert_close(c2w[:3, :3].T @ c2w[:3, :3], torch.eye(3))
        assert torch.det(c2w[:3, :3]).item() == pytest.approx(1.0)


def test_camera_intrinsics_use_requested_clipping_planes():
    intrinsics = circular_scan_with_camera.create_camera_intrinsics()

    assert intrinsics.near == pytest.approx(0.1)
    assert intrinsics.far == pytest.approx(10.0)
    assert intrinsics.cx == intrinsics.width / 2
    assert intrinsics.cy == intrinsics.height / 2


def test_camera_scene_converts_wxyz_once_for_both_renderers():
    scene_wxyz = GaussianCloud(
        means=torch.zeros((1, 3)),
        scales=torch.ones((1, 3)),
        rotations=torch.tensor([[0.9, 0.1, 0.2, 0.3]]),
        opacities=torch.tensor([0.5]),
    )

    scene_xyzw = circular_scan_with_camera.to_camera_scene_xyzw(scene_wxyz)

    torch.testing.assert_close(
        scene_xyzw.rotations, torch.tensor([[0.1, 0.2, 0.3, 0.9]])
    )
    assert scene_xyzw.means is scene_wxyz.means


def test_covariance_uses_course_xyzw_quaternions():
    half_sqrt = math.sqrt(0.5)
    scene = GaussianCloud(
        means=torch.zeros((1, 3)),
        scales=torch.tensor([[2.0, 1.0, 0.5]]),
        rotations=torch.tensor([[0.0, 0.0, half_sqrt, half_sqrt]]),
        opacities=torch.tensor([0.5]),
    )

    covariance = gaussian_covariances(scene)

    torch.testing.assert_close(
        covariance,
        torch.diag(torch.tensor([1.0, 4.0, 0.25]))[None],
        atol=1e-6,
        rtol=0,
    )


def test_metal_kernel_resources_are_packaged_next_to_renderer():
    assert "kernel void project_gaussians" in _read_kernel("gaussian_setup.metal")
    assert "kernel void tile_rasterizer_kernel" in _read_kernel("tile_rasterizer.metal")


def test_camera_frustum_is_a_pyramid_pointing_forward():
    c2w = circular_scan_with_camera.camera_c2w(torch.device("cpu"), 0.0)
    intrinsics = circular_scan_with_camera.create_camera_intrinsics()

    strips = circular_scan_with_camera.camera_frustum_line_strips(
        c2w, intrinsics, depth=0.5
    )

    assert len(strips) == 5
    assert strips[0].shape == (5, 3)
    torch.testing.assert_close(strips[0][0], strips[0][-1])
    for side, corner in zip(strips[1:], strips[0][:-1]):
        torch.testing.assert_close(side[0], c2w[:3, 3])
        torch.testing.assert_close(side[1], corner)
        assert torch.dot(side[1] - side[0], c2w[:3, 2]).item() == pytest.approx(0.5)


def test_log_frame_records_synchronized_frustum_and_image(monkeypatch):
    times = []
    logged = []

    class Archetype:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    rerun = SimpleNamespace(
        LineStrips3D=Archetype,
        Image=Archetype,
        set_time=lambda timeline, *, sequence: times.append((timeline, sequence)),
        log=lambda path, value: logged.append((path, value)),
    )
    monkeypatch.setitem(sys.modules, "rerun", rerun)

    c2w = circular_scan_with_camera.camera_c2w(torch.device("cpu"), 0.0)
    circular_scan_with_camera.log_frame(
        4,
        c2w,
        circular_scan_with_camera.create_camera_intrinsics(),
        torch.ones((2, 3, 3)),
    )

    assert times == [("frame", 4)]
    assert [path for path, _ in logged] == [
        "world/camera/frustum",
        "camera/render",
    ]
    assert len(logged[0][1].args[0]) == 5
    assert logged[-1][1].args[0].dtype.name == "uint8"


@pytest.mark.skipif(
    not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"),
    reason="requires Apple MPS compile_shader",
)
def test_metal_renderer_smoke():
    scene = GaussianCloud(
        means=torch.tensor([[0.0, 0.0, 2.0]], device="mps"),
        scales=torch.tensor([[0.25, 0.25, 0.25]], device="mps"),
        rotations=torch.tensor([[0.0, 0.0, 0.0, 1.0]], device="mps"),
        opacities=torch.tensor([0.9], device="mps"),
        colors=torch.tensor([[1.0, 0.0, 0.0]], device="mps"),
    )
    intrinsics = CameraIntrinsics(32, 32, 24.0, 24.0, 16.0, 16.0)

    image = MetalGaussianRenderer(scene).render(torch.eye(4, device="mps"), intrinsics)

    assert image.shape == (32, 32, 3)
    assert torch.isfinite(image).all()
    assert image[16, 16, 0].item() > 0.5
