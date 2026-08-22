import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from course_3dgs import GaussianData, MetalRenderer

from gs_lidar import CameraIntrinsics, GaussianPlyData

EXPERIMENT_PATH = (
    Path(__file__).parents[1] / "experiments" / "03_circular_scan_with_camera.py"
)
SPEC = importlib.util.spec_from_file_location(
    "circular_scan_with_camera", EXPERIMENT_PATH
)
assert SPEC is not None and SPEC.loader is not None
circular_scan_with_camera = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(circular_scan_with_camera)


def _write_test_ply(path: Path, *, sh_levels: int = 4) -> None:
    rest_count = 3 * (sh_levels * sh_levels - 1)
    properties = [
        ("x", 1.0),
        ("y", 2.0),
        ("z", 3.0),
        ("f_dc_0", 0.1),
        ("f_dc_1", 0.2),
        ("f_dc_2", 0.3),
    ]
    properties.extend((f"f_rest_{index}", float(index)) for index in range(rest_count))
    properties.extend(
        [
            ("opacity", 0.0),
            ("scale_0", math.log(2.0)),
            ("scale_1", math.log(1.0)),
            ("scale_2", math.log(0.5)),
            # Canonical PLY order is wxyz. Identity becomes xyzw [0,0,0,1].
            ("rot_0", 1.0),
            ("rot_1", 0.0),
            ("rot_2", 0.0),
            ("rot_3", 0.0),
        ]
    )
    header = ["ply", "format ascii 1.0", "element vertex 1"]
    header.extend(f"property float {name}" for name, _ in properties)
    header.append("end_header")
    values = " ".join(str(value) for _, value in properties)
    path.write_text("\n".join([*header, values, ""]), encoding="utf-8")


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

    assert (intrinsics.width, intrinsics.height) == (1280, 960)
    assert (intrinsics.fx, intrinsics.fy) == pytest.approx((1000.0, 1000.0))
    assert intrinsics.near == pytest.approx(0.1)
    assert intrinsics.far == pytest.approx(10.0)
    assert intrinsics.cx == intrinsics.width / 2
    assert intrinsics.cy == intrinsics.height / 2


def test_ply_data_uses_xyzw_and_preserves_all_available_sh(tmp_path):
    path = tmp_path / "scene.ply"
    _write_test_ply(path, sh_levels=4)

    scene = GaussianPlyData.from_ply(path)
    renderer_data = scene.to_renderer_data("cpu")
    lidar_scene = scene.to_gaussian_cloud()

    assert scene.sh_levels == 4
    torch.testing.assert_close(
        scene.rotations_xyzw, torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    )
    assert renderer_data.sh_levels == 4
    assert renderer_data.sh_coefficient_count == 16
    assert renderer_data.sh_coefficients.shape == (1, 16, 3)
    torch.testing.assert_close(
        renderer_data.sh_coefficients[0, 0], torch.tensor([0.1, 0.2, 0.3])
    )
    # f_rest is channel-major in canonical Inria PLY files.
    torch.testing.assert_close(
        renderer_data.sh_coefficients[0, 1], torch.tensor([0.0, 15.0, 30.0])
    )
    torch.testing.assert_close(
        renderer_data.sh_coefficients[0, 15], torch.tensor([14.0, 29.0, 44.0])
    )
    torch.testing.assert_close(
        renderer_data.sigma,
        torch.diag(torch.tensor([4.0, 1.0, 0.25]))[None],
        atol=1e-6,
        rtol=0,
    )
    # The old LiDAR data model remains wxyz for compatibility.
    torch.testing.assert_close(
        lidar_scene.rotations, torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    )


def test_ply_data_infers_lower_complete_sh_level(tmp_path):
    path = tmp_path / "scene_l2.ply"
    _write_test_ply(path, sh_levels=3)

    scene = GaussianPlyData.from_ply(path)

    assert scene.sh_levels == 3
    assert scene.to_renderer_data().sh_coefficients.shape == (1, 9, 3)


def test_create_metal_renderer_uses_course_renderer_and_flips_fy(monkeypatch):
    captured = {}

    class FakeRenderer:
        def __init__(self, data, **kwargs):
            captured["data"] = data
            captured.update(kwargs)

    monkeypatch.setattr(circular_scan_with_camera, "MetalRenderer", FakeRenderer)
    data = object()
    intrinsics = CameraIntrinsics(32, 24, 20.0, 21.0, 16.0, 12.0, 0.2, 5.0)

    circular_scan_with_camera.create_metal_renderer(data, intrinsics)

    assert captured["data"] is data
    assert captured["H"] == 24
    assert captured["W"] == 32
    assert captured["fx"] == 20.0
    assert captured["fy"] == -21.0
    assert captured["near"] == 0.2
    assert captured["far"] == 5.0


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


def test_initialize_rerun_passes_ply_xyzw_without_reordering(monkeypatch):
    logged = {}

    class GaussianSplats3D:
        def __init__(self, centers, *, scales, quaternions, colors):
            self.centers = centers
            self.scales = scales
            self.quaternions = quaternions
            self.colors = colors

    class Archetype:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    rerun = SimpleNamespace(
        GaussianSplats3D=GaussianSplats3D,
        LineStrips3D=Archetype,
        ViewCoordinates=SimpleNamespace(
            RIGHT_HAND_X_UP="x-up",
            RIGHT_HAND_X_DOWN="x-down",
            RIGHT_HAND_Y_UP="y-up",
            RIGHT_HAND_Y_DOWN="y-down",
            RIGHT_HAND_Z_UP="z-up",
            RIGHT_HAND_Z_DOWN="z-down",
        ),
        blueprint=SimpleNamespace(
            Blueprint=lambda value: value,
            Horizontal=lambda *values: values,
            Spatial3DView=Archetype,
            Spatial2DView=Archetype,
        ),
        init=lambda *args, **kwargs: None,
        send_blueprint=lambda *args, **kwargs: None,
        log=lambda path, value, **kwargs: logged.setdefault(path, value),
    )
    monkeypatch.setitem(sys.modules, "rerun", rerun)
    scene = GaussianPlyData(
        positions=torch.zeros((1, 3)),
        scale_raw=torch.zeros((1, 3)),
        rotations_xyzw=torch.tensor([[0.1, 0.2, 0.3, 0.9]]),
        opacity_raw=torch.zeros(1),
        f_dc=torch.zeros((1, 3)),
        f_rest=None,
    )

    circular_scan_with_camera.initialize_rerun(scene)

    torch.testing.assert_close(
        torch.from_numpy(logged["world/scene/gaussians"].quaternions),
        torch.tensor([[0.1, 0.2, 0.3, 0.9]]),
    )


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
def test_course_metal_renderer_smoke():
    device = torch.device("mps")
    data = GaussianData(
        positions=torch.tensor([[0.0, 0.0, 2.0]], device=device),
        sh_coefficients=torch.tensor([[[8.0, -8.0, -8.0]]], device=device),
        opacity_raw=torch.tensor([math.log(9.0)], device=device),
        sigma=torch.diag(torch.tensor([0.0625, 0.0625, 0.0625], device=device))[None],
        sh_levels=1,
    )
    renderer = MetalRenderer(
        data,
        H=32,
        W=32,
        fx=24.0,
        fy=-24.0,
        cx=16.0,
        cy=16.0,
        near=0.1,
        far=10.0,
    )

    image = renderer.render(torch.eye(4, device=device))
    torch.mps.synchronize()

    assert image.shape == (32, 32, 3)
    assert torch.isfinite(image).all()
    assert image[16, 16, 0].item() > image[16, 16, 1].item()
